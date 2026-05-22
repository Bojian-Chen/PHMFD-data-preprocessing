from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import resample as scipy_resample


DATASET_CONFIG = {
    "target": "IMS_FD",
    "method": "prepare_dataset",
    "task": "finetune",
    "raw_folders": ("IMS", "IMS_FD"),
    "save_folder": "IMS_FD",
}


class IMS_FD:
    def __init__(
        self,
        raw_dir=None,
        save_dir=None,
        sample_time=0.1,
        sampling_frequency=20480,
        norm_method="none",
        resampled_size=None,
        seed=42,
    ):
        self.dataset_name = DATASET_CONFIG["save_folder"]
        self.raw_dir = Path(raw_dir) if raw_dir is not None else default_raw_dir()
        self.save_dir = Path(save_dir) if save_dir is not None else default_save_dir()
        self.sample_time = sample_time
        self.sampling_frequency = sampling_frequency
        self.window_size = int(round(sampling_frequency * sample_time))
        self.norm_method = norm_method
        self.resampled_size = resampled_size
        self.seed = seed

        self.normal_range = ("2003.10.22.12.06.24", "2003.10.22.12.29.13")
        self.ir_range = ("2003.11.25.15.57.32", "2003.11.25.23.39.56")
        self.re_range = ("2003.11.25.15.57.32", "2003.11.25.23.39.56")
        self.or_range = ("2004.02.19.05.32.39", "2004.02.19.06.22.39")

    def prepare_dataset(self):
        samples, labels, groups = self.load_samples()
        split_indices = split_finetune_indices(groups, seed=self.seed)

        for split_name, indices in split_indices.items():
            split_samples = samples[indices]
            split_labels = labels[indices]
            split_samples = normalize_per_sample(split_samples, self.norm_method)
            split_samples = maybe_resample(split_samples, self.resampled_size)
            save_parquet(
                split_samples,
                split_labels,
                self.dataset_name,
                self.save_dir / f"{split_name}.parquet",
            )

        print(
            f"{self.dataset_name}: saved train_1p={len(split_indices['train_1p'])}, "
            f"val={len(split_indices['val'])}, test={len(split_indices['test'])} "
            f"to {self.save_dir}"
        )

    def load_samples(self):
        all_samples = []
        all_labels = []
        all_groups = []

        for file_path in sorted((self.raw_dir / "1st_test").iterdir()):
            if not file_path.is_file():
                continue
            file_name = file_path.name
            bearing1_1, _, bearing1_3, bearing1_4 = read_1st_file(file_path)

            if is_time_in_range(file_name, self.normal_range):
                self.add_signal(all_samples, all_labels, all_groups, bearing1_1, 0)
            if is_time_in_range(file_name, self.ir_range):
                self.add_signal(all_samples, all_labels, all_groups, bearing1_3, 1)
            if is_time_in_range(file_name, self.re_range):
                self.add_signal(all_samples, all_labels, all_groups, bearing1_4, 2)

        for file_path in sorted((self.raw_dir / "2nd_test").iterdir()):
            if not file_path.is_file():
                continue
            file_name = file_path.name
            bearing2_1, _, _, _ = read_2nd_file(file_path)

            if is_time_in_range(file_name, self.or_range):
                self.add_signal(all_samples, all_labels, all_groups, bearing2_1, 3)

        if not all_samples:
            raise RuntimeError(f"No IMS_FD samples were found under {self.raw_dir}")

        samples = np.concatenate(all_samples, axis=0).astype(np.float32)
        labels = np.concatenate(all_labels, axis=0)
        return samples, labels, all_groups

    def add_signal(self, all_samples, all_labels, all_groups, signal, label):
        signal = np.asarray(signal, dtype=np.float32).reshape(1, -1)
        windows = segment_signal(signal, self.window_size)
        if len(windows) == 0:
            return
        all_samples.append(windows)
        all_labels.append(np.full(len(windows), label, dtype=np.int64))
        all_groups.extend([(label,)] * len(windows))


def read_1st_file(path):
    signal_df = pd.read_csv(path, sep="\t", header=None)
    return (
        signal_df.iloc[:, 0].to_numpy(),
        signal_df.iloc[:, 2].to_numpy(),
        signal_df.iloc[:, 4].to_numpy(),
        signal_df.iloc[:, 6].to_numpy(),
    )


def read_2nd_file(path):
    signal_df = pd.read_csv(path, sep="\t", header=None)
    return (
        signal_df.iloc[:, 0].to_numpy(),
        signal_df.iloc[:, 1].to_numpy(),
        signal_df.iloc[:, 2].to_numpy(),
        signal_df.iloc[:, 3].to_numpy(),
    )


def is_time_in_range(time_str, time_range):
    time_format = "%Y.%m.%d.%H.%M.%S"
    current_time = datetime.strptime(time_str, time_format)
    start_time = datetime.strptime(time_range[0], time_format)
    end_time = datetime.strptime(time_range[1], time_format)
    return start_time <= current_time <= end_time


def segment_signal(signal, window_size):
    channels, length = signal.shape
    n_windows = length // window_size
    if n_windows == 0:
        return np.empty((0, channels, window_size), dtype=np.float32)
    trimmed = signal[:, : n_windows * window_size]
    return trimmed.reshape(channels, n_windows, window_size).transpose(1, 0, 2)


def split_finetune_indices(
    groups, seed=42, train_ratio=0.01, val_ratio=0.2, test_ratio=0.2
):
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for idx, group in enumerate(groups):
        grouped[group].append(idx)

    splits = {"train_1p": [], "val": [], "test": []}
    for indices in grouped.values():
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = max(1, int(np.ceil(n_total * train_ratio)))
        n_val = (
            max(1, int(np.floor(n_total * val_ratio)))
            if n_total - n_train > 1
            else max(0, n_total - n_train)
        )
        n_test = (
            max(1, int(np.floor(n_total * test_ratio)))
            if n_total - n_train - n_val > 0
            else 0
        )

        while n_train + n_val + n_test > n_total:
            if n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            else:
                n_train -= 1

        splits["train_1p"].extend(indices[:n_train])
        splits["val"].extend(indices[n_train : n_train + n_val])
        splits["test"].extend(indices[n_train + n_val : n_train + n_val + n_test])

    for split_name in splits:
        splits[split_name] = np.asarray(splits[split_name], dtype=np.int64)
        rng.shuffle(splits[split_name])
    return splits


def normalize_per_sample(samples, norm_method):
    method = normalize_method_name(norm_method)
    if method == "none":
        return samples.astype(np.float32, copy=False)
    if method == "min-max":
        min_values = samples.min(axis=-1, keepdims=True)
        max_values = samples.max(axis=-1, keepdims=True)
        return ((samples - min_values) / (max_values - min_values + 1e-8)).astype(
            np.float32
        )
    if method == "z-score":
        mean_values = samples.mean(axis=-1, keepdims=True)
        std_values = samples.std(axis=-1, keepdims=True)
        return ((samples - mean_values) / (std_values + 1e-8)).astype(np.float32)
    raise ValueError("norm_method must be one of: none, min-max, z-score")


def normalize_method_name(norm_method):
    if norm_method is None:
        return "none"
    method = str(norm_method).lower().replace("_", "-")
    if method == "minmax":
        return "min-max"
    if method == "zscore":
        return "z-score"
    return method


def maybe_resample(samples, resampled_size):
    if resampled_size is None:
        return samples.astype(np.float32, copy=False)
    return scipy_resample(samples, int(resampled_size), axis=-1).astype(np.float32)


def save_parquet(samples, labels, dataset_name, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "samples": samples.tolist(),
            "labels": labels.astype(np.int64).tolist(),
            "dataset": [dataset_name] * len(samples),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), save_path)


def default_raw_dir():
    return Path("Raw_data") / DATASET_CONFIG["raw_folders"][0]


def default_save_dir():
    return Path("Process_Data") / DATASET_CONFIG["save_folder"]


if __name__ == "__main__":
    dataset = IMS_FD(
        sample_time=0.1,
        sampling_frequency=20480,
        norm_method="none",
        resampled_size=None,
        seed=42,
    )
    dataset.prepare_dataset()
