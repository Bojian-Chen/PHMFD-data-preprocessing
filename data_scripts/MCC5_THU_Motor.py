from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import resample as scipy_resample

from data_scripts.fewshot import (
    sample_balanced_fraction_indices,
    sample_balanced_shot_indices,
)


DATASET_CONFIG = {
    "target": "MCC5THUMotorFinetuneProcessor",
    "method": "prepare_dataset",
    "task": "finetune",
    "raw_folders": ("MCC5-THU_Motor/MCC5-THU Motor_torque_circulation",),
    "save_folder": "MCC5-THU-Motor",
}


LABEL_MAP = {
    "health": 0,
    "bearing_ball_H": 1,
    "bearing_ball_L": 2,
    "bearing_inner_H": 3,
    "bearing_inner_L": 4,
    "bearing_outer_H": 5,
    "bearing_outer_L": 6,
    "bearing_outer_H_and_inner_H": 7,
    "bend": 8,
    "broken_bar": 9,
    "broken_bar_and_bearing_inner_H": 10,
    "broken_bar_and_bearing_outer_H": 11,
    "dynamic_eccentricity": 12,
    "dynamic_eccentricity_and_bearing_inner_H": 13,
    "dynamic_eccentricity_and_bearing_outer_H": 14,
    "static_eccentricity_H": 15,
    "static_eccentricity_L": 16,
    "static_eccentricity_H_and_bearing_inner_H": 17,
    "static_eccentricity_H_and_bearing_outer_H": 18,
    "voltage_unbalance_L": 19,
    "winding_H": 20,
    "winding_L": 21,
    "winding_H_and_bearing_inner_H": 22,
    "winding_H_and_bearing_outer_H": 23,
}


FILE_PATTERN = re.compile(
    r"(?P<label>.+)_torque_circulation_(?P<torque>\d+)Nm_(?P<rpm>\d+)rpm_.+"
)
SEGMENTS_SEC = ((15, 25), (65, 75))
MOTOR_VIBRATION_COLUMNS = (3, 4, 5)


class MCC5THUMotorFinetuneProcessor:
    def __init__(
        self,
        raw_dir=None,
        save_dir=None,
        sample_time=0.1,
        sampling_frequency=12800,
        norm_method="none",
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
        fewshot_seed=42,
        fewshot_shots=None,
    ):
        self.dataset_name = DATASET_CONFIG["save_folder"]
        self.raw_dir = Path(raw_dir) if raw_dir is not None else default_raw_dir()
        self.save_dir = Path(save_dir) if save_dir is not None else default_save_dir()
        self.sample_time = float(sample_time)
        self.sampling_frequency = int(sampling_frequency)
        self.window_size = int(round(self.sampling_frequency * self.sample_time))
        self.norm_method = norm_method
        self.resampled_size = (
            int(resampled_size) if resampled_size is not None else None
        )
        self.train_size = float(train_size)
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.seed = int(seed)
        self.fewshot_seed = int(fewshot_seed)
        self.fewshot_shots = (
            int(fewshot_shots) if fewshot_shots is not None else None
        )

        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.resampled_size is not None and self.resampled_size <= 0:
            raise ValueError("resampled_size must be positive.")
        if abs((self.train_size + self.val_size + self.test_size) - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0.")
        if self.fewshot_shots is not None and self.fewshot_shots <= 0:
            raise ValueError("fewshot_shots must be positive.")

    def prepare_dataset(self):
        samples, labels, groups = self.load_samples()

        if self.fewshot_shots is not None:
            split_name = f"train_{self.fewshot_shots}shot"
            indices = sample_balanced_shot_indices(
                labels,
                groups,
                self.fewshot_shots,
                self.fewshot_seed,
            )
            split_shape = self.save_indices(split_name, samples, labels, groups, indices)
            print(f"{self.dataset_name}: saved {split_name}={split_shape} to {self.save_dir}")
            return

        split_indices = split_finetune_indices(
            groups,
            self.train_size,
            self.val_size,
            self.seed,
            self.fewshot_seed,
        )

        split_shapes = {}
        for split_name, indices in split_indices.items():
            split_shapes[split_name] = self.save_indices(
                split_name,
                samples,
                labels,
                groups,
                indices,
            )

        shapes = ", ".join(
            f"{split_name}={shape}"
            for split_name, shape in split_shapes.items()
        )
        print(f"{self.dataset_name}: saved finetune splits {shapes} to {self.save_dir}")

    def save_indices(self, split_name, samples, labels, groups, indices):
        split_samples = samples[indices]
        split_labels = labels[indices]
        split_groups = [groups[int(idx)] for idx in indices]
        split_samples = normalize_per_sample(split_samples, self.norm_method)
        split_samples = maybe_resample(split_samples, self.resampled_size)
        save_parquet(
            split_samples,
            split_labels,
            self.dataset_name,
            self.save_dir / f"{split_name}.parquet",
            split_groups,
        )
        return tuple(split_samples.shape)

    def load_samples(self):
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"MCC5-THU-Motor data directory does not exist: {self.raw_dir}")

        samples = []
        labels = []
        groups = []
        for csv_path in sorted(self.raw_dir.glob("*_torque_circulation_*.csv")):
            label_key, condition = parse_mcc5_filename(csv_path.stem)
            signal = read_mcc5_signal(csv_path, self.sampling_frequency)
            windows = segment_signal(signal, self.window_size)
            if len(windows) == 0:
                continue
            samples.append(windows)
            labels.append(np.full(len(windows), LABEL_MAP[label_key], dtype=np.int64))
            groups.extend([(label_key, condition)] * len(windows))

        if not samples:
            raise RuntimeError(f"No MCC5-THU-Motor torque_circulation samples found under {self.raw_dir}")

        return (
            np.concatenate(samples, axis=0).astype(np.float32),
            np.concatenate(labels, axis=0),
            groups,
        )


def parse_mcc5_filename(file_stem):
    match = FILE_PATTERN.fullmatch(file_stem)
    if match is None:
        raise ValueError(f"Cannot infer MCC5-THU-Motor label/condition from {file_stem}")
    label_key = match.group("label")
    if label_key not in LABEL_MAP:
        raise ValueError(f"Unknown MCC5-THU-Motor label code: {label_key}")
    condition = f"{match.group('rpm')}rpm_{match.group('torque')}Nm"
    return label_key, condition


def read_mcc5_signal(path, sampling_frequency):
    segment_arrays = []
    for start_sec, end_sec in SEGMENTS_SEC:
        start = int(round(start_sec * sampling_frequency))
        nrows = int(round((end_sec - start_sec) * sampling_frequency))
        df = pd.read_csv(
            path,
            header=None,
            usecols=MOTOR_VIBRATION_COLUMNS,
            skiprows=range(start),
            nrows=nrows,
        )
        data = df.to_numpy(dtype=np.float32, copy=True)
        if len(data) != nrows:
            raise ValueError(f"MCC5-THU-Motor file {path} is shorter than requested {start_sec}-{end_sec}s segment.")
        segment_arrays.append(data.T)
    return np.concatenate(segment_arrays, axis=1)


def segment_signal(signal, window_size):
    channels, length = signal.shape
    n_windows = length // window_size
    if n_windows == 0:
        return np.empty((0, channels, window_size), dtype=np.float32)
    trimmed = signal[:, : n_windows * window_size]
    return trimmed.reshape(channels, n_windows, window_size).transpose(1, 0, 2)


def split_finetune_indices(
    groups,
    train_size,
    val_size,
    seed,
    fewshot_seed,
    tiny_train_ratio=0.01,
):
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for idx, group in enumerate(groups):
        grouped[group].append(idx)

    train_full = []
    splits = {"train": [], "train_1p": [], "val": [], "test": []}
    for indices in grouped.values():
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train = max(1, int(np.floor(n_total * train_size)))
        n_val = max(1, int(np.floor(n_total * val_size)))

        while n_train + n_val > n_total:
            if n_val > 0:
                n_val -= 1
            else:
                n_train -= 1

        train_full.extend(indices[:n_train])
        splits["val"].extend(indices[n_train : n_train + n_val])
        splits["test"].extend(indices[n_train + n_val :])

    splits["train"] = train_full
    splits["train_1p"] = sample_balanced_fraction_indices(
        train_full,
        groups,
        tiny_train_ratio,
        fewshot_seed,
    )
    for split_name in splits:
        splits[split_name] = np.asarray(splits[split_name], dtype=np.int64)
        rng.shuffle(splits[split_name])
    return splits


def normalize_per_sample(samples, norm_method):
    method = normalize_method_name(norm_method)
    if method == "none":
        return samples.astype(np.float32, copy=False)
    if method == "minmax":
        min_values = samples.min(axis=-1, keepdims=True)
        max_values = samples.max(axis=-1, keepdims=True)
        return ((samples - min_values) / (max_values - min_values + 1e-8)).astype(
            np.float32
        )
    if method == "zscore":
        mean_values = samples.mean(axis=-1, keepdims=True)
        std_values = samples.std(axis=-1, keepdims=True)
        return ((samples - mean_values) / (std_values + 1e-8)).astype(np.float32)
    raise ValueError("norm_method must be one of: none, minmax, zscore.")


def normalize_method_name(norm_method):
    return str(norm_method).lower().replace("-", "").replace("_", "")


def maybe_resample(samples, resampled_size):
    if resampled_size is None:
        return samples.astype(np.float32, copy=False)
    return scipy_resample(samples, int(resampled_size), axis=-1).astype(np.float32)


def save_parquet(samples, labels, dataset_name, save_path, groups=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "samples": samples.tolist(),
        "labels": labels.astype(np.int64).tolist(),
        "dataset": [dataset_name] * len(samples),
    }
    if groups is not None:
        data["group"] = [repr(group) for group in groups]
    df = pd.DataFrame(data)
    pq.write_table(pa.Table.from_pandas(df), save_path)


def default_raw_dir():
    return Path("Raw_data") / "Finetune" / DATASET_CONFIG["raw_folders"][0]


def default_save_dir():
    return Path("Process_data") / "Finetune" / DATASET_CONFIG["save_folder"]


if __name__ == "__main__":
    processor = MCC5THUMotorFinetuneProcessor(
        norm_method="minmax",
        resampled_size=1024,
    )
    processor.prepare_dataset()
