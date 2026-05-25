from collections import defaultdict
from pathlib import Path

import h5py
import math
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import resample as scipy_resample


DATASET_CONFIG = {
    "target": "PrepareCNC",
    "method": "prepare_dataset",
    "task": "finetune",
    "raw_folders": ("CNC",),
    "save_folder": "",
}


class PrepareCNC:
    def __init__(
        self,
        raw_dir=None,
        save_dir=None,
        sample_time=0.1,
        sampling_frequency=2000,
        norm_method="none",
        resampled_size=None,
        seed=42,
        fewshot_seed=20260504,
    ):
        self.raw_dir = Path(raw_dir) if raw_dir is not None else default_raw_dir()
        self.save_dir = Path(save_dir) if save_dir is not None else default_save_dir()
        self.sample_time = sample_time
        self.sampling_frequency = sampling_frequency
        self.window_size = int(round(sampling_frequency * sample_time))
        self.norm_method = norm_method
        self.resampled_size = resampled_size
        self.seed = seed
        self.fewshot_seed = fewshot_seed
        self.machines = ("M01", "M02", "M03")
        self.label_map = {"good": 0, "bad": 1}

    def prepare_dataset(self):
        for machine in self.machines:
            self.prepare_one_machine(machine)

    def prepare_one_machine(self, machine):
        samples, labels, groups = self.load_samples(machine)
        split_indices = split_finetune_indices(
            groups,
            seed=self.seed,
            fewshot_seed=self.fewshot_seed,
        )
        dataset_name = machine
        machine_save_dir = self.save_dir / dataset_name

        for split_name, indices in split_indices.items():
            split_samples = samples[indices]
            split_labels = labels[indices]
            split_samples = normalize_per_sample(split_samples, self.norm_method)
            split_samples = maybe_resample(split_samples, self.resampled_size)
            save_parquet(
                split_samples,
                split_labels,
                dataset_name,
                machine_save_dir / f"{split_name}.parquet",
            )

        print(
            f"{dataset_name}: saved train_1p={len(split_indices['train_1p'])}, "
            f"val={len(split_indices['val'])}, test={len(split_indices['test'])} "
            f"to {machine_save_dir}"
        )

    def prepare_CNC_dataset(self):
        self.prepare_dataset()

    def resolve_data_root(self):
        candidates = [
            self.raw_dir / "CNC_Machining-main" / "data",
            self.raw_dir / "data",
            self.raw_dir,
        ]
        for candidate in candidates:
            if all((candidate / machine).exists() for machine in self.machines):
                return candidate
        raise FileNotFoundError(f"Cannot find CNC machine folders under {self.raw_dir}")

    def load_samples(self, machine):
        data_root = self.resolve_data_root()
        all_samples = []
        all_labels = []
        all_groups = []

        machine_dir = data_root / machine
        for op_dir in sorted(machine_dir.glob("OP*")):
            if not op_dir.is_dir():
                continue
            for label_name, label in self.label_map.items():
                label_dir = op_dir / label_name
                if not label_dir.exists():
                    continue
                group_key = (label, op_dir.name)
                for h5_file in sorted(label_dir.glob("*.h5")):
                    signal = read_h5_signal(h5_file)
                    windows = segment_signal(signal, self.window_size)
                    if len(windows) == 0:
                        continue
                    all_samples.append(windows)
                    all_labels.append(np.full(len(windows), label, dtype=np.int64))
                    all_groups.extend([group_key] * len(windows))

        if not all_samples:
            raise RuntimeError(f"No CNC samples were found under {machine_dir}")

        samples = np.concatenate(all_samples, axis=0).astype(np.float32)
        labels = np.concatenate(all_labels, axis=0)
        return samples, labels, all_groups


def read_h5_signal(path):
    with h5py.File(path, "r") as h5_file:
        signal = np.asarray(h5_file["vibration_data"], dtype=np.float32)
    if signal.ndim != 2:
        raise ValueError(f"Expected 2D vibration_data in {path}, got {signal.shape}")
    return signal.T


def segment_signal(signal, window_size):
    channels, length = signal.shape
    n_windows = length // window_size
    if n_windows == 0:
        return np.empty((0, channels, window_size), dtype=np.float32)
    trimmed = signal[:, : n_windows * window_size]
    return trimmed.reshape(channels, n_windows, window_size).transpose(1, 0, 2)


def split_finetune_indices(
    groups,
    seed=42,
    fewshot_seed=42,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
    tiny_train_ratio=0.01,
):
    rng = np.random.default_rng(seed)
    grouped = defaultdict(list)
    for idx, group in enumerate(groups):
        grouped[group].append(idx)

    train_full = []
    splits = {"train_1p": [], "val": [], "test": []}
    for indices in grouped.values():
        indices = np.asarray(indices, dtype=np.int64)
        rng.shuffle(indices)
        n_total = len(indices)
        n_train_full = max(1, int(np.floor(n_total * train_ratio)))
        n_val = (
            max(1, int(np.floor(n_total * val_ratio)))
            if n_total - n_train_full > 1
            else max(0, n_total - n_train_full)
        )

        while n_train_full + n_val > n_total:
            if n_val > 0:
                n_val -= 1
            else:
                n_train_full -= 1

        train_full.extend(indices[:n_train_full])
        splits["val"].extend(indices[n_train_full : n_train_full + n_val])
        splits["test"].extend(indices[n_train_full + n_val :])

    splits["train_1p"] = sample_fewshot_train(
        train_full,
        groups,
        tiny_train_ratio,
        seed=fewshot_seed,
    )
    for split_name in splits:
        splits[split_name] = np.asarray(splits[split_name], dtype=np.int64)
    return splits


def sample_fewshot_train(train_indices, groups, fraction, seed):
    rng = np.random.default_rng(seed)
    by_condition = defaultdict(list)
    for idx in train_indices:
        _, condition = groups[int(idx)]
        by_condition[condition].append(int(idx))

    target_total = int(math.floor(len(train_indices) * fraction))
    target_total = max(target_total, len(by_condition))
    target_total = min(target_total, len(train_indices))
    target_by_condition = proportional_counts(
        {condition: len(indices) for condition, indices in by_condition.items()},
        target_total,
    )

    sampled = []
    for condition in sorted(by_condition):
        indices = np.asarray(by_condition[condition], dtype=np.int64)
        chosen_idx = rng.choice(
            len(indices), size=target_by_condition[condition], replace=False
        )
        sampled.extend(indices[chosen_idx].tolist())
    return sorted(sampled)


def proportional_counts(group_sizes, target_total):
    groups = sorted(group_sizes)
    if target_total < len(groups):
        raise ValueError(
            f"target_total={target_total} is smaller than non-empty groups={len(groups)}"
        )

    counts = {group: 1 for group in groups}
    remaining = target_total - len(groups)
    if remaining == 0:
        return counts

    total_after_min = sum(group_sizes[group] - 1 for group in groups)
    if total_after_min <= 0:
        return counts

    raw = {
        group: remaining * (group_sizes[group] - 1) / total_after_min
        for group in groups
    }
    for group in groups:
        counts[group] += int(math.floor(raw[group]))

    leftover = target_total - sum(counts.values())
    remainders = sorted(
        groups,
        key=lambda group: (raw[group] - math.floor(raw[group]), group),
        reverse=True,
    )
    for group in remainders[:leftover]:
        counts[group] += 1

    for group in groups:
        counts[group] = min(counts[group], group_sizes[group])
    return counts


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
    return Path("Raw_data") / "Finetune" / DATASET_CONFIG["raw_folders"][0]


def default_save_dir():
    save_folder = DATASET_CONFIG["save_folder"]
    if save_folder:
        return Path("Process_data") / "Finetune" / save_folder
    return Path("Process_data") / "Finetune"


Prepare_CNC = PrepareCNC


if __name__ == "__main__":
    dataset = PrepareCNC(
        sample_time=0.1,
        sampling_frequency=2000,
        norm_method="minmax",
        resampled_size=1024,
        seed=42,
        fewshot_seed=42,
    )
    dataset.prepare_dataset()
