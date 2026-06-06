from collections import defaultdict
from pathlib import Path

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
    "target": "ExampleFinetuneProcessor",
    "method": "prepare_dataset",
    "task": "finetune",
    "raw_folders": ("ExampleFinetune",),
    "save_folder": "ExampleFinetune",
}


class ExampleFinetuneProcessor:
    def __init__(
        self,
        raw_dir=Path("Raw_data") / "Finetune" / "ExampleFinetune",
        save_dir=Path("Process_data") / "Finetune" / "ExampleFinetune",
        sample_time=0.1,
        sampling_frequency=1000,
        norm_method="none",
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
        fewshot_seed=42,
        fewshot_shots=None,
    ):
        self.raw_dir = Path(raw_dir)
        self.save_dir = Path(save_dir)
        self.sample_time = float(sample_time)
        self.sampling_frequency = int(sampling_frequency)
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

        if abs(self.train_size + self.val_size + self.test_size - 1.0) > 1e-6:
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
            split_samples = samples[indices]
            split_labels = labels[indices]
            split_groups = [groups[int(idx)] for idx in indices]
            split_samples = normalize_per_sample(split_samples, self.norm_method)
            split_samples = maybe_resample(split_samples, self.resampled_size)
            split_shape = tuple(split_samples.shape)
            save_finetune_parquet(
                split_samples,
                split_labels,
                split_name,
                self.save_dir / f"{split_name}.parquet",
                split_groups,
            )
            print(f"ExampleFinetune saved {split_name}={split_shape} to {self.save_dir}")
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
            split_samples = samples[indices]
            split_labels = labels[indices]
            split_groups = [groups[int(idx)] for idx in indices]
            split_samples = normalize_per_sample(split_samples, self.norm_method)
            split_samples = maybe_resample(split_samples, self.resampled_size)
            split_shapes[split_name] = tuple(split_samples.shape)
            save_finetune_parquet(
                split_samples,
                split_labels,
                split_name,
                self.save_dir / f"{split_name}.parquet",
                split_groups,
            )

        shapes = ", ".join(
            f"{split_name}={shape}"
            for split_name, shape in split_shapes.items()
        )
        print(f"ExampleFinetune saved finetune splits {shapes} to {self.save_dir}")

    def load_samples(self):
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw directory does not exist: {self.raw_dir}")

        samples = []
        labels = []
        groups = []
        class_dirs = [path for path in sorted(self.raw_dir.iterdir()) if path.is_dir()]

        for label, class_dir in enumerate(class_dirs):
            for csv_path in sorted(class_dir.glob("*.csv")):
                signal = read_csv_signal(csv_path)
                windows = segment_signal(signal, self.window_size_from_signal(signal))
                if len(windows) == 0:
                    continue
                samples.append(windows)
                labels.append(np.full(len(windows), label, dtype=np.int64))
                groups.extend([(label, csv_path.stem)] * len(windows))

        if not samples:
            raise RuntimeError(f"No samples loaded from {self.raw_dir}")
        return (
            np.concatenate(samples, axis=0).astype(np.float32),
            np.concatenate(labels, axis=0),
            groups,
        )

    def window_size_from_signal(self, signal):
        return max(1, int(round(self.sampling_frequency * self.sample_time)))


def read_csv_signal(path):
    df = pd.read_csv(path)
    values = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
    if values.ndim != 2 or values.size == 0:
        raise ValueError(f"{path} must contain numeric signal columns.")
    return values.T


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
    splits["train_1p"] = sample_fewshot_train(
        train_full,
        groups,
        tiny_train_ratio,
        fewshot_seed,
    )
    for split_name in splits:
        splits[split_name] = np.asarray(splits[split_name], dtype=np.int64)
        rng.shuffle(splits[split_name])
    return splits


def sample_fewshot_train(train_indices, groups, fraction, seed):
    return sample_balanced_fraction_indices(train_indices, groups, fraction, seed)


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


def save_finetune_parquet(samples, labels, split_name, save_path, groups=None):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "samples": samples.tolist(),
        "labels": labels.astype(np.int64).tolist(),
        "dataset": [split_name] * len(samples),
    }
    if groups is not None:
        data["group"] = [repr(group) for group in groups]
    df = pd.DataFrame(data)
    pq.write_table(pa.Table.from_pandas(df), save_path)


if __name__ == "__main__":
    processor = ExampleFinetuneProcessor()
    processor.prepare_dataset()
