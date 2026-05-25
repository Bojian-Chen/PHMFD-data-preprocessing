from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.io import loadmat
from scipy.signal import resample as scipy_resample

from data_scripts.fewshot import (
    sample_balanced_fraction_indices,
    sample_balanced_shot_indices,
)


class SDUSTMatFinetuneProcessor:
    subset_dir_name = None
    label_map = {}

    def __init__(
        self,
        dataset_name,
        raw_dir,
        save_dir,
        sample_time=0.1,
        sampling_frequency=25600,
        norm_method="none",
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
        fewshot_seed=42,
        fewshot_shots=None,
    ):
        self.dataset_name = dataset_name
        self.raw_dir = Path(raw_dir)
        self.save_dir = Path(save_dir)
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
            split_shape = self.save_indices(split_name, samples, labels, indices)
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
                indices,
            )

        shapes = ", ".join(
            f"{split_name}={shape}"
            for split_name, shape in split_shapes.items()
        )
        print(f"{self.dataset_name}: saved finetune splits {shapes} to {self.save_dir}")

    def save_indices(self, split_name, samples, labels, indices):
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
        return tuple(split_samples.shape)

    def load_samples(self):
        subset_dir = self.resolve_subset_dir()
        samples = []
        labels = []
        groups = []

        for label_name, label in self.label_map.items():
            label_dir = subset_dir / label_name
            if not label_dir.exists():
                raise FileNotFoundError(f"SDUST label folder does not exist: {label_dir}")
            for mat_path in sorted(label_dir.glob("*.mat")):
                parsed = self.parse_file(mat_path)
                if parsed is None:
                    continue
                label_key, condition = parsed
                signal = read_sdust_signal(mat_path)
                windows = segment_signal(signal, self.window_size)
                if len(windows) == 0:
                    continue
                samples.append(windows)
                labels.append(np.full(len(windows), label, dtype=np.int64))
                groups.extend([(label_key, condition)] * len(windows))

        if not samples:
            raise RuntimeError(f"No {self.dataset_name} samples were found under {subset_dir}")

        return (
            np.concatenate(samples, axis=0).astype(np.float32),
            np.concatenate(labels, axis=0),
            groups,
        )

    def resolve_subset_dir(self):
        candidates = (
            self.raw_dir / self.subset_dir_name,
            self.raw_dir,
        )
        for candidate in candidates:
            if candidate.exists() and any(
                (candidate / label_name).exists()
                for label_name in self.label_map
            ):
                return candidate
        raise FileNotFoundError(
            f"Cannot find SDUST subset '{self.subset_dir_name}' under {self.raw_dir}"
        )

    def parse_file(self, mat_path):
        raise NotImplementedError


def is_variable_condition(tokens):
    joined = " ".join(tokens).lower()
    return "~" in joined or "-" in joined or "flu" in joined


def read_sdust_signal(path):
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    try:
        values = mat["Signal"].y_values.values
    except KeyError as exc:
        raise KeyError(f"Cannot find Signal.y_values.values in {path}") from exc
    data = np.asarray(values, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] != 6:
        raise ValueError(f"Expected SDUST signal with 6 channels in {path}, got {data.shape}")
    return data.T


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
