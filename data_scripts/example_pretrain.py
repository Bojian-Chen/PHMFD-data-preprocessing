from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.signal import resample as scipy_resample


DATASET_CONFIG = {
    "target": "ExamplePretrainProcessor",
    "method": "process_data",
    "task": "pretrain",
    "raw_folders": ("ExamplePretrain",),
    "save_folder": "ExamplePretrain",
}


class ExamplePretrainProcessor:
    def __init__(
        self,
        raw_dir=Path("Raw_data") / "ExamplePretrain",
        save_dir=Path("Process_Data") / "ExamplePretrain",
        sample_time=0.1,
        sampling_frequency=1000,
        norm_method="none",
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
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

        if abs(self.train_size + self.val_size + self.test_size - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0.")

    def process_data(self):
        samples = self.load_samples()
        split_indices = split_indices(
            len(samples),
            self.train_size,
            self.val_size,
            self.seed,
        )

        for split_name, indices in split_indices.items():
            split_samples = samples[indices]
            split_samples = normalize_per_sample(split_samples, self.norm_method)
            split_samples = maybe_resample(split_samples, self.resampled_size)
            save_pretrain_parquet(
                split_samples,
                split_name,
                self.save_dir / f"{split_name}.parquet",
            )

        print(f"ExamplePretrain saved to {self.save_dir}")

    def load_samples(self):
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"Raw directory does not exist: {self.raw_dir}")

        samples = []
        for csv_path in sorted(self.raw_dir.glob("*.csv")):
            signal = read_csv_signal(csv_path)
            windows = segment_signal(signal, self.window_size_from_signal(signal))
            if len(windows) > 0:
                samples.append(windows)

        if not samples:
            raise RuntimeError(f"No samples loaded from {self.raw_dir}")
        return np.concatenate(samples, axis=0).astype(np.float32)

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


def split_indices(total_size, train_size, val_size, seed):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_size)
    train_count = int(total_size * train_size)
    val_count = int(total_size * val_size)
    return {
        "train": indices[:train_count],
        "val": indices[train_count : train_count + val_count],
        "test": indices[train_count + val_count :],
    }


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


def save_pretrain_parquet(samples, split_name, save_path):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "samples": samples.tolist(),
            "dataset": [split_name] * len(samples),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), save_path)


if __name__ == "__main__":
    processor = ExamplePretrainProcessor()
    processor.process_data()
