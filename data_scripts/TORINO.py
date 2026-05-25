from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.io import loadmat
from scipy.signal import resample


DATASET_CONFIG = {
    "target": "TorinoBearingPreprocessor",
    "method": "process",
    "task": "pretrain",
    "raw_folders": ("DIRG", "TORINO"),
    "save_folder": "TORINO",
}


class TorinoBearingPreprocessor:
    def __init__(
        self,
        data_root=Path("Raw_data") / "Pretrain" / "TORINO",
        save_root=Path("Process_data") / "Pretrain" / "TORINO",
        time_interval=0.1,
        norm="none",
        sampling_frequency=51200,
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    ) -> None:
        self.data_root = Path(data_root)
        self.save_root = Path(save_root)
        self.desired_duration_sec = float(time_interval)
        self.sampling_frequency = int(sampling_frequency)
        self.window_size = int(
            round(self.sampling_frequency * self.desired_duration_sec)
        )
        self.stride = self.window_size
        self.resampled_size = (
            int(resampled_size) if resampled_size is not None else None
        )
        self.norm_method = self._normalize_norm_name(norm)
        self.train_size = float(train_size)
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.seed = int(seed)
        self.label_codes = {f"C{i}A" for i in range(7)}

        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.resampled_size is not None and self.resampled_size <= 0:
            raise ValueError("resampled_size must be positive.")
        if abs((self.train_size + self.val_size + self.test_size) - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0.")

    def process(self):
        samples = self.build_pretrain_samples()
        train, val, test = self.train_val_test_split(samples)
        train = self.resample_dataset(self.normalize_dataset(train))
        val = self.resample_dataset(self.normalize_dataset(val))
        test = self.resample_dataset(self.normalize_dataset(test))

        self.save_root.mkdir(parents=True, exist_ok=True)
        save_parquet(
            train["samples"].squeeze(1).numpy(),
            "train",
            self.save_root / "train.parquet",
        )
        save_parquet(
            val["samples"].squeeze(1).numpy(), "val", self.save_root / "val.parquet"
        )
        save_parquet(
            test["samples"].squeeze(1).numpy(), "test", self.save_root / "test.parquet"
        )

        print(f"TORINO pretrain saved to {self.save_root}")
        print(
            f"Train: {tuple(train['samples'].shape)}, Val: {tuple(val['samples'].shape)}, Test: {tuple(test['samples'].shape)}"
        )
        return train, val, test

    def build_pretrain_samples(self):
        files = self.collect_files()
        if not files:
            raise RuntimeError(
                f"No TORINO C0A-C6A .mat files found under {self.data_root}"
            )

        samples = []
        for file_path in files:
            signal = self.load_mat_all_channels(file_path)
            windows = self.subsample_channels_independently(signal)
            if windows.numel() > 0:
                samples.append(windows)

        if not samples:
            raise RuntimeError(f"No TORINO samples loaded from {self.data_root}")
        return torch.cat(samples, dim=0).float()

    def collect_files(self):
        if not self.data_root.exists():
            raise FileNotFoundError(
                f"TORINO data directory does not exist: {self.data_root}"
            )

        files = []
        for file_path in sorted(self.data_root.glob("**/*.mat")):
            if file_path.name.startswith("._"):
                continue
            label_code = file_path.stem.split("_", 1)[0]
            if label_code in self.label_codes:
                files.append(file_path)
        return files

    @staticmethod
    def load_mat_all_channels(file_path):
        mat = loadmat(file_path)
        var_name = Path(file_path).stem
        if var_name in mat:
            signal = mat[var_name]
        else:
            keys = [key for key in mat.keys() if not key.startswith("__")]
            if not keys:
                raise KeyError(f"No signal key found in {file_path}")
            signal = mat[keys[-1]]
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 1:
            signal = signal[:, None]
        return torch.from_numpy(signal.T)

    def subsample_channels_independently(self, signal):
        channel_windows = []
        for channel_signal in signal:
            if channel_signal.numel() < self.window_size:
                continue
            windows = channel_signal.unfold(
                dimension=0, size=self.window_size, step=self.stride
            )
            channel_windows.append(windows.unsqueeze(1))
        if not channel_windows:
            return torch.empty((0, 1, self.window_size), dtype=torch.float32)
        return torch.cat(channel_windows, dim=0)

    def train_val_test_split(self, samples):
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(samples.size(0), generator=generator)
        samples = samples[indices]

        total_size = samples.size(0)
        train_count = int(total_size * self.train_size)
        val_count = int(total_size * self.val_size)
        test_count = total_size - train_count - val_count

        train = {"samples": samples[:train_count]}
        val = {"samples": samples[train_count : train_count + val_count]}
        test = {
            "samples": samples[
                train_count + val_count : train_count + val_count + test_count
            ]
        }
        return train, val, test

    def normalize_dataset(self, dataset):
        if self.norm_method == "none":
            return dataset
        x = dataset["samples"]
        if self.norm_method == "zscore":
            mean = x.mean(dim=2, keepdim=True)
            std = x.std(dim=2, keepdim=True).clamp_min(1e-8)
            x = (x - mean) / std
        elif self.norm_method == "minmax":
            min_value = x.amin(dim=2, keepdim=True)
            max_value = x.amax(dim=2, keepdim=True)
            x = (x - min_value) / (max_value - min_value + 1e-8)
        else:
            raise ValueError("norm_method must be one of: none, zscore, minmax.")
        return {"samples": x}

    def resample_dataset(self, dataset):
        if self.resampled_size is None:
            return dataset
        x = dataset["samples"]
        if x.shape[-1] == self.resampled_size:
            return dataset
        return {
            "samples": torch.from_numpy(
                resample(x.numpy(), self.resampled_size, axis=-1)
            ).float()
        }

    @staticmethod
    def _normalize_norm_name(norm_method):
        norm = str(norm_method).lower().replace("-", "").replace("_", "")
        if norm in {"none", "nonorm", "raw"}:
            return "none"
        if norm in {"zscore", "standard", "standardization"}:
            return "zscore"
        if norm in {"minmax", "minmaxnorm"}:
            return "minmax"
        raise ValueError("norm_method must be one of: none, zscore, minmax.")


def save_parquet(samples, dataset_name, save_path):
    data = {
        "samples": samples.tolist(),
        "dataset": [dataset_name] * len(samples),
    }
    table = pa.Table.from_pandas(pd.DataFrame(data))
    pq.write_table(table, save_path)


def load_parquet_data(path):
    table = pq.read_table(path)
    samples = np.array(table["samples"].to_pylist())
    labels = (
        np.array(table["labels"].to_pylist())
        if "labels" in table.column_names
        else None
    )
    return samples, labels


if __name__ == "__main__":
    processor = TorinoBearingPreprocessor(
        data_root=Path("Raw_data") / "Pretrain" / "TORINO",
        save_root=Path("Process_data") / "Pretrain" / "TORINO",
        time_interval=0.1,
        norm="minmax",
        resampled_size=1024,
    )
    processor.process()
