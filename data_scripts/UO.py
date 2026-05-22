from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.io import loadmat
from scipy.signal import resample


DATASET_CONFIG = {
    "target": "UOBearingPreprocessor",
    "method": "process",
    "task": "pretrain",
    "raw_folders": ("UO", "UniFault_rawdata/UO"),
    "save_folder": "UO",
}


class UOBearingPreprocessor:
    def __init__(
        self,
        data_root=Path("Raw_data") / "UO",
        save_root=Path("Process_Data") / "UO",
        time_interval=0.1,
        norm="none",
        sampling_frequency=200000,
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
        self.label_map = {
            "healthy": 0,
            "inner": 1,
            "outer": 2,
            "ball": 3,
            "combination": 4,
        }

        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.resampled_size is not None and self.resampled_size <= 0:
            raise ValueError("resampled_size must be positive.")
        if abs((self.train_size + self.val_size + self.test_size) - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0.")

    def process(self):
        samples, labels = self.build_samples()
        train, val, test = self.train_val_test_split(samples, labels)
        train = self.resample_dataset(self.normalize_dataset(train))
        val = self.resample_dataset(self.normalize_dataset(val))
        test = self.resample_dataset(self.normalize_dataset(test))

        self.save_root.mkdir(parents=True, exist_ok=True)
        save_parquet(
            train["samples"].squeeze(1).numpy(),
            train["labels"].numpy(),
            "train",
            self.save_root / "train.parquet",
        )
        save_parquet(
            val["samples"].squeeze(1).numpy(),
            val["labels"].numpy(),
            "val",
            self.save_root / "val.parquet",
        )
        save_parquet(
            test["samples"].squeeze(1).numpy(),
            test["labels"].numpy(),
            "test",
            self.save_root / "test.parquet",
        )

        print(f"UO pretrain saved to {self.save_root}")
        print(
            f"Train: {tuple(train['samples'].shape)}, Val: {tuple(val['samples'].shape)}, Test: {tuple(test['samples'].shape)}"
        )
        return train, val, test

    def build_samples(self):
        files = self.collect_files()
        if not files:
            raise RuntimeError(f"No UO .mat files found under {self.data_root}")

        samples = []
        labels = []
        for file_path, label in files:
            signal = self.load_mat_vibration_only(file_path)
            windows = self.subsample_channels_independently(signal)
            if windows.numel() == 0:
                continue
            samples.append(windows)
            labels.extend([label] * windows.shape[0])

        if not samples:
            raise RuntimeError(f"No UO samples loaded from {self.data_root}")
        return torch.cat(samples, dim=0).float(), torch.tensor(labels, dtype=torch.long)

    def collect_files(self):
        if not self.data_root.exists():
            raise FileNotFoundError(f"UO data directory does not exist: {self.data_root}")

        files = []
        for folder in sorted(self.data_root.iterdir()):
            if not folder.is_dir():
                continue
            label = self.label_from_folder(folder.name)
            if label is None:
                continue
            for file_path in sorted(folder.glob("*.mat")):
                if not file_path.name.startswith("._"):
                    files.append((file_path, label))
        return files

    def label_from_folder(self, folder_name):
        lower_name = folder_name.lower()
        for label_key, label in self.label_map.items():
            if label_key in lower_name:
                return label
        return None

    @staticmethod
    def load_mat_vibration_only(file_path):
        mat = loadmat(file_path)
        if "Channel_1" in mat:
            signal = mat["Channel_1"]
        else:
            keys = [key for key in mat if not key.startswith("__")]
            if not keys:
                raise KeyError(f"No signal key found in {file_path}")
            signal = mat[keys[0]]
        signal = np.asarray(signal, dtype=np.float32).reshape(-1)
        return torch.from_numpy(signal).unsqueeze(0)

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

    def train_val_test_split(self, samples, labels):
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(samples.size(0), generator=generator)
        samples = samples[indices]
        labels = labels[indices]

        total_size = samples.size(0)
        train_count = int(total_size * self.train_size)
        val_count = int(total_size * self.val_size)
        test_count = total_size - train_count - val_count

        train = {
            "samples": samples[:train_count],
            "labels": labels[:train_count],
        }
        val = {
            "samples": samples[train_count : train_count + val_count],
            "labels": labels[train_count : train_count + val_count],
        }
        test = {
            "samples": samples[
                train_count + val_count : train_count + val_count + test_count
            ],
            "labels": labels[
                train_count + val_count : train_count + val_count + test_count
            ],
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
        return {"samples": x, "labels": dataset["labels"]}

    def resample_dataset(self, dataset):
        if self.resampled_size is None:
            return dataset
        x = dataset["samples"]
        if x.shape[-1] == self.resampled_size:
            return dataset
        x = torch.from_numpy(resample(x.numpy(), self.resampled_size, axis=-1)).float()
        return {"samples": x, "labels": dataset["labels"]}

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


def save_parquet(samples, labels, dataset_name, save_path):
    data = {
        "samples": samples.tolist(),
        "labels": labels.tolist(),
        "dataset": [dataset_name] * len(samples),
    }
    table = pa.Table.from_pandas(pd.DataFrame(data))
    pq.write_table(table, save_path)


def load_parquet_data(path):
    table = pq.read_table(path)
    samples = np.array(table["samples"].to_pylist())
    labels = np.array(table["labels"].to_pylist())
    return samples, labels


if __name__ == "__main__":
    processor = UOBearingPreprocessor(
        data_root=Path("Raw_data") / "UniFault_rawdata" / "UO",
        save_root=Path("Process_Data") / "UO",
        time_interval=0.1,
        norm="minmax",
        resampled_size=1024,
    )
    processor.process()
