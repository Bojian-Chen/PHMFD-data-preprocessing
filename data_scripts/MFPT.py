from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.io import loadmat
from scipy.signal import resample


class MFPTDatasetPreprocessor:
    def __init__(
        self,
        data_root=r"H:\PHMFD_rawdata\UniFault\MFPT",
        save_root=r"H:\PHMFD_data_all\MFPT",
        time_interval=0.1,
        norm="none",
        resampled_size=1024,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    ) -> None:
        self.data_root = Path(data_root)
        self.save_root = Path(save_root)
        self.desired_duration_sec = float(time_interval)
        self.resampled_size = int(resampled_size) if resampled_size is not None else None
        self.norm_method = self._normalize_norm_name(norm)
        self.train_size = float(train_size)
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.seed = int(seed)
        self.folder_names = (
            "1 - Three Baseline Conditions",
            "2 - Three Outer Race Fault Conditions",
            "3 - Seven More Outer Race Fault Conditions",
            "4 - Seven Inner Race Fault Conditions",
        )

        if self.desired_duration_sec <= 0:
            raise ValueError("time_interval must be positive.")
        if self.resampled_size is not None and self.resampled_size <= 0:
            raise ValueError("resampled_size must be positive.")
        if abs((self.train_size + self.val_size + self.test_size) - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0.")

    def process(self):
        sample_groups = self.build_pretrain_sample_groups()
        outputs = {}
        for group_name, samples in sample_groups.items():
            train, val, test = self.train_val_test_split(samples)
            train = self.normalize_dataset(train)
            val = self.normalize_dataset(val)
            test = self.normalize_dataset(test)

            if self.resampled_size is not None:
                train = self.resample_dataset(train)
                val = self.resample_dataset(val)
                test = self.resample_dataset(test)
                group_save_root = self.save_root
            else:
                group_save_root = self.save_root / group_name

            group_save_root.mkdir(parents=True, exist_ok=True)
            save_parquet(train["samples"].squeeze(1).numpy(), "train", group_save_root / "train.parquet")
            save_parquet(val["samples"].squeeze(1).numpy(), "val", group_save_root / "val.parquet")
            save_parquet(test["samples"].squeeze(1).numpy(), "test", group_save_root / "test.parquet")

            print(f"MFPT {group_name} pretrain saved to {group_save_root}")
            print(f"Train: {tuple(train['samples'].shape)}, Val: {tuple(val['samples'].shape)}, Test: {tuple(test['samples'].shape)}")
            outputs[group_name] = (train, val, test)
        return outputs

    def build_pretrain_sample_groups(self):
        files = self.collect_files()
        if not files:
            raise RuntimeError(f"No MFPT .mat files found under {self.data_root}")

        sample_groups = {}
        for file_path in files:
            signal, sampling_frequency = self.load_mat_mfpt(file_path)
            windows = self.subsample_channels_independently(signal, sampling_frequency)
            if windows.numel() > 0:
                if self.resampled_size is not None:
                    windows = self.resample_windows(windows)
                    group_name = "resampled"
                else:
                    group_name = f"fs_{sampling_frequency}"
                sample_groups.setdefault(group_name, []).append(windows)

        if not sample_groups:
            raise RuntimeError(f"No MFPT samples loaded from {self.data_root}")
        return {group_name: torch.cat(samples, dim=0).float() for group_name, samples in sample_groups.items()}

    def collect_files(self):
        if not self.data_root.exists():
            raise FileNotFoundError(f"MFPT data directory does not exist: {self.data_root}")

        files = []
        for folder_name in self.folder_names:
            folder = self.data_root / folder_name
            if not folder.exists():
                raise FileNotFoundError(f"Missing MFPT folder: {folder}")
            files.extend(sorted(folder.glob("*.mat")))
        return files

    @staticmethod
    def load_mat_mfpt(file_path):
        mat = loadmat(file_path)
        signal_block = mat["bearing"][0][0]
        signal = None
        sampling_frequency = None
        for idx in range(len(signal_block)):
            item = signal_block[idx]
            if getattr(item, "shape", (0,))[0] >= 100:
                signal = np.asarray(signal_block[idx], dtype=np.float32)
                if signal.ndim == 1:
                    signal = signal[:, None]
            elif np.asarray(item).size == 1:
                value = np.asarray(item).reshape(-1)[0]
                if isinstance(value, np.generic):
                    value = value.item()
                if isinstance(value, (int, float, np.integer, np.floating)) and value > 1000:
                    sampling_frequency = int(round(float(value)))

        if signal is None:
            raise ValueError(f"No valid signal found in {file_path}")
        if sampling_frequency is None:
            raise ValueError(f"No sampling frequency found in {file_path}")
        return torch.from_numpy(signal.T), sampling_frequency

    def subsample_channels_independently(self, signal, sampling_frequency):
        window_size = int(round(float(sampling_frequency) * self.desired_duration_sec))
        if window_size <= 0:
            raise ValueError(f"Invalid window_size {window_size} for sampling_frequency {sampling_frequency}")
        channel_windows = []
        for channel_signal in signal:
            if channel_signal.numel() < window_size:
                continue
            windows = channel_signal.unfold(dimension=0, size=window_size, step=window_size)
            channel_windows.append(windows.unsqueeze(1))
        if not channel_windows:
            return torch.empty((0, 1, window_size), dtype=torch.float32)
        return torch.cat(channel_windows, dim=0)

    def resample_windows(self, windows):
        if windows.shape[-1] == self.resampled_size:
            return windows
        return torch.from_numpy(resample(windows.numpy(), self.resampled_size, axis=-1)).float()

    def train_val_test_split(self, samples):
        generator = torch.Generator().manual_seed(self.seed)
        indices = torch.randperm(samples.size(0), generator=generator)
        samples = samples[indices]

        total_size = samples.size(0)
        train_count = int(total_size * self.train_size)
        val_count = int(total_size * self.val_size)
        test_count = total_size - train_count - val_count

        train = {"samples": samples[:train_count]}
        val = {"samples": samples[train_count:train_count + val_count]}
        test = {"samples": samples[train_count + val_count:train_count + val_count + test_count]}
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
        x = dataset["samples"]
        if self.resampled_size is None:
            return dataset
        if x.shape[-1] == self.resampled_size:
            return dataset
        return {"samples": self.resample_windows(x)}

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
    labels = np.array(table["labels"].to_pylist()) if "labels" in table.column_names else None
    return samples, labels


if __name__ == "__main__":
    processor = MFPTDatasetPreprocessor(
        data_root=r"H:\PHMFD_rawdata\UniFault\MFPT",
        save_root=r"H:\PHMFD_data_all\UniFault\MFPT",
        time_interval=0.1,
        norm="minmax",
        resampled_size=1024,
    )
    processor.process()
