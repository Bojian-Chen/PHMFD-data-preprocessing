import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.signal import resample


DATASET_CONFIG = {
    "target": "FEMTO_pretrain",
    "method": "process_data",
    "task": "pretrain",
    "raw_folders": ("FEMTO",),
    "save_folder": "FEMTO",
}


class FEMTO_pretrain:
    def __init__(
        self,
        args=None,
        data_dir=Path("Raw_data") / "Pretrain" / "FEMTO",
        save_dir=Path("Process_data") / "Pretrain" / "FEMTO",
        desired_duration_sec=0.1,
        sampling_frequency=25600,
        resampled_size=None,
        norm_method="none",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    ) -> None:
        if args is not None:
            data_dir = Path(getattr(args, "raw_dir", Path("Raw_data"))) / "Pretrain" / "FEMTO"
            save_dir = (
                Path(getattr(args, "processed_dir", Path("Process_data"))) / "Pretrain" / "FEMTO"
            )
            norm_method = getattr(args, "norm_method", norm_method)
            desired_duration_sec = getattr(
                args, "desired_duration_sec", desired_duration_sec
            )
            sampling_frequency = getattr(args, "sampling_frequency", sampling_frequency)
            resampled_size = getattr(args, "resampled_size", resampled_size)
            train_size = getattr(args, "train_size", train_size)
            val_size = getattr(args, "val_size", val_size)
            test_size = getattr(args, "test_size", test_size)
            seed = getattr(args, "seed", seed)

        self.folder_path = Path(data_dir)
        self.parquet_save_path = Path(save_dir)
        self.desired_duration_sec = float(desired_duration_sec)
        self.sampling_frequency = int(sampling_frequency)
        self.window_size = int(
            round(self.sampling_frequency * self.desired_duration_sec)
        )
        self.stride = self.window_size
        self.resampled_size = (
            int(resampled_size) if resampled_size is not None else None
        )
        self.norm_method = self._normalize_norm_name(norm_method)
        self.train_size = float(train_size)
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.seed = int(seed)

        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.resampled_size is not None and self.resampled_size <= 0:
            raise ValueError("resampled_size must be positive.")
        if abs((self.train_size + self.val_size + self.test_size) - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0.")

        self.bearing_names = [
            "Bearing1_1",
            "Bearing1_2",
            "Bearing1_3",
            "Bearing1_4",
            "Bearing1_5",
            "Bearing1_6",
            "Bearing1_7",
            "Bearing2_1",
            "Bearing2_2",
            "Bearing2_3",
            "Bearing2_4",
            "Bearing2_5",
            "Bearing2_6",
            "Bearing2_7",
            "Bearing3_1",
            "Bearing3_2",
            "Bearing3_3",
        ]
        self.bearing_source_dirs = {
            "Bearing1_1": "Learning_set",
            "Bearing1_2": "Learning_set",
            "Bearing1_3": "Full_Test_Set",
            "Bearing1_4": "Full_Test_Set",
            "Bearing1_5": "Full_Test_Set",
            "Bearing1_6": "Full_Test_Set",
            "Bearing1_7": "Full_Test_Set",
            "Bearing2_1": "Learning_set",
            "Bearing2_2": "Learning_set",
            "Bearing2_3": "Full_Test_Set",
            "Bearing2_4": "Full_Test_Set",
            "Bearing2_5": "Full_Test_Set",
            "Bearing2_6": "Full_Test_Set",
            "Bearing2_7": "Full_Test_Set",
            "Bearing3_1": "Learning_set",
            "Bearing3_2": "Learning_set",
            "Bearing3_3": "Full_Test_Set",
        }

    def process_data(self):
        samples = self.build_pretrain_samples()
        train, val, test = self.train_val_test_split(samples)
        train = self.resample_dataset(self.normalize_dataset(train))
        val = self.resample_dataset(self.normalize_dataset(val))
        test = self.resample_dataset(self.normalize_dataset(test))

        self.parquet_save_path.mkdir(parents=True, exist_ok=True)
        save_parquet(
            train["samples"].squeeze(1).numpy(),
            "train",
            self.parquet_save_path / "train.parquet",
        )
        save_parquet(
            val["samples"].squeeze(1).numpy(),
            "val",
            self.parquet_save_path / "val.parquet",
        )
        save_parquet(
            test["samples"].squeeze(1).numpy(),
            "test",
            self.parquet_save_path / "test.parquet",
        )

        print(f"FEMTO pretrain saved to {self.parquet_save_path}")
        print(
            f"Train: {tuple(train['samples'].shape)}, Val: {tuple(val['samples'].shape)}, Test: {tuple(test['samples'].shape)}"
        )
        return train, val, test

    def build_pretrain_samples(self):
        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"FEMTO data directory does not exist: {self.folder_path}"
            )

        samples = []
        missing_dirs = []
        for bearing_name in self.bearing_names:
            bearing_dir = self.find_bearing_dir(bearing_name)
            if bearing_dir is None:
                missing_dirs.append(bearing_name)
                continue
            for csv_path in sorted(bearing_dir.glob("acc*.csv")):
                signal = self.read_acc_csv(csv_path)
                windows = self.subsample_channels_independently(signal)
                if windows.numel() > 0:
                    samples.append(windows)

        if missing_dirs:
            raise FileNotFoundError(f"Missing FEMTO bearing folders: {missing_dirs}")
        if not samples:
            raise RuntimeError(f"No FEMTO samples loaded from {self.folder_path}")
        return torch.cat(samples, dim=0).float()

    def find_bearing_dir(self, bearing_name):
        parent = self.bearing_source_dirs[bearing_name]
        candidate = self.folder_path / parent / bearing_name
        if candidate.exists():
            return candidate

        # Backward-compatible layout after the original script moved
        # Learning_set and Full_Test_Set bearing folders into the FEMTO root.
        direct = self.folder_path / bearing_name
        if direct.exists():
            return direct
        return None

    @staticmethod
    def read_acc_csv(csv_path):
        signal = pd.read_csv(csv_path, header=None)
        if len(signal.columns) == 1:
            signal = pd.read_csv(csv_path, header=None, delimiter=";")
        return torch.tensor(signal[[4, 5]].to_numpy(dtype=np.float32).T)

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
    processor = FEMTO_pretrain(
        data_dir=Path("Raw_data") / "Pretrain" / "FEMTO",
        save_dir=Path("Process_data") / "Pretrain" / "FEMTO",
        desired_duration_sec=0.1,
        sampling_frequency=25600,
        resampled_size=None,
        norm_method="none",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    )
    processor.process_data()
