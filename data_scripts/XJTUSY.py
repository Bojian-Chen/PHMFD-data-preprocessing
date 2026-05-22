from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.signal import resample


DATASET_CONFIG = {
    "target": "PrepareXJTUSY",
    "method": "load_prepared_dataset",
    "task": "pretrain",
    "raw_folders": ("XJTU-SY", "XJTUSY"),
    "save_folder": "XJTUSY",
}


class PrepareXJTUSY:
    def __init__(
        self,
        data_dir=Path("Raw_data") / "XJTU-SY",
        save_dir=Path("Process_Data") / "XJTUSY",
        time_interval=0.1,
        norm_method="none",
        sampling_frequency=25600,
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    ) -> None:
        self.folder_path = Path(data_dir)
        self.save_path = Path(save_dir)
        self.desired_duration_sec = float(time_interval)
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

        self.condition_name = ["35Hz12kN", "37.5Hz11kN", "40Hz10kN"]
        self.bearing_names = [
            "Bearing1_1",
            "Bearing1_2",
            "Bearing1_3",
            "Bearing1_4",
            "Bearing1_5",
            "Bearing2_1",
            "Bearing2_2",
            "Bearing2_3",
            "Bearing2_4",
            "Bearing2_5",
            "Bearing3_1",
            "Bearing3_2",
            "Bearing3_3",
            "Bearing3_4",
            "Bearing3_5",
        ]

    def load_prepared_dataset(self):
        return self.prepare_XJTUSY_dataset()

    def prepare_XJTUSY_dataset(self):
        samples = self.build_pretrain_samples()
        train, val, test = self.train_val_test_split(samples)
        train = self.resample_dataset(self.normalize_dataset(train))
        val = self.resample_dataset(self.normalize_dataset(val))
        test = self.resample_dataset(self.normalize_dataset(test))

        self.save_path.mkdir(parents=True, exist_ok=True)
        save_parquet(
            train["samples"].squeeze(1).numpy(),
            "train",
            self.save_path / "train.parquet",
        )
        save_parquet(
            val["samples"].squeeze(1).numpy(), "val", self.save_path / "val.parquet"
        )
        save_parquet(
            test["samples"].squeeze(1).numpy(), "test", self.save_path / "test.parquet"
        )

        print(f"XJTUSY pretrain saved to {self.save_path}")
        print(
            f"Train: {tuple(train['samples'].shape)}, Val: {tuple(val['samples'].shape)}, Test: {tuple(test['samples'].shape)}"
        )
        return train, val, test

    def build_pretrain_samples(self):
        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"XJTU-SY data directory does not exist: {self.folder_path}"
            )

        samples = []
        missing_dirs = []
        for condition_name in self.condition_name:
            for bearing_index in range(1, 6):
                bearing_name = f"Bearing{self.condition_name.index(condition_name) + 1}_{bearing_index}"
                bearing_dir = self.folder_path / condition_name / bearing_name
                if not bearing_dir.exists():
                    missing_dirs.append(str(bearing_dir))
                    continue
                for csv_path in self.sorted_csv_files(bearing_dir):
                    signal = self.read_acceleration_csv(csv_path)
                    windows = self.subsample_channels_independently(signal)
                    if windows.numel() > 0:
                        samples.append(windows)

        if missing_dirs:
            missing = "\n".join(missing_dirs)
            raise FileNotFoundError(f"Missing XJTU-SY bearing folders:\n{missing}")
        if not samples:
            raise RuntimeError(f"No XJTU-SY samples loaded from {self.folder_path}")
        return torch.cat(samples, dim=0).float()

    @staticmethod
    def sorted_csv_files(bearing_dir):
        def sort_key(path):
            try:
                return int(path.stem)
            except ValueError:
                return path.stem

        return sorted(bearing_dir.glob("*.csv"), key=sort_key)

    @staticmethod
    def read_acceleration_csv(csv_path):
        df = pd.read_csv(csv_path)
        return torch.tensor(df.to_numpy(dtype=np.float32).T)

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
    xjtusy_preproc = PrepareXJTUSY(
        data_dir=Path("Raw_data") / "XJTU-SY",
        save_dir=Path("Process_Data") / "XJTUSY",
        time_interval=0.1,
        norm_method="minmax",
        resampled_size=1024,
    )
    xjtusy_preproc.load_prepared_dataset()
