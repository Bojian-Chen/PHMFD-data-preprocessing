from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.io import loadmat
from scipy.signal import resample


DATASET_CONFIG = {
    "target": "HITSM",
    "method": "process_data",
    "task": "pretrain",
    "raw_folders": ("HIT-SM", "HITSM"),
    "save_folder": "",
}


class HITSM:
    def __init__(
        self,
        args=None,
        data_dir=Path("Raw_data") / "Pretrain" / "HIT-SM",
        save_dir=Path("Process_data") / "Pretrain",
        desired_duration_sec=0.1,
        sampling_frequency=51200,
        resampled_size=None,
        norm_method="none",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    ) -> None:
        if args is not None:
            data_root = Path(getattr(args, "raw_dir", Path("Raw_data")))
            data_dir = data_root / "Pretrain" / "HIT-SM"
            save_dir = Path(getattr(args, "processed_dir", Path("Process_data"))) / "Pretrain"
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

        self.file_names = [
            "Normal_600.mat",
            "Normal_900.mat",
            "Normal_1200.mat",
            "IR2_600.mat",
            "IR2_900.mat",
            "IR2_1200.mat",
            "IR5_600.mat",
            "IR5_900.mat",
            "IR5_1200.mat",
            "IR8_600.mat",
            "IR8_900.mat",
            "IR8_1200.mat",
            "OR2_600.mat",
            "OR2_900.mat",
            "OR2_1200.mat",
            "OR5_600.mat",
            "OR5_900.mat",
            "OR5_1200.mat",
            "OR8_600.mat",
            "OR8_900.mat",
            "OR8_1200.mat",
        ]
        self.subsets = {
            "HITSM_self_built": "Self-built dataset",
            "HITSM_SpectraQuest": "SpectraQuest MFS dataset",
        }

    def process_data(self):
        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"HIT-SM data directory does not exist: {self.folder_path}"
            )

        outputs = {}
        for dataset_name, subset_dir in self.subsets.items():
            samples = self.build_pretrain_samples(self.folder_path / subset_dir)
            train, val, test = self.train_val_test_split(samples)
            train = self.resample_dataset(self.normalize_dataset(train))
            val = self.resample_dataset(self.normalize_dataset(val))
            test = self.resample_dataset(self.normalize_dataset(test))

            save_path = self.parquet_save_path / dataset_name
            save_path.mkdir(parents=True, exist_ok=True)
            save_parquet(
                train["samples"].squeeze(1).numpy(),
                "train",
                save_path / "train.parquet",
            )
            save_parquet(
                val["samples"].squeeze(1).numpy(), "val", save_path / "val.parquet"
            )
            save_parquet(
                test["samples"].squeeze(1).numpy(), "test", save_path / "test.parquet"
            )

            print(f"{dataset_name} pretrain saved to {save_path}")
            print(
                f"Train: {tuple(train['samples'].shape)}, Val: {tuple(val['samples'].shape)}, Test: {tuple(test['samples'].shape)}"
            )
            outputs[dataset_name] = (train, val, test)
        return outputs

    def build_pretrain_samples(self, subset_path):
        if not subset_path.exists():
            raise FileNotFoundError(
                f"HIT-SM subset directory does not exist: {subset_path}"
            )

        samples = []
        missing_files = []
        for file_name in self.file_names:
            mat_path = subset_path / file_name
            if not mat_path.exists():
                missing_files.append(str(mat_path))
                continue
            signal = self.read_mat_file(mat_path)
            windows = self.subsample_channels_independently(signal)
            if windows.numel() > 0:
                samples.append(windows)

        if missing_files:
            missing = "\n".join(missing_files)
            raise FileNotFoundError(f"Missing HIT-SM .mat files:\n{missing}")
        if not samples:
            raise RuntimeError(f"No HIT-SM samples loaded from {subset_path}")
        return torch.cat(samples, dim=0).float()

    @staticmethod
    def read_mat_file(mat_path):
        mat = loadmat(mat_path)
        keys = [key for key in mat.keys() if not key.startswith("__")]
        if not keys:
            raise KeyError(f"No signal key found in {mat_path}")
        signal = np.asarray(mat[keys[-1]], dtype=np.float32)
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
    processor = HITSM(
        data_dir=Path("Raw_data") / "Pretrain" / "HIT-SM",
        save_dir=Path("Process_data") / "Pretrain",
        desired_duration_sec=0.1,
        sampling_frequency=51200,
        resampled_size=None,
        norm_method="none",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    )
    processor.process_data()
