from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.io import loadmat
from scipy.signal import resample


DATASET_CONFIG = {
    "target": "PrepareCWRU",
    "method": "process_data",
    "task": "pretrain",
    "raw_folders": ("CWRU",),
    "save_folder": "CWRU",
    "extra_kwargs": {"stride": None},
}


class PrepareCWRU:
    def __init__(
        self,
        data_dir=Path("Raw_data") / "CWRU",
        desired_duration_sec=0.1,
        stride=None,
        norm_method="none",
        save_dir=None,
        sampling_frequency=12000,
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    ) -> None:
        self.folder_path = Path(data_dir)
        self.sampling_frequency = int(sampling_frequency)
        self.desired_duration_sec = float(desired_duration_sec)
        self.window_size = int(
            round(self.sampling_frequency * self.desired_duration_sec)
        )
        self.resampled_size = (
            int(resampled_size) if resampled_size is not None else None
        )
        self.stride = int(stride) if stride is not None else self.window_size
        self.norm_method = self._normalize_norm_name(norm_method)
        self.train_size = float(train_size)
        self.val_size = float(val_size)
        self.test_size = float(test_size)
        self.seed = int(seed)

        if self.window_size <= 0:
            raise ValueError(
                "window_size must be positive. Check sampling_frequency and desired_duration_sec."
            )
        if self.resampled_size is not None and self.resampled_size <= 0:
            raise ValueError("resampled_size must be positive.")
        if self.stride <= 0:
            raise ValueError("stride must be positive.")
        if abs((self.train_size + self.val_size + self.test_size) - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0.")

        norm_dir = {
            "none": "none_norm",
            "zscore": "z_score_norm",
            "minmax": "min_max_norm",
        }[self.norm_method]
        self.parquet_save_path = (
            Path(save_dir)
            if save_dir
            else Path("Process_Data")
            / norm_dir
            / "multi_scale"
            / f"CWRU_{self.desired_duration_sec:g}s"
        )
        self.parquet_save_path.mkdir(parents=True, exist_ok=True)

        self.healthy_files = ["97", "98", "99", "100"]
        self.inner_fault_files = [
            "105",
            "106",
            "107",
            "108",
            "169",
            "170",
            "171",
            "172",
            "209",
            "210",
            "211",
            "212",
        ]
        self.ball_fault_files = [
            "118",
            "119",
            "120",
            "121",
            "185",
            "186",
            "187",
            "188",
            "222",
            "223",
            "224",
            "225",
        ]
        self.outer_fault_files = [
            "130",
            "131",
            "132",
            "133",
            "197",
            "198",
            "199",
            "200",
            "234",
            "235",
            "236",
            "237",
        ]

        self.class_files = {
            0: self.healthy_files,
            1: self.inner_fault_files,
            2: self.ball_fault_files,
            3: self.outer_fault_files,
        }

    def process_data(self):
        samples = self.build_pretrain_samples()
        train, val, test = self.train_val_test_split(samples)
        train = self.resample_dataset(self.normalize_dataset(train))
        val = self.resample_dataset(self.normalize_dataset(val))
        test = self.resample_dataset(self.normalize_dataset(test))

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

        print(f"CWRU saved to {self.parquet_save_path}")
        print(
            f"Train: {tuple(train['samples'].shape)}, Val: {tuple(val['samples'].shape)}, Test: {tuple(test['samples'].shape)}"
        )
        return train, val, test

    def load_prepared_dataset(self, norm=None):
        if norm is not None:
            self.norm_method = self._normalize_norm_name(norm)
        return self.process_data()

    def build_pretrain_samples(self):
        if not self.folder_path.exists():
            raise FileNotFoundError(
                f"CWRU data directory does not exist: {self.folder_path}"
            )

        samples = []
        missing_files = []

        for file_ids in self.class_files.values():
            for file_id in file_ids:
                mat_path = self.folder_path / f"{file_id}.mat"
                if not mat_path.exists():
                    missing_files.append(str(mat_path))
                    continue
                mat_signal = self.load_mat(mat_path)
                channel_windows = self.subsample_channels_independently(mat_signal)
                if channel_windows.numel() > 0:
                    samples.append(channel_windows)

        if missing_files:
            missing = "\n".join(missing_files)
            raise FileNotFoundError(f"Missing CWRU .mat files:\n{missing}")
        if not samples:
            raise RuntimeError(f"No CWRU samples loaded from {self.folder_path}")

        return torch.cat(samples, dim=0).float()

    def load_mat(self, mat_path):
        mat = loadmat(mat_path)
        de_keys = [key for key in mat if key.endswith("DE_time")]
        fe_keys = [key for key in mat if key.endswith("FE_time")]
        if not de_keys or not fe_keys:
            raise KeyError(
                f"{mat_path} must contain both DE_time and FE_time channels."
            )

        de_signal = np.asarray(mat[de_keys[0]], dtype=np.float32).reshape(-1)
        fe_signal = np.asarray(mat[fe_keys[0]], dtype=np.float32).reshape(-1)
        return [torch.from_numpy(de_signal), torch.from_numpy(fe_signal)]

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

    def resample_windows(self, windows):
        resampled = resample(windows.numpy(), self.resampled_size, axis=-1)
        return torch.from_numpy(resampled).float()

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
    labels = (
        np.array(table["labels"].to_pylist())
        if "labels" in table.column_names
        else None
    )
    return samples, labels


if __name__ == "__main__":
    processor = PrepareCWRU(
        data_dir=Path("Raw_data") / "CWRU",
        desired_duration_sec=0.1,
        sampling_frequency=12000,
        resampled_size=None,
        stride=None,
        norm_method="none",
        save_dir=Path("Process_Data") / "CWRU",
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    )
    processor.process_data()
