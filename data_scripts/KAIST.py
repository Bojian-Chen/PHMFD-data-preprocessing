from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.signal import resample


FAULT_TYPES = ["normal", "ball", "inner", "outer"]
CSV_INDICES = [0, 1, 2]
KAIST_PARTS = (
    ("KAIST1", "part1", (0, 1, 2)),
    ("KAIST2", "part2", (3, 4)),
    ("KAIST3", "part3", (5, 6)),
)
DATASET_CONFIG = {
    "target": "process_all_kaist_parts",
    "task": "pretrain",
    "raw_folders": ("KAIST",),
    "save_folder": "",
}


class KAISTProcessor:
    def __init__(
        self,
        raw_dir=Path("Raw_data") / "Pretrain" / "KAIST",
        save_path=Path("Process_data") / "Pretrain" / "KAIST",
        time_interval=0.1,
        norm="none",
        sampling_frequency=25600,
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
        csv_indices=None,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.save_path = Path(save_path)
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
        self.csv_indices = tuple(csv_indices) if csv_indices is not None else tuple(CSV_INDICES)

        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.resampled_size is not None and self.resampled_size <= 0:
            raise ValueError("resampled_size must be positive.")
        if abs((self.train_size + self.val_size + self.test_size) - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0.")

        self.csv_fault_types = FAULT_TYPES

    def process(self):
        return self.process_data()

    def process_data(self):
        files = self.collect_csv_files()
        samples = self.build_pretrain_samples(files)
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

        print(f"KAIST pretrain saved to {self.save_path}")
        print(
            f"Train: {tuple(train['samples'].shape)}, Val: {tuple(val['samples'].shape)}, Test: {tuple(test['samples'].shape)}"
        )
        return train, val, test

    def build_pretrain_samples(self, files):
        samples = []
        for file_path in files:
            signal = self.read_signal(file_path)
            windows = self.subsample_channels_independently(signal)
            if windows.numel() > 0:
                samples.append(windows)

        if not samples:
            raise RuntimeError(f"No KAIST samples loaded from {files}")
        return torch.cat(samples, dim=0).float()

    def collect_csv_files(self):
        if not self.raw_dir.exists():
            raise FileNotFoundError(
                f"KAIST data directory does not exist: {self.raw_dir}"
            )

        files = []
        missing_files = []
        for fault_type in self.csv_fault_types:
            for index in self.csv_indices:
                file_path = self.raw_dir / f"vibration_{fault_type}_{index}.csv"
                if file_path.exists():
                    files.append(file_path)
                else:
                    missing_files.append(str(file_path))

        if missing_files:
            missing = "\n".join(missing_files)
            raise FileNotFoundError(f"Incomplete KAIST csv files:\n{missing}")

        print(f"Using KAIST csv root: {self.raw_dir}")
        return files

    def read_signal(self, file_path):
        return self.read_csv_tensor(file_path)

    @staticmethod
    def read_csv_tensor(csv_file):
        df = pd.read_csv(csv_file)
        return torch.tensor(df.values.T, dtype=torch.float32)

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


def has_complete_kaist_csv_set(root, csv_indices=CSV_INDICES):
    root = Path(root)
    return all(
        (root / f"vibration_{fault_type}_{index}.csv").exists()
        for fault_type in FAULT_TYPES
        for index in csv_indices
    )


def find_kaist_parts(raw_root):
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"KAIST data directory does not exist: {raw_root}")

    parts = []
    if has_complete_kaist_csv_set(raw_root):
        parts.append(("KAIST1", raw_root, tuple(CSV_INDICES)))

    for output_name, folder_name, csv_indices in KAIST_PARTS:
        part_dir = raw_root / folder_name
        if part_dir.exists() and has_complete_kaist_csv_set(part_dir, csv_indices):
            parts.append((output_name, part_dir, tuple(csv_indices)))

    if not parts:
        raise RuntimeError(f"No complete KAIST vibration csv set found under {raw_root}")
    return parts


def process_all_kaist_parts(
    raw_root=Path("Raw_data") / "Pretrain" / "KAIST",
    save_root=Path("Process_data") / "Pretrain",
    time_interval=0.1,
    norm="none",
    sampling_frequency=25600,
    resampled_size=None,
    train_size=0.6,
    val_size=0.2,
    test_size=0.2,
    seed=42,
):
    outputs = {}
    for output_name, part_dir, csv_indices in find_kaist_parts(raw_root):
        print(f"Processing {output_name} from {part_dir}")
        processor = KAISTProcessor(
            raw_dir=part_dir,
            save_path=Path(save_root) / output_name,
            time_interval=time_interval,
            norm=norm,
            sampling_frequency=sampling_frequency,
            resampled_size=resampled_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            csv_indices=csv_indices,
        )
        outputs[output_name] = processor.process_data()
    return outputs


if __name__ == "__main__":
    process_all_kaist_parts(
        raw_root=Path("Raw_data") / "Pretrain" / "KAIST",
        save_root=Path("Process_data") / "Pretrain",
        time_interval=0.1,
        norm="none",
        sampling_frequency=25600,
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    )
