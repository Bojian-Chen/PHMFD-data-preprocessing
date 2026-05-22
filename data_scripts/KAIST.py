from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy.signal import resample


FAULT_TYPES = ["normal", "ball", "inner", "outer"]
CSV_INDICES = [0, 1, 2]
DATASET_CONFIG = {
    "target": "process_all_kaist_parts",
    "task": "pretrain",
    "raw_folders": ("KAIST",),
    "save_folder": "",
}


class KAISTProcessor:
    def __init__(
        self,
        raw_dir=Path("Raw_data") / "KAIST",
        save_path=Path("Process_Data") / "KAIST",
        time_interval=0.1,
        norm="none",
        sampling_frequency=25600,
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
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

        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.resampled_size is not None and self.resampled_size <= 0:
            raise ValueError("resampled_size must be positive.")
        if abs((self.train_size + self.val_size + self.test_size) - 1.0) > 1e-6:
            raise ValueError("train_size + val_size + test_size must equal 1.0.")

        self.csv_fault_types = FAULT_TYPES
        self.csv_indices = CSV_INDICES

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

        roots = [self.raw_dir]
        roots.extend(path for path in sorted(self.raw_dir.iterdir()) if path.is_dir())

        files = []
        missing_files = []
        selected_root = None
        for root in roots:
            candidate_files = []
            candidate_missing = []
            for fault_type in self.csv_fault_types:
                for index in self.csv_indices:
                    file_path = root / f"vibration_{fault_type}_{index}.csv"
                    if file_path.exists():
                        candidate_files.append(file_path)
                    else:
                        candidate_missing.append(str(file_path))
            if candidate_files and not candidate_missing:
                selected_root = root
                files = candidate_files
                break
            if candidate_files:
                missing_files = candidate_missing

        if not files:
            if missing_files:
                missing = "\n".join(missing_files)
                raise FileNotFoundError(f"Incomplete KAIST csv files:\n{missing}")
            raise RuntimeError(
                f"No KAIST vibration csv files found under {self.raw_dir}"
            )

        print(f"Using KAIST csv root: {selected_root}")
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


def has_complete_kaist_csv_set(root):
    root = Path(root)
    return all(
        (root / f"vibration_{fault_type}_{index}.csv").exists()
        for fault_type in FAULT_TYPES
        for index in CSV_INDICES
    )


def kaist_output_name(raw_part_dir, fallback_index):
    name = raw_part_dir.name.lower()
    digits = "".join(char for char in name if char.isdigit())
    if digits:
        return f"KAIST{int(digits)}"
    return f"KAIST{fallback_index}"


def find_kaist_parts(raw_root):
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError(f"KAIST data directory does not exist: {raw_root}")

    if has_complete_kaist_csv_set(raw_root):
        return [("KAIST1", raw_root)]

    candidate_roots = []
    for path in sorted(raw_root.rglob("*")):
        if path.is_dir() and has_complete_kaist_csv_set(path):
            candidate_roots.append(path)

    if not candidate_roots:
        raise RuntimeError(
            f"No complete KAIST vibration csv set found under {raw_root}"
        )

    parts = []
    used_names = set()
    for fallback_index, part_dir in enumerate(candidate_roots, start=1):
        output_name = kaist_output_name(part_dir, fallback_index)
        if output_name in used_names:
            output_name = f"KAIST{fallback_index}"
        used_names.add(output_name)
        parts.append((output_name, part_dir))
    return parts


def process_all_kaist_parts(
    raw_root=Path("Raw_data") / "KAIST",
    save_root=Path("Process_Data"),
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
    for output_name, part_dir in find_kaist_parts(raw_root):
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
        )
        outputs[output_name] = processor.process_data()
    return outputs


if __name__ == "__main__":
    process_all_kaist_parts(
        raw_root=Path("Raw_data") / "KAIST",
        save_root=Path("Process_Data"),
        time_interval=0.1,
        norm="none",
        sampling_frequency=25600,
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
    )
