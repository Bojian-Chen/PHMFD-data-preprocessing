import re
from pathlib import Path

from data_scripts.hust_common import HUSTTextFinetuneProcessor


DATASET_CONFIG = {
    "target": "HUSTBearingFinetuneProcessor",
    "method": "prepare_dataset",
    "task": "finetune",
    "raw_folders": ("HUST_Bearing",),
    "save_folder": "HUST_Bearing",
}


LABEL_MAP = {
    "H": 0,
    "0.5X_I": 1,
    "I": 2,
    "0.5X_O": 3,
    "O": 4,
    "0.5X_B": 5,
    "B": 6,
    "0.5X_C": 7,
    "C": 8,
}


class HUSTBearingFinetuneProcessor(HUSTTextFinetuneProcessor):
    file_suffixes = (".xls",)

    def __init__(
        self,
        raw_dir=None,
        save_dir=None,
        sample_time=0.1,
        sampling_frequency=25600,
        norm_method="none",
        resampled_size=None,
        train_size=0.6,
        val_size=0.2,
        test_size=0.2,
        seed=42,
        fewshot_seed=42,
        fewshot_shots=None,
    ):
        super().__init__(
            dataset_name=DATASET_CONFIG["save_folder"],
            raw_dir=Path(raw_dir) if raw_dir is not None else default_raw_dir(),
            save_dir=Path(save_dir) if save_dir is not None else default_save_dir(),
            sample_time=sample_time,
            sampling_frequency=sampling_frequency,
            norm_method=norm_method,
            resampled_size=resampled_size,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            seed=seed,
            fewshot_seed=fewshot_seed,
            fewshot_shots=fewshot_shots,
        )

    def parse_file_name(self, file_stem):
        match = re.fullmatch(r"(?:(0\.5X)_)?([HBIOC])_(\d+)[Hh][Zz]", file_stem)
        if not match:
            raise ValueError(f"Cannot infer HUST_Bearing label/condition from {file_stem}")

        severity, fault_code, speed = match.groups()
        label_key = f"{severity}_{fault_code}" if severity else fault_code
        if label_key not in LABEL_MAP:
            raise ValueError(f"Unknown HUST_Bearing label code: {label_key}")
        return LABEL_MAP[label_key], label_key, f"{speed}Hz"


def default_raw_dir():
    return Path("Raw_data") / "Finetune" / DATASET_CONFIG["raw_folders"][0]


def default_save_dir():
    return Path("Process_data") / "Finetune" / DATASET_CONFIG["save_folder"]


if __name__ == "__main__":
    processor = HUSTBearingFinetuneProcessor(
        norm_method="minmax",
        resampled_size=1024,
    )
    processor.prepare_dataset()
