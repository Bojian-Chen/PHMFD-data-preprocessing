from pathlib import Path

from data_scripts.sdust_common import SDUSTMatFinetuneProcessor, is_variable_condition


DATASET_CONFIG = {
    "target": "SDUSTGearFinetuneProcessor",
    "method": "prepare_dataset",
    "task": "finetune",
    "raw_folders": ("SUDST", "SDUST"),
    "save_folder": "SDUST_Gear",
}


LABEL_MAP = {
    "NC": 0,
    "太阳断裂": 1,
    "太阳点蚀": 2,
    "太阳磨损": 3,
    "行星断裂": 4,
    "行星点蚀": 5,
    "行星磨损": 6,
}


class SDUSTGearFinetuneProcessor(SDUSTMatFinetuneProcessor):
    subset_dir_name = "齿轮数据集"
    label_map = LABEL_MAP

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

    def parse_file(self, mat_path):
        parts = mat_path.stem.split()
        if len(parts) < 4:
            raise ValueError(f"Cannot infer SDUST_Gear condition from {mat_path.name}")

        label_key = mat_path.parent.name
        _fault_name, speed, load, repeat = parts[:4]
        if is_variable_condition((speed, load)):
            return None
        if load != "0.5A" or repeat != "1":
            return None
        if label_key not in self.label_map:
            raise ValueError(f"Unknown SDUST_Gear label folder: {label_key}")
        return label_key, f"{speed}rpm_{load}"


def default_raw_dir():
    return Path("Raw_data") / "Finetune" / DATASET_CONFIG["raw_folders"][0]


def default_save_dir():
    return Path("Process_data") / "Finetune" / DATASET_CONFIG["save_folder"]


if __name__ == "__main__":
    processor = SDUSTGearFinetuneProcessor(
        norm_method="minmax",
        resampled_size=1024,
    )
    processor.prepare_dataset()
