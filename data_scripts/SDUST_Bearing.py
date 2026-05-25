from pathlib import Path

from data_scripts.sdust_common import SDUSTMatFinetuneProcessor, is_variable_condition


DATASET_CONFIG = {
    "target": "SDUSTBearingFinetuneProcessor",
    "method": "prepare_dataset",
    "task": "finetune",
    "raw_folders": ("SUDST", "SDUST"),
    "save_folder": "SDUST_Bearing",
}


LABEL_MAP = {
    "NC": 0,
    "IF0.2": 1,
    "IF0.4": 2,
    "IF0.6": 3,
    "OF0.2": 4,
    "OF0.4": 5,
    "OF0.6": 6,
    "RF0.2": 7,
    "RF0.4": 8,
    "RF0.6": 9,
}


class SDUSTBearingFinetuneProcessor(SDUSTMatFinetuneProcessor):
    subset_dir_name = "轴承数据集"
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
        if len(parts) < 3:
            raise ValueError(f"Cannot infer SDUST_Bearing condition from {mat_path.name}")

        label_key, speed, load = parts[:3]
        if is_variable_condition((speed, load)):
            return None
        if speed == "1797" or load != "60":
            return None
        if label_key not in self.label_map:
            raise ValueError(f"Unknown SDUST_Bearing label code: {label_key}")
        return label_key, f"{speed}rpm_{load}N"


def default_raw_dir():
    return Path("Raw_data") / "Finetune" / DATASET_CONFIG["raw_folders"][0]


def default_save_dir():
    return Path("Process_data") / "Finetune" / DATASET_CONFIG["save_folder"]


if __name__ == "__main__":
    processor = SDUSTBearingFinetuneProcessor(
        norm_method="minmax",
        resampled_size=1024,
    )
    processor.prepare_dataset()
