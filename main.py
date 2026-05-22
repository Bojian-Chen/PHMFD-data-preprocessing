import argparse
import inspect
from importlib import import_module
from pathlib import Path

from data_scripts import DATASET_ALIASES, DATASET_MODULES


DEFAULT_RAW_ROOT = Path("Raw_data/UniFault_rawdata")
DEFAULT_SAVE_ROOT = Path("Process_Data")
DEFAULT_SAMPLE_TIME = 0.1
DEFAULT_NORM_METHOD = "none"
DEFAULT_RESAMPLED_SIZE = None
DEFAULT_SEED = 42
DEFAULT_FEWSHOT_SEED = 42
RESAMPLED_SIZE_SENTINEL = "__default__"

COMMON_PARAM_ALIASES = {
    "raw_path": ("raw_dir", "data_dir", "data_root", "raw_root"),
    "save_path": ("save_dir", "save_root", "save_path"),
    "sample_time": ("sample_time", "time_interval", "desired_duration_sec"),
    "norm_method": ("norm_method", "norm"),
    "resampled_size": ("resampled_size",),
    "train_size": ("train_size",),
    "val_size": ("val_size",),
    "test_size": ("test_size",),
    "seed": ("seed",),
    "fewshot_seed": ("fewshot_seed",),
}


def parse_optional_int(value):
    if value is None:
        return None
    if str(value).lower() in {"none", "null"}:
        return None
    return int(value)


def value_or_default(value, default):
    return default if value is None else value


def resampled_size_or_default(value):
    if value == RESAMPLED_SIZE_SENTINEL:
        return DEFAULT_RESAMPLED_SIZE
    return value


def first_existing_path(root, folder_names):
    root = Path(root)
    candidates = [root / folder_name for folder_name in folder_names]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def dataset_save_path(args, config):
    save_root = Path(args.save_root)
    save_folder = config.get("save_folder")
    if save_folder:
        return save_root / save_folder
    return save_root


def load_dataset_config(dataset_name):
    module = import_module(f"data_scripts.{dataset_name}")
    config = dict(module.DATASET_CONFIG)
    config["module"] = module
    return config


def unified_values(args, config):
    values = {
        "raw_path": first_existing_path(args.raw_root, config["raw_folders"]),
        "save_path": dataset_save_path(args, config),
        "sample_time": value_or_default(args.sample_time, DEFAULT_SAMPLE_TIME),
        "norm_method": value_or_default(args.norm_method, DEFAULT_NORM_METHOD),
        "resampled_size": resampled_size_or_default(args.resampled_size),
        "train_size": args.train_size,
        "val_size": args.val_size,
        "test_size": args.test_size,
        "seed": value_or_default(args.seed, DEFAULT_SEED),
    }
    if config.get("task") == "finetune":
        values["fewshot_seed"] = value_or_default(
            args.fewshot_seed,
            DEFAULT_FEWSHOT_SEED,
        )
    return values


def target_signature(target):
    if inspect.isclass(target):
        return inspect.signature(target.__init__)
    return inspect.signature(target)


def build_dataset_kwargs(config, args, target):
    values = unified_values(args, config)
    parameters = target_signature(target).parameters
    kwargs = {}

    for common_name, aliases in COMMON_PARAM_ALIASES.items():
        if common_name not in values:
            continue
        for alias in aliases:
            if alias in parameters:
                kwargs[alias] = values[common_name]
                break

    for key, value in config.get("extra_kwargs", {}).items():
        if key in parameters:
            kwargs[key] = value

    return kwargs


def run_dataset(dataset_name, args):
    config = load_dataset_config(dataset_name)
    target = getattr(config["module"], config["target"])
    kwargs = build_dataset_kwargs(config, args, target)

    if "method" not in config:
        return target(**kwargs)

    dataset = target(**kwargs)
    return getattr(dataset, config["method"])()


def normalize_dataset_names(dataset_names):
    if any(name.lower() == "all" for name in dataset_names):
        return list(DATASET_MODULES)

    normalized = []
    aliases = {name.lower(): name for name in DATASET_MODULES}
    aliases.update(DATASET_ALIASES)

    for dataset_name in dataset_names:
        key = dataset_name.lower()
        if key not in aliases:
            options = ", ".join(("all", *DATASET_MODULES))
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Choose from: {options}"
            )
        normalized.append(aliases[key])
    return normalized


def build_parser():
    parser = argparse.ArgumentParser(description="Run PHMFD dataset preprocessing.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help="Datasets to process. Use 'all' or names like CWRU PU IMS_FD.",
    )
    parser.add_argument(
        "--raw-root",
        default=DEFAULT_RAW_ROOT,
        help="Root directory containing raw dataset folders.",
    )
    parser.add_argument(
        "--save-root",
        default=DEFAULT_SAVE_ROOT,
        help="Root directory for generated parquet files.",
    )
    parser.add_argument("--sample-time", type=float, help="Window length in seconds.")
    parser.add_argument(
        "--norm-method",
        default="minmax",
        help="Normalization method: none, minmax, zscore.",
    )
    parser.add_argument(
        "--resampled-size",
        type=parse_optional_int,
        default=1024,
        help="Target sample length. Use 'none' to disable resampling.",
    )
    parser.add_argument("--train-size", type=float, default=0.6)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fewshot-seed",
        type=int,
        default=42,
        help="Seed for finetune few-shot sampling when the dataset supports it.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue processing remaining datasets if one dataset fails.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "resampled_size"):
        args.resampled_size = RESAMPLED_SIZE_SENTINEL
    try:
        dataset_names = normalize_dataset_names(args.datasets)
    except ValueError as exc:
        parser.error(str(exc))

    failures = []
    for dataset_name in dataset_names:
        print(f"\n===== Processing {dataset_name} =====")
        try:
            run_dataset(dataset_name, args)
        except Exception as exc:
            if not args.continue_on_error:
                raise
            print(f"{dataset_name} failed: {exc}")
            failures.append((dataset_name, exc))

    if failures:
        failed_names = ", ".join(dataset_name for dataset_name, _ in failures)
        raise RuntimeError(f"Failed datasets: {failed_names}")


if __name__ == "__main__":
    main()
