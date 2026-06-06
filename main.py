import argparse
import ast
import json
import inspect
from collections import Counter
from importlib import import_module
from pathlib import Path

import pyarrow.parquet as pq

from data_scripts import DATASET_ALIASES, DATASET_MODULES


DEFAULT_RAW_ROOT = Path("Raw_data")
DEFAULT_SAVE_ROOT = Path("Process_data_02s")
DEFAULT_SAMPLE_TIME = 0.2
DEFAULT_NORM_METHOD = "none"
DEFAULT_RESAMPLED_SIZE = None
DEFAULT_SEED = 42
DEFAULT_FEWSHOT_SEED = 42
RESAMPLED_SIZE_SENTINEL = "__default__"
SKIP_DATASETS_FOR_ALL = {"FEMTO"}

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
    "fewshot_shots": ("fewshot_shots",),
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


def task_folder(config):
    task = config.get("task")
    if task == "pretrain":
        return "Pretrain"
    if task == "finetune":
        return "Finetune"
    raise ValueError(f"Unknown dataset task: {task}")


def first_existing_path(roots, folder_names):
    if isinstance(roots, (str, Path)):
        roots = (roots,)
    roots = tuple(Path(root) for root in roots)
    candidates = [
        root / folder_name
        for root in roots
        for folder_name in folder_names
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def dataset_raw_roots(args, config):
    raw_root = Path(args.raw_root)
    return (raw_root / task_folder(config), raw_root)


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
        "raw_path": first_existing_path(
            dataset_raw_roots(args, config),
            config["raw_folders"],
        ),
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
        values["fewshot_shots"] = args.fewshot_shots
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


def split_name_from_path(path):
    return path.stem


def parse_group_condition(group_value):
    try:
        parsed = ast.literal_eval(str(group_value))
    except (SyntaxError, ValueError):
        return str(group_value)
    if isinstance(parsed, tuple) and len(parsed) >= 2:
        return repr(parsed[-1])
    return repr(parsed)


def counter_json(values):
    counter = Counter(str(value) for value in values)
    return json.dumps(dict(sorted(counter.items())), ensure_ascii=False)


def first_sample_shape(path):
    try:
        pf = pq.ParquetFile(path)
        batch_iter = pf.iter_batches(batch_size=1, columns=["samples"])
        batch = next(batch_iter, None)
    except Exception:
        return None
    if batch is None or batch.num_rows == 0:
        return None
    sample = batch.column(0)[0].as_py()
    shape = []
    while isinstance(sample, list):
        shape.append(len(sample))
        sample = sample[0] if sample else None
    return tuple(shape)


def summarize_parquet_file(path, save_root):
    pf = pq.ParquetFile(path)
    columns = set(pf.schema_arrow.names)
    split_name = split_name_from_path(path)
    relative_path = path.relative_to(save_root)

    dataset_values = []
    if "dataset" in columns:
        dataset_values = pq.read_table(path, columns=["dataset"])["dataset"].to_pylist()
    dataset_names = sorted({str(value) for value in dataset_values})
    dataset_name = dataset_names[0] if len(dataset_names) == 1 else str(relative_path.parent)

    labels = []
    if "labels" in columns:
        labels = pq.read_table(path, columns=["labels"])["labels"].to_pylist()

    groups = []
    conditions = []
    if "group" in columns:
        groups = pq.read_table(path, columns=["group"])["group"].to_pylist()
        conditions = [parse_group_condition(group) for group in groups]

    is_train_1p = split_name == "train_1p"
    return {
        "dataset": dataset_name,
        "output_dir": str(relative_path.parent),
        "split": split_name,
        "path": str(path),
        "rows": pf.metadata.num_rows,
        "sample_shape": first_sample_shape(path),
        "has_labels": "labels" in columns,
        "label_count": len(set(labels)) if labels else 0,
        "label_distribution": counter_json(labels) if labels else "{}",
        "has_group": "group" in columns,
        "group_count": len(set(groups)) if groups else 0,
        "condition_count": len(set(conditions)) if conditions else 0,
        "condition_distribution": counter_json(conditions) if conditions else "{}",
        "train_1p_condition_count": len(set(conditions)) if is_train_1p and conditions else "",
        "train_1p_group_count": len(set(groups)) if is_train_1p and groups else "",
    }


def write_process_data_summary(save_root):
    save_root = Path(save_root)
    rows = [
        summarize_parquet_file(path, save_root)
        for path in sorted(save_root.rglob("*.parquet"))
    ]
    if not rows:
        print(f"No parquet files found under {save_root}; skipped summary CSV.")
        return None

    import pandas as pd

    summary_path = save_root / "process_data_build_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved process data summary CSV: {summary_path}")
    return summary_path


def normalize_dataset_names(dataset_names):
    if any(name.lower() == "all" for name in dataset_names):
        return [
            dataset_name
            for dataset_name in DATASET_MODULES
            if dataset_name not in SKIP_DATASETS_FOR_ALL
        ]

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
        default="zscore",
        help="Normalization method: none, minmax, zscore.",
    )
    parser.add_argument(
        "--resampled-size",
        type=parse_optional_int,
        default="none",
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
        "--fewshot-shots",
        type=int,
        default=None,
        help=(
            "If set for finetune datasets, only generate train_<N>shot.parquet "
            "with up to N samples per class, balanced across condition groups."
        ),
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

    write_process_data_summary(args.save_root)

    if failures:
        failed_names = ", ".join(dataset_name for dataset_name, _ in failures)
        raise RuntimeError(f"Failed datasets: {failed_names}")


if __name__ == "__main__":
    main()
