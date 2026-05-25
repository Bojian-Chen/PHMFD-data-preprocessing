import argparse
import itertools
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


DEFAULT_DATASETS = (
    "HITSM_self_built",
    "HITSM_SpectraQuest",
    "KAIST1",
    "KAIST2",
    "KAIST3",
    "CWRU",
    "TORINO",
    "XJTUSY",
    "MFPT",
    "FEMTO",
)
DEFAULT_TRAIN_FILES = ("train.parquet",)
DEFAULT_DATA_SUBDIRS = ("Pretrain",)


def mixup_data(src_x, trg_x, mix_ratio, temporal_shift=50):
    """Mix two one-dimensional samples with local temporal shifts."""
    h = temporal_shift // 2
    shifts = range(-h, h)

    trg_shift_mean = torch.stack(
        [torch.roll(trg_x, shifts=-i, dims=0) for i in shifts],
        dim=0,
    ).mean(dim=0)
    src_shift_mean = torch.stack(
        [torch.roll(src_x, shifts=-i, dims=0) for i in shifts],
        dim=0,
    ).mean(dim=0)

    mixed_x1 = mix_ratio * src_x + (1 - mix_ratio) * trg_shift_mean
    mixed_x2 = mix_ratio * trg_x + (1 - mix_ratio) * src_shift_mean
    return mixed_x1.numpy(), mixed_x2.numpy()


def normalize_samples(samples):
    """Convert parquet sample lists to a 2D float32 tensor: [num_samples, length]."""
    x_np = np.asarray(samples, dtype=np.float32)
    if x_np.ndim == 2:
        return torch.from_numpy(x_np)
    if x_np.ndim == 3 and x_np.shape[1] == 1:
        return torch.from_numpy(x_np[:, 0, :])
    raise ValueError(
        f"Expected samples with shape [N, L] or [N, 1, L], got {x_np.shape}."
    )


def load_parquet(data_file):
    print(f"Loading {data_file}")
    table = pq.read_table(data_file, columns=["samples"])
    return normalize_samples(table["samples"].to_pylist())


def make_output_table(samples, dataset_name):
    x_np = np.asarray(samples, dtype=np.float32)
    return pa.Table.from_pydict(
        {
            "samples": x_np.tolist(),
            "labels": [0] * len(x_np),
            "dataset": [dataset_name] * len(x_np),
        }
    )


def write_mixed_pair(
    file1,
    file2,
    data_root,
    output_dir,
    alpha,
    batch_size,
    temporal_shift,
    max_samples=None,
    skip_existing=False,
):
    data_id1 = dataset_id_from_train_file(file1, data_root)
    data_id2 = dataset_id_from_train_file(file2, data_root)
    dataset_name = f"mixup_{data_id1}_{data_id2}"
    output_file = output_dir / f"{dataset_name}.parquet"
    if skip_existing and output_file.exists():
        print(f"Skipped existing mixup file: {output_file}")
        return

    x1 = load_parquet(file1)
    x2 = load_parquet(file2)
    if x1.shape[1] != x2.shape[1]:
        raise ValueError(
            f"Sample length mismatch: {file1} has {x1.shape[1]}, "
            f"{file2} has {x2.shape[1]}."
        )

    sample_count = min(len(x1), len(x2))
    if max_samples is not None:
        sample_count = min(sample_count, max_samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for start in range(0, sample_count, batch_size):
            end = min(start + batch_size, sample_count)
            mixed_samples = []
            for i in range(start, end):
                mixed_x1, mixed_x2 = mixup_data(
                    x1[i],
                    x2[i],
                    alpha,
                    temporal_shift=temporal_shift,
                )
                mixed_samples.append(mixed_x1)
                mixed_samples.append(mixed_x2)

            table = make_output_table(mixed_samples, dataset_name)
            if writer is None:
                writer = pq.ParquetWriter(output_file, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    mixed_shape = (sample_count * 2, *np.asarray(x1[0]).shape)
    print(f"Saved mixed samples shape={mixed_shape} to {output_file}")


def resolve_train_file(data_root, dataset_name):
    candidate_dirs = [data_root / subdir / dataset_name for subdir in DEFAULT_DATA_SUBDIRS]
    candidate_dirs.append(data_root / dataset_name)
    for dataset_dir in candidate_dirs:
        for file_name in DEFAULT_TRAIN_FILES:
            candidate = dataset_dir / file_name
            if candidate.exists():
                return candidate
    return None


def dataset_id_from_train_file(train_file, data_root):
    try:
        relative_parent = train_file.parent.relative_to(data_root)
    except ValueError:
        return train_file.parent.name
    parts = relative_parent.parts
    if parts and parts[0] in {"Pretrain", "Finetune"}:
        parts = parts[1:]
    if not parts:
        return train_file.parent.name
    return "_".join(parts)


def collect_train_files(data_root, dataset_names):
    train_files = []
    missing = []
    for dataset_name in dataset_names:
        train_file = resolve_train_file(data_root, dataset_name)
        if train_file is None:
            missing.append(dataset_name)
            continue
        train_files.append(train_file)

    if missing:
        print(
            "Skipped missing datasets or train files: "
            + ", ".join(sorted(missing))
        )
    return train_files


def perform_mixup_on_files(
    file_paths,
    data_root,
    output_dir,
    alpha,
    batch_size,
    temporal_shift,
    max_samples=None,
    skip_existing=False,
):
    if len(file_paths) < 2:
        raise ValueError("At least two available train files are required for mixup.")

    for file1, file2 in itertools.combinations(file_paths, 2):
        write_mixed_pair(
            file1,
            file2,
            data_root,
            output_dir,
            alpha,
            batch_size,
            temporal_shift,
            max_samples=max_samples,
            skip_existing=skip_existing,
        )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate mixup parquet files from processed PHMFD datasets."
    )
    parser.add_argument(
        "--data-root",
        default="Process_data",
        help="Root directory containing processed dataset folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset folders to mix. Missing datasets are skipped.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to <data-root>/mixed.",
    )
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--temporal-shift", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per dataset pair for debugging or quick runs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print resolved input and output paths.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a dataset pair when its output parquet already exists.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else data_root / "mixed"

    train_files = collect_train_files(data_root, args.datasets)
    print("Resolved train files:")
    for train_file in train_files:
        print(f"  {train_file}")
    print(f"Output directory: {output_dir}")

    if args.dry_run:
        return

    perform_mixup_on_files(
        train_files,
        data_root,
        output_dir,
        alpha=args.alpha,
        batch_size=args.batch_size,
        temporal_shift=args.temporal_shift,
        max_samples=args.max_samples,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
