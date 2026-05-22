# PHMFD 数据预处理

本仓库用于将 PHMFD 涉及的多个机械故障诊断数据集统一预处理为 Parquet 格式。统一入口为 `main.py`，每个数据集的读取、切分、归一化、重采样和保存逻辑放在 `data_scripts/` 下。

## 数据仓库

原始数据和已经处理好的 Parquet 数据托管在 Hugging Face：

- 数据仓库：[bojian1/PHMFD-data](https://huggingface.co/datasets/bojian1/PHMFD-data/)
- 处理后数据目录：`Process_Data/`
- 重新预处理时的默认原始数据目录：`Raw_data/UniFault/`

如只需要使用处理好的数据，可以直接下载 Hugging Face 仓库中的 `Process_Data/`。如需重新运行本仓库的预处理脚本，请将原始数据保持在 `Raw_data/UniFault/<dataset_name>/` 结构下。

```bash
pip install -U huggingface_hub

# 下载完整数据仓库到当前目录
huggingface-cli download bojian1/PHMFD-data \
  --repo-type dataset \
  --local-dir .

# 只下载处理后的 Parquet 数据
huggingface-cli download bojian1/PHMFD-data \
  --repo-type dataset \
  --include "Process_Data/**" \
  --local-dir .
```

## 目录结构

```text
.
├── main.py                 # 统一调度入口
├── data_scripts/           # 各数据集预处理脚本
├── Raw_data/UniFault/      # 默认原始数据根目录
├── Process_Data/           # 默认处理后数据输出目录
└── README.md
```

## 支持的数据集

下表中的原始目录均相对于 `--raw-root`，默认 `--raw-root Raw_data/UniFault`；输出目录均相对于 `--save-root`，默认 `--save-root Process_Data`。

| 名称 | 原始目录候选 | 输出目录 | 任务类型 |
| --- | --- | --- | --- |
| `CNC` | `CNC` | `CNC` | finetune |
| `CWRU` | `CWRU` | `CWRU` | pretrain |
| `FEMTO` | `FEMTO` | `FEMTO` | pretrain |
| `HITSM` | `HIT-SM`, `HITSM` | `HITSM` | pretrain |
| `IMS_FD` | `IMS`, `IMS_FD` | `IMS_FD` | finetune |
| `KAIST` | `KAIST` | `KAIST1`, `KAIST2`, ... | pretrain |
| `MFPT` | `MFPT` | `MFPT` | pretrain |
| `PU` | `PU`, `RM_027_PU` | `PU` | finetune |
| `TORINO` | `DIRG`, `TORINO` | `TORINO` | pretrain |
| `XJTUSY` | `XJTU-SY`, `XJTUSY` | `XJTUSY` | pretrain |

`main.py` 会在候选原始目录中优先使用实际存在的目录。`KAIST` 会根据原始数据中的完整 CSV 子集自动生成 `KAIST1`、`KAIST2` 等输出目录。

## 环境依赖

建议使用 Python 3.10+。

```bash
pip install numpy pandas pyarrow scipy torch h5py
```

主要依赖用途：

- `torch`：张量处理、随机切分和部分数据集的窗口构造。
- `pyarrow`：写入和读取 Parquet 文件。
- `h5py`：读取 CNC 数据集。
- `scipy`：读取 `.mat` 文件和重采样。

## 快速开始

查看命令行参数：

```bash
python main.py --help
```

处理全部数据集：

```bash
python main.py --datasets all
```

处理指定数据集：

```bash
python main.py --datasets CWRU PU IMS_FD
```

指定输入和输出根目录：

```bash
python main.py \
  --datasets PU \
  --raw-root Raw_data/UniFault \
  --save-root Process_Data
```

设置窗口长度、归一化方式和重采样长度：

```bash
python main.py \
  --datasets CWRU FEMTO \
  --sample-time 0.1 \
  --norm-method zscore \
  --resampled-size 1024
```

禁用重采样：

```bash
python main.py --datasets CWRU --resampled-size none
```

遇到单个数据集失败时继续处理后续数据集：

```bash
python main.py --datasets all --continue-on-error
```

## 全局参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--datasets` | `all` | 要处理的数据集，支持 `all` 或多个数据集名 |
| `--raw-root` | `Raw_data/UniFault` | 原始数据根目录 |
| `--save-root` | `Process_Data` | 处理后数据输出根目录 |
| `--sample-time` | `0.1` | 每个样本窗口长度，单位秒 |
| `--norm-method` | `minmax` | 归一化方式，可选 `none`、`minmax`、`zscore` |
| `--resampled-size` | `1024` | 重采样后的长度；传入 `none` 可禁用 |
| `--train-size` | `0.6` | 训练集比例 |
| `--val-size` | `0.2` | 验证集比例 |
| `--test-size` | `0.2` | 测试集比例 |
| `--seed` | `42` | 数据切分随机种子 |
| `--fewshot-seed` | `42` | finetune 数据集的 few-shot 采样种子 |
| `--continue-on-error` | `False` | 单个数据集失败后是否继续处理其他数据集 |

`sampling_frequency` 等数据集固有参数保留在各自的 `data_scripts/*.py` 默认值中，不在 `main.py` 中统一覆盖。

## 输出格式

预训练数据集通常输出：

```text
Process_Data/<dataset>/
├── train.parquet
├── val.parquet
└── test.parquet
```

微调数据集通常输出小样本训练划分：

```text
Process_Data/<dataset>/
├── train_1p.parquet
├── val.parquet
└── test.parquet
```

Parquet 文件通常包含以下字段：

| 字段 | 说明 |
| --- | --- |
| `samples` | 一条或多通道时序样本 |
| `labels` | 故障类别标签；部分预训练数据可能不包含该字段 |
| `dataset` | 数据集或划分标识 |

## 新增数据集

每个数据集脚本需要在文件顶部提供 `DATASET_CONFIG`，用于声明入口类或函数、任务类型、原始目录候选名和输出目录名。

```python
DATASET_CONFIG = {
    "target": "PreparePaderborn",
    "method": "prepare_dataset",
    "task": "finetune",
    "raw_folders": ("PU", "RM_027_PU"),
    "save_folder": "PU",
}
```

`main.py` 会根据入口类或函数签名自动映射常见参数名：

- 原始数据路径：`raw_dir`、`data_dir`、`data_root`、`raw_root`
- 输出路径：`save_dir`、`save_root`、`save_path`
- 窗口长度：`sample_time`、`time_interval`、`desired_duration_sec`
- 归一化方式：`norm_method`、`norm`
- 其他统一参数：`resampled_size`、`train_size`、`val_size`、`test_size`、`seed`、`fewshot_seed`

新增数据集步骤：

1. 在 `data_scripts/` 下新增数据集脚本。
2. 在脚本中添加 `DATASET_CONFIG`。
3. 在 `data_scripts/__init__.py` 的 `DATASET_MODULES` 中注册模块名。

`data_scripts/` 下提供了两个扩展示例：

- `example_pretrain.py`：pretrain 数据集模板，输出 `train.parquet`、`val.parquet`、`test.parquet`。
- `example_finetune.py`：finetune 数据集模板，输出 `train_1p.parquet`、`val.parquet`、`test.parquet`，并演示 `fewshot_seed` 的使用。

这两个示例默认不注册到 `DATASET_MODULES`，不会被 `python main.py --datasets all` 执行。
