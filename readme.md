# PHMFD 数据预处理

本项目用于将多个故障诊断原始数据集统一预处理为 Parquet 文件。统一入口是 `main.py`，各数据集的具体读取、切分、归一化和保存逻辑放在 `data_scripts/` 下。

## 目录结构

```text
.
├── main.py                 # 统一调度入口
├── data_scripts/           # 各数据集预处理脚本
├── Raw_data/               # 原始数据根目录
├── Process_Data/           # 处理后数据输出目录
└── readme.md
```

默认情况下，原始数据从 `Raw_data` 读取，处理结果保存到 `Process_Data`。

## 支持的数据集

| 名称 | 默认原始目录 | 默认输出目录 | 类型 |
| --- | --- | --- | --- |
| `CNC` | `Raw_data/CNC` | `Process_Data/CNC` | finetune |
| `CWRU` | `Raw_data/CWRU` | `Process_Data/CWRU` | pretrain |
| `FEMTO` | `Raw_data/FEMTO` | `Process_Data/FEMTO` | pretrain |
| `HITSM` | `Raw_data/HIT-SM` 或 `Raw_data/HITSM` | `Process_Data/HITSM` | pretrain |
| `IMS_FD` | `Raw_data/IMS` 或 `Raw_data/IMS_FD` | `Process_Data/IMS_FD` | finetune |
| `KAIST` | `Raw_data/KAIST` | `Process_Data/KAIST*` | pretrain |
| `MFPT` | `Raw_data/MFPT` | `Process_Data/MFPT` | pretrain |
| `PU` | `Raw_data/PU` 或 `Raw_data/RM_027_PU` | `Process_Data/PU` | finetune |
| `TORINO` | `Raw_data/DIRG` 或 `Raw_data/TORINO` | `Process_Data/TORINO` | pretrain |
| `XJTUSY` | `Raw_data/XJTU-SY` 或 `Raw_data/XJTUSY` | `Process_Data/XJTUSY` | pretrain |

`KAIST` 会根据原始数据中的完整 CSV 子集自动生成 `KAIST1`、`KAIST2` 等输出目录。

## 环境依赖

建议使用 Python 3.10+。主要依赖包括：

```powershell
pip install numpy pandas pyarrow scipy torch h5py
```

其中：

- `torch` 用于 CWRU、FEMTO、HITSM、KAIST、MFPT、TORINO、XJTUSY 等脚本中的张量处理。
- `h5py` 用于 CNC 数据集。
- `scipy` 用于 `.mat` 文件读取和重采样。

## 数据放置

将原始数据放到 `Raw_data` 下，保持各数据集自身的目录结构。例如：

```text
Raw_data/
├── CWRU/
├── PU/
├── IMS/
├── CNC/
├── KAIST/
└── XJTU-SY/
```

部分数据集支持多个原始目录名，`main.py` 会优先使用实际存在的候选目录。例如 `TORINO` 可使用 `Raw_data/DIRG` 或 `Raw_data/TORINO`。

## 使用方法

查看帮助：

```powershell
python main.py --help
```

处理全部数据集：

```powershell
python main.py --datasets all
```

处理指定数据集：

```powershell
python main.py --datasets CWRU PU IMS_FD
```

指定输入和输出根目录：

```powershell
python main.py --datasets PU --raw-root Raw_data --save-root Process_Data
```

设置统一窗口长度、归一化方式和重采样长度：

```powershell
python main.py --datasets CWRU FEMTO --sample-time 0.1 --norm-method zscore --resampled-size 1024
```

禁用重采样：

```powershell
python main.py --datasets CWRU --resampled-size none
```

遇到单个数据集失败时继续处理后续数据集：

```powershell
python main.py --datasets all --continue-on-error
```

## 全局参数

`main.py` 统一控制以下默认值：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--raw-root` | `Raw_data` | 原始数据根目录 |
| `--save-root` | `Process_Data` | 输出数据根目录 |
| `--sample-time` | `0.1` | 每个样本窗口长度，单位秒 |
| `--norm-method` | `none` | 归一化方式，可用 `none`、`minmax`、`zscore` |
| `--resampled-size` | `none` | 重采样后的长度 |
| `--train-size` | `0.6` | 训练集比例 |
| `--val-size` | `0.2` | 验证集比例 |
| `--test-size` | `0.2` | 测试集比例 |
| `--seed` | `42` | 数据切分随机种子 |
| `--fewshot-seed` | `43` | finetune 数据集的 few-shot 采样种子 |

`sampling_frequency` 等数据集固有参数保留在各自的 `data_scripts/*.py` 默认值中，不在 `main.py` 中统一覆盖。

## 数据集注册方式

每个数据集脚本在文件顶部提供 `DATASET_CONFIG`，描述该数据集的入口类或函数、任务类型、原始目录候选名和输出目录名。例如：

```python
DATASET_CONFIG = {
    "target": "PreparePaderborn",
    "method": "prepare_dataset",
    "task": "finetune",
    "raw_folders": ("PU", "RM_027_PU"),
    "save_folder": "PU",
}
```

`main.py` 会根据目标类或函数的签名自动映射同义参数，例如：

- `raw_dir`、`data_dir`、`data_root`、`raw_root`
- `save_dir`、`save_root`、`save_path`
- `sample_time`、`time_interval`、`desired_duration_sec`
- `norm_method`、`norm`

新增数据集时，需要：

1. 在 `data_scripts/` 下新增数据集脚本。
2. 在脚本中添加 `DATASET_CONFIG`。
3. 在 `data_scripts/__init__.py` 的 `DATASET_MODULES` 中注册模块名。

`data_scripts/` 下提供了两个扩展示例：

- `example_pretrain.py`：pretrain 数据集模板，输出 `train.parquet`、`val.parquet`、`test.parquet`。
- `example_finetune.py`：finetune 数据集模板，输出 `train_1p.parquet`、`val.parquet`、`test.parquet`，并演示 `fewshot_seed` 的使用。

这两个示例默认不注册到 `DATASET_MODULES`，因此不会被 `python main.py --datasets all` 执行。扩展新数据集时，可以复制其中一个模板，改成真实的数据读取逻辑，然后再注册模块名。

## Finetune 数据集

当前显式标记为 finetune 的数据集是：

- `CNC`
- `IMS_FD`
- `PU`

这类数据集会生成 `train_1p`、`val`、`test` 等用于小样本微调的划分。`fewshot_seed` 只会传给支持该参数的数据集脚本。
