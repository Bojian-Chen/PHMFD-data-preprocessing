# AGENT.md

本文件给后续维护本仓库的 coding agent 使用。当前分支 `paper/unifault` 专门用于 UniFault 论文实验，目标是保持数据预处理代码风格一致，避免绕过统一入口或重复引入旧脚本习惯。

## 项目定位

本分支将 UniFault 论文实验涉及的多个机械故障诊断原始数据集统一预处理为 Parquet 文件，供预训练、微调和 mixup 数据增强使用。

核心入口是 `main.py`，数据集实现放在 `data_scripts/`。不要为单个数据集新增独立调度入口，除非只是保留脚本自身的 `if __name__ == "__main__"` 调试用法。

## 关键路径

- 默认原始数据根目录以 `main.py` 为准：`DEFAULT_RAW_ROOT`，当前为 `Raw_data`
- 默认输出目录：`Process_data`
- 当前使用的数据集按任务分类：`Raw_data/Pretrain`、`Raw_data/Finetune`、`Process_data/Pretrain`、`Process_data/Finetune`
- `mixup.py` 输出保持在 `Process_data/mixed`
- 原始数据和处理后数据仓库：`https://huggingface.co/datasets/bojian1/PHMFD-data/`
- UniFault 已制作数据托管目录：`https://huggingface.co/datasets/bojian1/PHMFD-data/tree/main/Process_data_UniFault`
- 本地大数据目录通常不应提交：`Raw_data/`、`Process_data/`

如果 README 和代码默认值不一致，优先以 `main.py` 为准，并同步修 README。

## 数据集接入约定

每个数据集脚本必须提供 `DATASET_CONFIG`：

```python
DATASET_CONFIG = {
    "target": "ProcessorClassOrFunction",
    "method": "process",
    "task": "pretrain",
    "raw_folders": ("DatasetRawFolder",),
    "save_folder": "DatasetOutputFolder",
}
```

`save_folder` 的含义：

- `main.py` 会根据 `task` 自动在 `Raw_data` 和 `Process_data` 下增加 `Pretrain` 或 `Finetune` 分类目录。
- 普通单输出数据集使用目录名，例如 `"CWRU"` 会输出到 `Process_data/Pretrain/CWRU`，`"PU"` 会输出到 `Process_data/Finetune/PU`。
- 一个脚本拆成多个顶层数据集时使用空字符串 `""`，由脚本自己写入 `Process_data/<Task>/<dataset>`。
- 当前多输出脚本包括 `CNC`、`HITSM`、`KAIST`。

`task` 使用：

- `pretrain`：通常输出 `train.parquet`、`val.parquet`、`test.parquet`
- `finetune`：通常输出完整训练集 `train.parquet`、1% 训练集 `train_1p.parquet`、`val.parquet`、`test.parquet`

新增数据集时还要在 `data_scripts/__init__.py` 的 `DATASET_MODULES` 中注册模块名。必要时在 `DATASET_ALIASES` 中加常用别名。

不要再新增重复常量，例如 `DATASET_NAME`、`DEFAULT_RAW_DIR`、`DEFAULT_SAVE_DIR`。默认数据集名、原始目录和输出目录应从 `DATASET_CONFIG` 或 `main.py` 传参派生。

当前特殊输出约定：

- `CNC` 输出为 `Process_data/Finetune/M01`、`Process_data/Finetune/M02`、`Process_data/Finetune/M03`，不要加 `CNC_` 前缀，也不要再嵌套到 `Process_data/Finetune/CNC/`。
- `HITSM` 输出为 `Process_data/Pretrain/HITSM_self_built` 和 `Process_data/Pretrain/HITSM_SpectraQuest`，不要再嵌套到 `Process_data/Pretrain/HITSM/`。
- `KAIST` 输出为 `Process_data/Pretrain/KAIST1`、`Process_data/Pretrain/KAIST2`、`Process_data/Pretrain/KAIST3`；当前显式映射为 `part1 -> (0,1,2)`、`part2 -> (3,4)`、`part3 -> (5,6)`。

## main.py 参数映射

`main.py` 会按目标类或函数签名自动注入统一参数。优先使用以下参数名：

- 原始数据路径：`raw_dir`、`data_dir`、`data_root`、`raw_root`
- 输出路径：`save_dir`、`save_root`、`save_path`
- 窗口长度：`sample_time`、`time_interval`、`desired_duration_sec`
- 归一化：`norm_method`、`norm`
- 其他：`resampled_size`、`train_size`、`val_size`、`test_size`、`seed`、`fewshot_seed`

数据集固有参数如 `sampling_frequency` 保留在各脚本默认值中，不放到全局 CLI，除非确实所有数据集都需要统一控制。

## 输出格式

Parquet 字段约定：

- `samples`：时序样本，常见形状为 `[1024]` 或 `[1, 1024]`
- `labels`：故障类别标签；部分 pretrain 数据可能没有
- `dataset`：数据集名或 split 名

预训练脚本一般保存一维样本，即写出前可对 `[N, 1, L]` 使用 `squeeze(1)`。微调脚本可保留标签列。

## 数据集元信息维护

`README.md` 中的“数据集元信息”表需要随代码和资料持续更新。新增数据集或确认已有数据集信息时，自动补充以下字段：

- 数据集类型，例如轴承、齿轮、工业/CNC 加工等。
- 采样频率，以脚本默认值和原始数据说明为准。
- 原始通道数和实际使用通道，以 `data_scripts/*.py` 当前读取逻辑为准。
- 来源 link，优先填官方数据页、作者仓库或可信数据仓库。
- 引用文献 bib；未确认完整 BibTeX 时保持空白，不要编造。

如果来源 link 或 bib 不确定，留空并继续推进当前任务。

## Mixup

`mixup.py` 基于 `Process_data/Pretrain` 中已有的 `train.parquet` 生成数据集两两组合文件，默认输出到 `Process_data/mixed/`。
默认 mixup 数据集使用预训练处理后目录名，例如 `HITSM_self_built`、`HITSM_SpectraQuest`、`KAIST1`、`KAIST2`、`KAIST3`、`CWRU`、`TORINO`、`XJTUSY`。输出文件名不带 `Pretrain_` 前缀，不要使用旧的嵌套路径如 `HITSM/HITSM_self_built`。

常用命令：

```bash
python mixup.py --dry-run
python mixup.py --skip-existing
python mixup.py --datasets CWRU TORINO --max-samples 100
```

`--skip-existing` 会在输出文件已存在时跳过该组合，避免重复生成和覆盖。

## 验证命令

改动后至少运行：

```bash
python -m py_compile main.py data_scripts/*.py mixup.py
python main.py --help
python mixup.py --dry-run
```

验证单个数据集时优先输出到 `/tmp`，避免污染正式 `Process_data`：

```bash
python main.py \
  --datasets UO \
  --raw-root Raw_data \
  --save-root /tmp/phmfd_test_output \
  --sample-time 0.1 \
  --norm-method minmax \
  --resampled-size 1024
```

必要时用 PyArrow 快速检查输出：

```bash
python - <<'PY'
import pyarrow.parquet as pq
t = pq.read_table('/tmp/phmfd_test_output/Pretrain/UO/train.parquet')
print(t.num_rows, t.column_names, len(t['samples'][0].as_py()))
PY
```

## 维护注意事项

- 使用 `rg` 搜索代码，不要依赖肉眼查找。
- 不要提交或重写大数据文件。
- 不要把旧脚本里的绝对路径带入仓库。
- 不要引入绘图后端依赖到预处理脚本中。
- 修改默认路径时同步 `README.md`、`main.py` 和相关脚本说明。
- 修改统一入口、数据集注册、默认路径、输出目录结构、Parquet 字段约定或 `mixup.py` 默认行为时，必须同步更新本文件。
- 修改或新增数据集分类、Raw/Process 目录位置、mixup 输入输出目录时，必须同步更新本文件和 README。
- 对已有用户改动保持谨慎，只改当前任务相关文件。
