# PHMFD 数据预处理

本仓库用于将 PHM 基础模型训练和微调涉及的多个机械故障诊断数据集统一预处理为 Parquet 格式。统一入口为 `main.py`，每个数据集的读取、切分、归一化、重采样和保存逻辑放在 `data_scripts/` 下。

## 数据仓库

原始数据和已经处理好的 Parquet 数据托管在 Hugging Face：

- 数据仓库：[bojian1/PHMFD-data](https://huggingface.co/datasets/bojian1/PHMFD-data/)
- 原始数据目录：`Raw_data/Pretrain/`、`Raw_data/Finetune/`
- 处理后数据目录：`Process_data/Pretrain/`、`Process_data/Finetune/`
- mixup 数据目录：`Process_data/mixed/`

如只需要使用处理好的数据，可以直接下载 Hugging Face 仓库中的 `Process_data/`。如需重新运行本仓库的预处理脚本，请将当前使用的数据集按任务保持在 `Raw_data/Pretrain/<dataset_name>/` 或 `Raw_data/Finetune/<dataset_name>/` 结构下。

```bash
pip install -U huggingface_hub

# 下载完整数据仓库到当前目录
huggingface-cli download bojian1/PHMFD-data \
  --repo-type dataset \
  --local-dir .

# 只下载处理后的 Parquet 数据
huggingface-cli download bojian1/PHMFD-data \
  --repo-type dataset \
  --include "Process_data/**" \
  --local-dir .
```

## 目录结构

```text
.
├── main.py                 # 统一调度入口
├── data_scripts/           # 各数据集预处理脚本
├── Raw_data/
│   ├── Pretrain/           # 预训练原始数据
│   └── Finetune/           # 微调原始数据
├── Process_data/
│   ├── Pretrain/           # 预训练处理后数据
│   ├── Finetune/           # 微调处理后数据
│   └── mixed/              # mixup 输出，保持在 Process_data 根下
└── README.md
```

## 支持的数据集

下表中的原始目录会按任务自动映射到 `--raw-root/Pretrain` 或 `--raw-root/Finetune`，默认 `--raw-root Raw_data`；输出目录会按任务自动映射到 `--save-root/Pretrain` 或 `--save-root/Finetune`，默认 `--save-root Process_data`。

| 名称               | 原始目录候选              | 输出目录                                 | 任务类型 |
| ------------------ | ------------------------- | ---------------------------------------- | -------- |
| `CNC`              | `CNC`                     | `M01`, `M02`, `M03`                      | finetune |
| `CWRU`             | `CWRU`                    | `CWRU`                                   | pretrain |
| `FEMTO`            | `FEMTO`                   | `FEMTO`                                  | pretrain |
| `HITSM`            | `HIT-SM`, `HITSM`         | `HITSM_self_built`, `HITSM_SpectraQuest` | pretrain |
| `HUST_Bearing`     | `HUST_Bearing`            | `HUST_Bearing`                           | finetune |
| `HUST_Gearbox`     | `HUST_Gearbox`            | `HUST_Gearbox`                           | finetune |
| `HUST_Motor`       | `HUST_Motor`, `HIT_Motor` | `HUST_Motor`                             | finetune |
| `IMS_FD`           | `IMS`, `IMS_FD`           | `IMS_FD`                                 | finetune |
| `JNU`              | `JNU`                     | `JNU`                                    | finetune |
| `KAIST`            | `KAIST`                   | `KAIST1`, `KAIST2`, ...                  | pretrain |
| `MCC5-THU-Gearbox` | `MCC5`                    | `MCC5-THU-Gearbox`                       | finetune |
| `MFPT`             | `MFPT`                    | `MFPT`                                   | pretrain |
| `PU`               | `PU`, `RM_027_PU`         | `PU`                                     | finetune |
| `SDUST_Bearing`    | `SUDST`, `SDUST`          | `SDUST_Bearing`                          | finetune |
| `SDUST_Gear`       | `SUDST`, `SDUST`          | `SDUST_Gear`                             | finetune |
| `SEU`              | `SEU`                     | `SEU_Bearing`, `SEU_Gear`                | finetune |
| `TORINO`           | `DIRG`, `TORINO`          | `TORINO`                                 | pretrain |
| `UO`               | `UO`                      | `UO`                                     | pretrain |
| `WT`               | `WT`                      | `WT`                                     | finetune |
| `XJTUSY`           | `XJTU-SY`, `XJTUSY`       | `XJTUSY`                                 | pretrain |

`main.py` 会优先在分类后的任务目录中查找候选原始目录，并兼容未分类的旧目录结构。`KAIST` 会根据原始数据中的完整 CSV 子集生成 `KAIST1`、`KAIST2`、`KAIST3` 输出目录。

## 数据集元信息

下表中的采样频率、通道数、使用通道、类别和工况信息以当前 `data_scripts/` 预处理代码为准；来源 link 和 bib 信息未确认时留空，后续可继续补充。

| 数据集             | 数据集类型    | 采样频率                 | 原始通道数               | 使用的通道                                                    | 类别/工况信息                                                                                                                                                                                                                                                                                               | 来源 link                                                                                                                                                                  | 引用文献 bib                                                                         |
| ------------------ | ------------- | ------------------------ | ------------------------ | ------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `CNC`              | 工业/CNC 加工 | 2 kHz                    | 3                        | `vibration_data` 全部 3 通道                                  | 类别：`good`, `bad`；工况：`OP00`-`OP14`，按机器 `M01`/`M02`/`M03` 分别输出                                                                                                                                                                                                                                 | [Github](https://github.com/boschresearch/CNC_Machining/tree/main)                                                                                                         | [Ref](https://www.sciencedirect.com/science/article/pii/S2212827122002384)           |
| `CWRU`             | 轴承          | 12/48 kHz (只用了12 kHz) | 2                        | `DE_time`, `FE_time`                                          |                                                                                                                                                                                                                                                                                                             | [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file)                                                                              | [MSSP](https://www.sciencedirect.com/science/article/abs/pii/S0888327015002034)      |
| `FEMTO`            | 轴承          | 25.6 kHz                 | 6                        | CSV 第 5、6 列加速度通道                                      |                                                                                                                                                                                                                                                                                                             | [Github](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset)                                                                                               | [Ref](https://hal.science/hal-00719503/)                                             |
| `HITSM`            | 轴承          | 51.2 kHz                 | 1                        | `.mat` 信号通道                                               |                                                                                                                                                                                                                                                                                                             | [Github](https://github.com/hitwzc/Bearing-datasets)                                                                                                                       |                                                                                      |
| `HUST_Bearing`     | 轴承          | 25.6 kHz                 | 4                        | 最后三个通道，即 x/y/z 振动通道                               | 类别：`H`, `0.5X_I`, `I`, `0.5X_O`, `O`, `0.5X_B`, `B`, `0.5X_C`, `C`；工况：固定转速 `20`, `25`, `30`, `35`, `40`, `60`, `65`, `70`, `75`, `80` Hz；忽略文件名包含 `_VS_` 的变转速数据                                                                                                                     | [Github](https://github.com/CHAOZHAO-1/HUSTbearing-dataset)                                                                                                                | [Zhao2024](https://doi.org/10.1016/j.ress.2024.109964)                               |
| `HUST_Gearbox`     | 齿轮箱        | 25.6 kHz                 | 4                        | 最后三个通道，即 x/y/z 振动通道                               | 类别：`H`, `B`, `M`；工况：固定转速 `20`, `25`, `30`, `35`, `40` Hz 与负载 `L0`-`L4`；忽略文件名包含 `_VS_` 的变转速数据                                                                                                                                                                                    | [Github](https://github.com/CHAOZHAO-1/HUSTgearbox-dataset)                                                                                                                | [Zhao2024](https://doi.org/10.1016/j.ress.2024.109964)                               |
| `HUST_Motor`       | 电机          | 25.6 kHz                 | 4                        | 全部 4 个数据通道，即 x/y/z 振动通道和 `Sound` 声学通道       | 类别：`H`, `BF`, `BOW`, `BROKEN`, `MISAL`, `UNBAL`；工况：转速 `5`, `10`, `20`, `30` Hz                                                                                                                                                                                                                     | [Github](https://github.com/CHAOZHAO-1/HUSTmotor-multi-modal-dataset)                                                                                                      | [EAAI](https://www.sciencedirect.com/science/article/abs/pii/S0952197625025904)      |
| `IMS_FD`           | 轴承          | 20.48 kHz                | 1st test: 8；2nd test: 4 | `bearing1_1`, `bearing1_3`, `bearing1_4`, `bearing2_1` 对应列 | 类别：normal, inner race, roller element, outer race；工况：无单独工况分组                                                                                                                                                                                                                                  | [NASA IMS Bearings](https://data.nasa.gov/dataset/ims-bearings)                                                                                                            |                                                                                      |
| `JNU`              | 轴承          | 50 kHz                   | 1                        | 垂直方向振动信号                                              | 类别：`n`, `ib`, `ob`, `tb`；工况：转速 `600`, `800`, `1000` r/min                                                                                                                                                                                                                                          | [Github](https://github.com/ClarkGableWang/JNU-Bearing-Dataset)                                                                                                            |                                                                                      |
| `KAIST`            | 轴承          | 25.6 kHz                 | 4                        | `bearingA_x`, `bearingA_y`, `bearingB_x`, `bearingB_y`        |                                                                                                                                                                                                                                                                                                             | [Part1](https://data.mendeley.com/datasets/vxkj334rzv/7) [Part2](https://data.mendeley.com/datasets/x3vhp8t6hg/7) [Part3](https://data.mendeley.com/datasets/j8d8pfkvj2/4) | [Data_in_brief](https://www.sciencedirect.com/science/article/pii/S2352340923001671) |
| `MCC5-THU-Gearbox` | 齿轮箱        | 12.8 kHz                 | 8                        | 最后三个通道，即 `gearbox_vibration_x/y/z`                    | 类别：`health`, `gear_pitting_H/M/L`, `gear_wear_H/M/L`, `miss_teeth`, `teeth_break_H/M/L`, `teeth_crack_H/M/L`, `teeth_break_and_bearing_inner_H/M/L`, `teeth_break_and_bearing_outer_H/M/L`；仅使用 `torque_circulation` 文件，截取 10-20s 和 40-50s 两段固定工况；工况：`1000/2000/3000rpm` 与 `10/20Nm` | [Github]{https://github.com/liuzy0708/MCC5-THU-Gearbox-Benchmark-Datasets}                                                                                                 | [Data_in_brief]{https://www.sciencedirect.com/science/article/pii/S2352340924004220} |
| `MFPT`             | 轴承          | 97656/48828              | 1                        | `bearing` 结构中的振动信号                                    |                                                                                                                                                                                                                                                                                                             | [MFPT Fault Data Sets](https://www.mfpt.org/fault-data-sets/)                                                                                                              |                                                                                      |
| `PU`               | 轴承          | 64 kHz                   | 1                        | `Y` 结构中的振动信号                                          | 类别：`K001`, `KA04`, `KA15`, `KA16`, `KA22`, `KA30`, `KB23`, `KB24`, `KB27`, `KI14`, `KI16`, `KI17`, `KI18`, `KI21`；工况：`N15_M07_F10`, `N09_M07_F10`, `N15_M01_F10`, `N15_M07_F04`                                                                                                                      | [Paderborn Bearing DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter)                                                                             |                                                                                      |
| `SDUST_Bearing`    | 轴承          | 25.6 kHz                 | 6                        | `Signal.y_values.values` 全部 6 通道                          | 类别：`NC`, `IF0.2`, `IF0.4`, `IF0.6`, `OF0.2`, `OF0.4`, `OF0.6`, `RF0.2`, `RF0.4`, `RF0.6`；工况：固定转速 `1000`, `1500`, `1800`, `2000`, `2500`, `3000` rpm，且仅保留负载 `60` N；忽略 `1797`、`800~1500`、`1000-2000` 等数据                                                                            | [Github](https://github.com/JRWang-SDUST/SDUST-Dataset/tree/main)                                                                                                          | [KBS](https://www.sciencedirect.com/science/article/abs/pii/S095070512301033X)       |
| `SDUST_Gear`       | 齿轮箱        | 25.6 kHz                 | 6                        | `Signal.y_values.values` 全部 6 通道                          | 类别：`NC`, `太阳断裂`, `太阳点蚀`, `太阳磨损`, `行星断裂`, `行星点蚀`, `行星磨损`；工况：固定转速 `1000`, `1500`, `1800`, `2000`, `2500` rpm，且仅保留负载 `0.5A` 的第 `1` 次采集；忽略 `0-0.2A`, `0-0.35A`, `0-0.5A` 和 `flu` 波动工况                                                                    | [Github](https://github.com/JRWang-SDUST/SDUST-Dataset/tree/main)                                                                                                          | [KBS](https://www.sciencedirect.com/science/article/abs/pii/S095070512301033X)       |
| `SEU_Bearing`      | 轴承          | 5.12 kHz                 | 8                        | 第 2、3、4 通道                                               | 类别：`health`, `ball`, `inner`, `outer`, `comb`；工况：`20_0`, `30_2`                                                                                                                                                                                                                                      | [Github](https://github.com/cathysiyu/Mechanical-datasets/tree/master/gearbox)                                                                                             |                                                                                      |
| `SEU_Gear`         | 齿轮          | 5.12 kHz                 | 8                        | 第 2、3、4 通道                                               | 类别：`health`, `chipped`, `miss`, `root`, `surface`；工况：`20_0`, `30_2`                                                                                                                                                                                                                                  | [Github](https://github.com/cathysiyu/Mechanical-datasets/tree/master/gearbox)                                                                                             |                                                                                      |
| `TORINO`           | 轴承          | 51.2 kHz                 | 6                        | `.mat` 中全部 6 个通道                                        |                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                            |                                                                                      |
| `UO`               | 轴承          | 200 kHz                  | 2                        | `Channel_1`                                                   |                                                                                                                                                                                                                                                                                                             |                                                                                                                                                                            |                                                                                      |
| `WT`               | 齿轮/传动系统 | 48 kHz                   | 4                        | `Data` 前 2 通道，即 x/y 振动通道；仅取 70-90s                | 类别：`healthy`, `broken`, `missing_tooth`, `root_crack`, `wear`；工况：`.MAT` 文件名下划线后的条件值，如 `20`, `25`, `30`, `35`, `40`, `45`, `50`, `55`；当前仅使用每类 `1/` 文件夹                                                                                                                        | [Github](https://github.com/Liudd-BJUT/WT-planetary-gearbox-dataset)                                                                                                       | [MST](https://iopscience.iop.org/article/10.1088/1361-6501/acf390)                   |
| `XJTUSY`           | 轴承          | 25.6 kHz                 | 2                        | `Horizontal_vibration_signals`, `Vertical_vibration_signals`  |                                                                                                                                                                                                                                                                                                             | [XJTU-SY Bearing Datasets](http://biaowang.tech/xjtu-sy-bearing-datasets/)                                                                                                 |                                                                                      |

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
  --raw-root Raw_data \
  --save-root Process_data
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

只制作 finetune 的 N-shot 小样本训练集：

```bash
python main.py --datasets PU IMS_FD SEU --fewshot-shots 10
```

该模式只生成 `train_10shot.parquet`，不会生成 `train.parquet`、`train_1p.parquet`、`val.parquet`、`test.parquet`。N-shot 采样按类别抽取每类最多 N 条，并在该类别内部的工况组之间尽量均衡。

## 全局参数

| 参数                  | 默认值         | 说明                                                                        |
| --------------------- | -------------- | --------------------------------------------------------------------------- |
| `--datasets`          | `all`          | 要处理的数据集，支持 `all` 或多个数据集名                                   |
| `--raw-root`          | `Raw_data`     | 原始数据根目录                                                              |
| `--save-root`         | `Process_data` | 处理后数据输出根目录                                                        |
| `--sample-time`       | `0.1`          | 每个样本窗口长度，单位秒                                                    |
| `--norm-method`       | `minmax`       | 归一化方式，可选 `none`、`minmax`、`zscore`                                 |
| `--resampled-size`    | `1024`         | 重采样后的长度；传入 `none` 可禁用                                          |
| `--train-size`        | `0.6`          | 训练集比例                                                                  |
| `--val-size`          | `0.2`          | 验证集比例                                                                  |
| `--test-size`         | `0.2`          | 测试集比例                                                                  |
| `--seed`              | `42`           | 数据切分随机种子                                                            |
| `--fewshot-seed`      | `42`           | finetune 数据集的 few-shot 采样种子                                         |
| `--fewshot-shots`     | `None`         | 只制作 finetune 的 `train_Nshot.parquet`，每类最多 N 条，并尽量保持工况均衡 |
| `--continue-on-error` | `False`        | 单个数据集失败后是否继续处理其他数据集                                      |

`sampling_frequency` 等数据集固有参数保留在各自的 `data_scripts/*.py` 默认值中，不在 `main.py` 中统一覆盖。

## 输出格式

预训练数据集通常输出：

```text
Process_data/Pretrain/<dataset>/
├── train.parquet
├── val.parquet
└── test.parquet
```

微调数据集通常输出小样本训练划分：

```text
Process_data/Finetune/<dataset>/
├── train.parquet
├── train_1p.parquet
├── val.parquet
└── test.parquet
```

如果指定 `--fewshot-shots N`，finetune 数据集只输出：

```text
Process_data/Finetune/<dataset>/
└── train_Nshot.parquet
```

Parquet 文件通常包含以下字段：

| 字段      | 说明                                         |
| --------- | -------------------------------------------- |
| `samples` | 一条或多通道时序样本                         |
| `labels`  | 故障类别标签；部分预训练数据可能不包含该字段 |
| `dataset` | 数据集或划分标识                             |

finetune 数据集不强制转换为单通道：单通道数据通常保存为 `[1, L]`，多通道数据保留为 `[C, L]`。例如 `CNC` 保留 3 通道，`SEU_Bearing` 和 `SEU_Gear` 保留第 2、3、4 通道组成的 3 通道样本。

`train_1p.parquet` 按完整 `train.parquet` 数量的约 1% 生成，并在可行范围内尽量保持类别/工况组均衡；它不是“每个工况至少一条”的采样。

预处理完成后的终端日志会输出各 split 的实际 data shape，例如 `train=(N, C, L)`，不只输出样本数量。

## Mixup 数据增强

`mixup.py` 可基于 `Process_data/Pretrain` 中已有的预训练数据集生成两两组合的 mixup Parquet 文件，默认输出仍保留在 `Process_data/mixed/`。
运行 `python mixup.py` 即可使用默认数据集组合，也可通过 `--datasets CWRU TORINO --max-samples 100` 指定数据集和样本数量；加上 `--skip-existing` 可跳过输出目录中已经存在的组合文件。

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
- 其他统一参数：`resampled_size`、`train_size`、`val_size`、`test_size`、`seed`、`fewshot_seed`、`fewshot_shots`

新增数据集步骤：

1. 在 `data_scripts/` 下新增数据集脚本。
2. 在脚本中添加 `DATASET_CONFIG`。
3. 在 `data_scripts/__init__.py` 的 `DATASET_MODULES` 中注册模块名。

`data_scripts/` 下提供了两个扩展示例：

- `example_pretrain.py`：pretrain 数据集模板，输出 `train.parquet`、`val.parquet`、`test.parquet`。
- `example_finetune.py`：finetune 数据集模板，输出 `train.parquet`、`train_1p.parquet`、`val.parquet`、`test.parquet`，并演示 `fewshot_seed` 和 `fewshot_shots` 的使用。

这两个示例默认不注册到 `DATASET_MODULES`，不会被 `python main.py --datasets all` 执行。
