# UniFault 数据预处理

本分支专门用于 UniFault 论文实验的数据预处理，将论文涉及的多个机械故障诊断数据集统一转换为 Parquet 格式。统一入口为 `main.py`，每个数据集的读取、切分、归一化、重采样和保存逻辑放在 `data_scripts/` 下。

## 数据仓库

原始数据和已经处理好的 Parquet 数据托管在 Hugging Face：

- 数据仓库：[bojian1/PHMFD-data](https://huggingface.co/datasets/bojian1/PHMFD-data/)
- UniFault 处理后数据目录：[Process_data_UniFault](https://huggingface.co/datasets/bojian1/PHMFD-data/tree/main/Process_data_UniFault)
- 原始数据目录：`Raw_data/Pretrain/`、`Raw_data/Finetune/`
- 本地脚本默认处理后数据目录：`Process_data/Pretrain/`、`Process_data/Finetune/`
- mixup 数据目录：`Process_data/mixed/`

如只需要使用 UniFault 已经制作好的数据，可以直接下载 Hugging Face 仓库中的 `Process_data_UniFault/`。如需重新运行本仓库的预处理脚本，请将当前使用的数据集按任务保持在 `Raw_data/Pretrain/<dataset_name>/` 或 `Raw_data/Finetune/<dataset_name>/` 结构下；脚本默认会重新生成到本地 `Process_data/`。

```bash
pip install -U huggingface_hub

# 下载完整数据仓库到当前目录
huggingface-cli download bojian1/PHMFD-data \
  --repo-type dataset \
  --local-dir .

# 只下载 UniFault 处理后的 Parquet 数据
huggingface-cli download bojian1/PHMFD-data \
  --repo-type dataset \
  --include "Process_data_UniFault/**" \
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

| 名称 | 原始目录候选 | 输出目录 | 任务类型 |
| --- | --- | --- | --- |
| `CNC` | `CNC` | `M01`, `M02`, `M03` | finetune |
| `CWRU` | `CWRU` | `CWRU` | pretrain |
| `FEMTO` | `FEMTO` | `FEMTO` | pretrain |
| `HITSM` | `HIT-SM`, `HITSM` | `HITSM_self_built`, `HITSM_SpectraQuest` | pretrain |
| `IMS_FD` | `IMS`, `IMS_FD` | `IMS_FD` | finetune |
| `KAIST` | `KAIST` | `KAIST1`, `KAIST2`, ... | pretrain |
| `MFPT` | `MFPT` | `MFPT` | pretrain |
| `PU` | `PU`, `RM_027_PU` | `PU` | finetune |
| `TORINO` | `DIRG`, `TORINO` | `TORINO` | pretrain |
| `UO` | `UO` | `UO` | pretrain |
| `XJTUSY` | `XJTU-SY`, `XJTUSY` | `XJTUSY` | pretrain |

`main.py` 会优先在分类后的任务目录中查找候选原始目录，并兼容未分类的旧目录结构。`KAIST` 会根据原始数据中的完整 CSV 子集生成 `KAIST1`、`KAIST2`、`KAIST3` 输出目录。

## 数据集元信息

下表中的采样频率、通道数和使用通道以当前 `data_scripts/` 预处理代码为准；来源 link 和 bib 信息未确认时留空，后续可继续补充。

| 数据集 | 数据集类型 | 采样频率 | 原始通道数 | 使用的通道 | 来源 link | 引用文献 bib |
| --- | --- | --- | --- | --- | --- | --- |
| `CNC` | 工业/CNC 加工 | 2 kHz | 3 | `vibration_data` 全部 3 通道 | [Github](https://github.com/boschresearch/CNC_Machining/tree/main) | [Ref](https://www.sciencedirect.com/science/article/pii/S2212827122002384) |
| `CWRU` | 轴承 | 12/48 kHz (只用了12 kHz) | 2 | `DE_time`, `FE_time` | [CWRU Bearing Data Center](https://engineering.case.edu/bearingdatacenter/download-data-file) | [MSSP](https://www.sciencedirect.com/science/article/abs/pii/S0888327015002034) |
| `FEMTO` | 轴承 | 25.6 kHz | 6 | CSV 第 5、6 列加速度通道 | [Github](https://github.com/Lucky-Loek/ieee-phm-2012-data-challenge-dataset) | [Ref](https://hal.science/hal-00719503/) |
| `HITSM` | 轴承 | 51.2 kHz | 1 | `.mat` 信号通道 |  |  |
| `IMS_FD` | 轴承 | 20.48 kHz | 1st test: 8；2nd test: 4 | `bearing1_1`, `bearing1_3`, `bearing1_4`, `bearing2_1` 对应列 | [NASA IMS Bearings](https://data.nasa.gov/dataset/ims-bearings) | `Qiu2006` |
| `KAIST` | 轴承 | 25.6 kHz | 4 | `bearingA_x`, `bearingA_y`, `bearingB_x`, `bearingB_y` |[Part1](https://data.mendeley.com/datasets/vxkj334rzv/7) [Part2](https://data.mendeley.com/datasets/x3vhp8t6hg/7) [Part3](https://data.mendeley.com/datasets/j8d8pfkvj2/4) | [Data_in_brief](https://www.sciencedirect.com/science/article/pii/S2352340923001671) |
| `MFPT` | 轴承 | 97656/48828 | 1 | `bearing` 结构中的振动信号 | [MFPT Fault Data Sets](https://www.mfpt.org/fault-data-sets/) |  |
| `PU` | 轴承 | 64 kHz | 1 | `Y` 结构中的振动信号 | [Paderborn Bearing DataCenter](https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter) | `Lessmeier2016` |
| `TORINO` | 轴承 | 51.2 kHz | 6 | `.mat` 中全部 6 个通道 |  |  |
| `UO` | 轴承 | 200 kHz | 2 | `Channel_1` |  |  |
| `XJTUSY` | 轴承 | 25.6 kHz | 2 | `Horizontal_vibration_signals`, `Vertical_vibration_signals` | [XJTU-SY Bearing Datasets](http://biaowang.tech/xjtu-sy-bearing-datasets/) |  |

## 使用的故障类

下表按当前 `data_scripts/` 实际读取的文件、目录或轴承编号整理。轴承数据集统一使用“内圈故障、外圈故障、滚子故障、保持架故障”的表述；原始数据中的 `ball` 统一记为“滚子故障”，`cage` 统一记为“保持架故障”。同一部位的不同损伤尺寸、严重程度或轴承编号在本表中按不同故障类列出。部分 pretrain 脚本只把这些类别的数据混合用于预训练，输出 Parquet 不一定保留 `labels` 字段。缩写中 `N` 表示正常，`IR` 表示内圈故障，`OR` 表示外圈故障，`RE` 表示滚子故障，`CA` 表示保持架故障，`MIX` 表示复合故障。

| 数据集 | 缩写总结 | 代码使用的故障类 |
| --- | --- | --- |
| `CNC` | `GOOD`, `BAD` | `good`、`bad`；非轴承数据，不转换为内圈/外圈/滚子故障。 |
| `CWRU` | `N`, `IR007`, `IR014`, `IR021`, `RE007`, `RE014`, `RE021`, `OR007`, `OR014`, `OR021` | 正常；内圈故障 0.007 inch、0.014 inch、0.021 inch；滚子故障 0.007 inch、0.014 inch、0.021 inch；外圈故障 0.007 inch、0.014 inch、0.021 inch。对应文件组分别为正常 `97`-`100`，内圈 `105`-`108`、`169`-`172`、`209`-`212`，滚子 `118`-`121`、`185`-`188`、`222`-`225`，外圈 `130`-`133`、`197`-`200`、`234`-`237`。 |
| `FEMTO` | `RUN_TO_FAILURE` | 代码使用 `Bearing1_1`-`Bearing1_7`、`Bearing2_1`-`Bearing2_7`、`Bearing3_1`-`Bearing3_3` 的全寿命退化数据；原始挑战数据没有给每个轴承固定的内圈/外圈/滚子/保持架标签，且可能同时包含滚子、内外圈和保持架退化，因此这里按“自然退化轴承实例”使用，不拆成单一故障部位类。 |
| `HITSM` | `N`, `IR2`, `IR5`, `IR8`, `OR2`, `OR5`, `OR8` | 正常；内圈故障 `IR2`、`IR5`、`IR8`；外圈故障 `OR2`、`OR5`、`OR8`。其中 `2`、`5`、`8` 表示故障区域圆心角尺寸，三个转速 `600`、`900`、`1200` rpm 不是故障类。`HITSM_self_built` 和 `HITSM_SpectraQuest` 使用同一套故障类。 |
| `IMS_FD` | `N`, `IR`, `RE`, `OR` | 正常 `bearing1_1`；内圈故障 `bearing1_3`；滚子故障 `bearing1_4`；外圈故障 `bearing2_1`。 |
| `KAIST` | `N`, `RE`, `IR`, `OR` | 正常；滚子故障；内圈故障；外圈故障。代码读取 `vibration_normal_*`、`vibration_ball_*`、`vibration_inner_*`、`vibration_outer_*`；`KAIST1`、`KAIST2`、`KAIST3` 只对应不同 CSV 子集。 |
| `MFPT` | `N`, `OR1`-`OR3`, `ORv1`-`ORv7`, `IRv1`-`IRv7` | 正常；外圈故障 `OuterRaceFault_1`-`OuterRaceFault_3`、`OuterRaceFault_vload_1`-`OuterRaceFault_vload_7`；内圈故障 `InnerRaceFault_vload_1`-`InnerRaceFault_vload_7`。 |
| `PU` | `N-K001`, `OR-KA04`, `OR-KA15`, `OR-KA16`, `OR-KA22`, `OR-KA30`, `IR-KI16`, `IR-KI17`, `IR-KI18`, `IR-KI21`, `MIX-KB23`, `MIX-KB24`, `MIX-KB27`, `MIX-KI14` | 正常 `K001`；外圈故障 `KA04` 2 x 3 mm、`KA15` <1 x <1 mm、`KA16` 2 x total mm 和 3 x total mm、`KA22` <2 x 1 mm、`KA30` <1 x <1 mm；内圈故障 `KI16` 6 x total mm、`KI17` 1 x 2 mm 两处、`KI18` 2.5 x total mm、`KI21` 1 x 2 mm；内圈+外圈复合故障 `KB23`、`KB24`、`KB27`、`KI14`。代码按这些轴承编号分别赋 label，因此不同编号保留为不同类。 |
| `TORINO` | `N`, `IR450`, `IR250`, `IR150`, `RE450`, `RE250`, `RE150` | 正常 `C0A`；内圈故障 `C1A` 450 um、`C2A` 250 um、`C3A` 150 um；滚子故障 `C4A` 450 um、`C5A` 250 um、`C6A` 150 um。代码只使用 `C0A`-`C6A` 文件，不使用 `E4A` endurance 文件。 |
| `UO` | `N`, `IR`, `OR`, `RE`, `MIX` | 正常 `healthy`；内圈故障 `inner`；外圈故障 `outer`；滚子故障 `ball`；复合故障 `combination`。 |
| `XJTUSY` | `OR-B1_1`, `OR-B1_2`, `OR-B1_3`, `OR-B2_2`, `OR-B2_4`, `OR-B2_5`, `OR-B3_1`, `OR-B3_5`, `IR-B2_1`, `IR-B3_3`, `IR-B3_4`, `CA-B1_4`, `CA-B2_3`, `MIX-B1_5`, `MIX-B3_2` | 外圈故障 `Bearing1_1`、`Bearing1_2`、`Bearing1_3`、`Bearing2_2`、`Bearing2_4`、`Bearing2_5`、`Bearing3_1`、`Bearing3_5`；内圈故障 `Bearing2_1`、`Bearing3_3`、`Bearing3_4`；保持架故障 `Bearing1_4`、`Bearing2_3`；内圈+外圈复合故障 `Bearing1_5`；内圈+外圈+滚子+保持架复合故障 `Bearing3_2`。 |

### Fault Classes Used

The table below summarizes the files, folders, or bearing IDs actually consumed by `data_scripts/`. Bearing datasets use unified terms: inner-race fault, outer-race fault, rolling-element fault, and cage fault. The original `ball` label is normalized to rolling-element fault, and `cage` is normalized to cage fault. Different damage sizes, severities, or bearing IDs are treated as different fault classes. Some pretrain scripts mix these classes for pretraining and do not necessarily keep a `labels` column in the output Parquet files. In the abbreviation column, `N` means normal, `IR` means inner-race fault, `OR` means outer-race fault, `RE` means rolling-element fault, `CA` means cage fault, and `MIX` means compound fault.

| Dataset | Abbreviation summary | Fault classes used by the code |
| --- | --- | --- |
| `CNC` | `GOOD`, `BAD` | `good`, `bad`; this is not a bearing dataset, so its labels are not converted to inner-race, outer-race, rolling-element, or cage faults. |
| `CWRU` | `N`, `IR007`, `IR014`, `IR021`, `RE007`, `RE014`, `RE021`, `OR007`, `OR014`, `OR021` | Normal; inner-race fault 0.007 inch, 0.014 inch, 0.021 inch; rolling-element fault 0.007 inch, 0.014 inch, 0.021 inch; outer-race fault 0.007 inch, 0.014 inch, 0.021 inch. The corresponding file groups are normal `97`-`100`, inner race `105`-`108`, `169`-`172`, `209`-`212`, rolling element `118`-`121`, `185`-`188`, `222`-`225`, and outer race `130`-`133`, `197`-`200`, `234`-`237`. |
| `FEMTO` | `RUN_TO_FAILURE` | The code uses full run-to-failure data from `Bearing1_1`-`Bearing1_7`, `Bearing2_1`-`Bearing2_7`, and `Bearing3_1`-`Bearing3_3`. The original challenge data does not assign a fixed inner-race, outer-race, rolling-element, or cage label to each bearing, and degradation may involve rolling elements, races, and cage at the same time. Therefore these data are used as natural degradation bearing instances rather than single-location fault classes. |
| `HITSM` | `N`, `IR2`, `IR5`, `IR8`, `OR2`, `OR5`, `OR8` | Normal; inner-race faults `IR2`, `IR5`, `IR8`; outer-race faults `OR2`, `OR5`, `OR8`. The numbers `2`, `5`, and `8` denote the angular size of the fault area; speeds `600`, `900`, and `1200` rpm are not fault classes. `HITSM_self_built` and `HITSM_SpectraQuest` use the same fault classes. |
| `IMS_FD` | `N`, `IR`, `RE`, `OR` | Normal `bearing1_1`; inner-race fault `bearing1_3`; rolling-element fault `bearing1_4`; outer-race fault `bearing2_1`. |
| `KAIST` | `N`, `RE`, `IR`, `OR` | Normal; rolling-element fault; inner-race fault; outer-race fault. The code reads `vibration_normal_*`, `vibration_ball_*`, `vibration_inner_*`, and `vibration_outer_*`; `KAIST1`, `KAIST2`, and `KAIST3` only correspond to different CSV subsets. |
| `MFPT` | `N`, `OR1`-`OR3`, `ORv1`-`ORv7`, `IRv1`-`IRv7` | Normal; outer-race faults `OuterRaceFault_1`-`OuterRaceFault_3` and `OuterRaceFault_vload_1`-`OuterRaceFault_vload_7`; inner-race faults `InnerRaceFault_vload_1`-`InnerRaceFault_vload_7`. |
| `PU` | `N-K001`, `OR-KA04`, `OR-KA15`, `OR-KA16`, `OR-KA22`, `OR-KA30`, `IR-KI16`, `IR-KI17`, `IR-KI18`, `IR-KI21`, `MIX-KB23`, `MIX-KB24`, `MIX-KB27`, `MIX-KI14` | Normal `K001`; outer-race faults `KA04` 2 x 3 mm, `KA15` <1 x <1 mm, `KA16` 2 x total mm and 3 x total mm, `KA22` <2 x 1 mm, `KA30` <1 x <1 mm; inner-race faults `KI16` 6 x total mm, `KI17` two 1 x 2 mm damages, `KI18` 2.5 x total mm, `KI21` 1 x 2 mm; inner-race + outer-race compound faults `KB23`, `KB24`, `KB27`, `KI14`. The code assigns labels by bearing ID, so different IDs remain different classes. |
| `TORINO` | `N`, `IR450`, `IR250`, `IR150`, `RE450`, `RE250`, `RE150` | Normal `C0A`; inner-race faults `C1A` 450 um, `C2A` 250 um, `C3A` 150 um; rolling-element faults `C4A` 450 um, `C5A` 250 um, `C6A` 150 um. The code only uses `C0A`-`C6A` files and does not use `E4A` endurance files. |
| `UO` | `N`, `IR`, `OR`, `RE`, `MIX` | Normal `healthy`; inner-race fault `inner`; outer-race fault `outer`; rolling-element fault `ball`; compound fault `combination`. |
| `XJTUSY` | `OR-B1_1`, `OR-B1_2`, `OR-B1_3`, `OR-B2_2`, `OR-B2_4`, `OR-B2_5`, `OR-B3_1`, `OR-B3_5`, `IR-B2_1`, `IR-B3_3`, `IR-B3_4`, `CA-B1_4`, `CA-B2_3`, `MIX-B1_5`, `MIX-B3_2` | Outer-race faults `Bearing1_1`, `Bearing1_2`, `Bearing1_3`, `Bearing2_2`, `Bearing2_4`, `Bearing2_5`, `Bearing3_1`, `Bearing3_5`; inner-race faults `Bearing2_1`, `Bearing3_3`, `Bearing3_4`; cage faults `Bearing1_4`, `Bearing2_3`; inner-race + outer-race compound fault `Bearing1_5`; inner-race + outer-race + rolling-element + cage compound fault `Bearing3_2`. |

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

## 全局参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--datasets` | `all` | 要处理的数据集，支持 `all` 或多个数据集名 |
| `--raw-root` | `Raw_data` | 原始数据根目录 |
| `--save-root` | `Process_data` | 处理后数据输出根目录 |
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

Parquet 文件通常包含以下字段：

| 字段 | 说明 |
| --- | --- |
| `samples` | 一条或多通道时序样本 |
| `labels` | 故障类别标签；部分预训练数据可能不包含该字段 |
| `dataset` | 数据集或划分标识 |

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
- 其他统一参数：`resampled_size`、`train_size`、`val_size`、`test_size`、`seed`、`fewshot_seed`

新增数据集步骤：

1. 在 `data_scripts/` 下新增数据集脚本。
2. 在脚本中添加 `DATASET_CONFIG`。
3. 在 `data_scripts/__init__.py` 的 `DATASET_MODULES` 中注册模块名。

`data_scripts/` 下提供了两个扩展示例：

- `example_pretrain.py`：pretrain 数据集模板，输出 `train.parquet`、`val.parquet`、`test.parquet`。
- `example_finetune.py`：finetune 数据集模板，输出 `train.parquet`、`train_1p.parquet`、`val.parquet`、`test.parquet`，并演示 `fewshot_seed` 的使用。

这两个示例默认不注册到 `DATASET_MODULES`，不会被 `python main.py --datasets all` 执行。
