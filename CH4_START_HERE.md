# 第四章开发项目：Interval Robust

这是按用户 2026-11-12 的明确请求，从 **AUV-Thesis-Active** 当前工作树创建的独立文件副本。
目录实际名称是 `AUV-Thesis-Active`，不是 `AUV-Thesis-activate`。

## 现在从哪里工作

- 2026-11-13后续授权：已新增显式配置选择的SIRSA A+B v2实现，入口和限制见[代码实施说明](docs/chapter4/implementation_ab_v2_20261113/README.md)。原v1路径、旧方案和验证证据保留；v2正式性能仍待验收。
- 2026-11-15续接：参数/schema、A+B与匹配对照/消融已落地；受限验收与仍待研究验证的边界见[验证报告](docs/chapter4/implementation_ab_v2_20261113/VALIDATION.md)。后续训练须使用当前源码和目标平台的v2收据，不混用历史权重或v1收据。

- 第三章继续在 `E:\gym\code\WORKSPACE\AUV-Thesis-Active` 修改和验证。
- 第四章后续在 `E:\gym\code\WORKSPACE\AUV-Thesis-CH4-IntervalRobust` 开发；两者没有共享文件、符号链接或自动同步。
- 2026-11-13：已在本副本新增独立第四章实现；当前入口为 `ch4_cli.py`，实现与运行说明见 [第四章开发指南](docs/chapter4/README.md)。第三章仍未在本任务中被认定为定稿，正式性能验收尚待完成。
- 根目录 README.md/AGENTS.md 与原项目逐字节保留。旧文档里的绝对路径仍可能指向 Active，执行命令前应先确认当前路径及输出路径。

## 已保留

当前已知状态：原项目和本副本的两项来源校验均未通过。`path_option_identity` 报告
`chapter3_bser/learning/provenance.py` 与已审查清单不一致；独立验证器报告
`pure-MARL source manifest/inventory mismatch`。本次未修复或重新签署任何源码清单。
完整错误和源/副本对照在 `_snapshot/validation_report.json`；这份目录是当前开发现场快照，后续训练前需处理来源一致性与执行准入。

1. `core/`、`chapter3_bser/` 的当前实现（包括 path_option、safe_rl、reach_avoid 和未提交的 baselines）。
2. `configs/`、`checks/`、`tools/`、`scripts/`、`requirements.txt` 和原入口文件。
3. `docs/`、`provenance/` 和 `verification/`；包括验证器引用的历史源码证据压缩包，原有来源身份不改写。
4. `references/research/` 中的第四章详细 PDF 方案、可检索全文、原参考论文 `1-223081.pdf` 和方案生成时的检查记录。
5. `references/legacy_ch4_source/` 中的旧 CH4 文本/代码，只供后续比较、迁移和重写参考，不导入当前运行链。

## 旧 CH4 的候选参考

- `registry/rbe_disturbance.py`、`utils/rbe_sampler.py`：扰动参数、采样和边界实验组织。
- `utils/reb_dataset.py`、`utils/reb_model.py`、`utils/rbe_metrics.py`：数据记录、边界模型与指标思路。
- `registry/ch4_stress_tiers.py`、`tools/eval_ch4_recoverable_boundary_grid.py`：分级扰动和边界评测组织。
- 旧 `env.py`、`train.py`、MADDPG 与配套工具：仅用于核对既有假设；尚未完成对当前代码的兼容性审查。

这些候选均未接入、未宣称可直接运行，也不继承旧模型与性能结论。旧数据/权重没有复制或重新命名。

## 后续开发顺序

初始快照的来源边界和第三章公共接口已保留；独立实现位于 `chapter4_interval_robust/`，
配置位于 `configs/ch4/`，来源演化位于 `provenance/ch4/`。实现和已完成检查以第四章文档为准。
当前快照同时保留第三章不同研究分支，复制完成不代表已经确定第四章最终底层策略。
第三章后续必要修复应逐项审查并迁移；不要把整个 Active 再覆盖到本目录。

后续运行前仍须完成所选路径的来源校验和原生执行准入；本次只做静态核验。
`work/` 是本副本自己的新输出位置；本次没有启动运行或建立指向旧输出的链接。

## 快照与保留范围

- 来源 HEAD：`18ea616ae256d2bc93f1bd640fe6bd46b5af8180`，分支 `main`；实际复制内容还包含当时未提交和未跟踪文件。
- `.git/` 未复制，没有创建分支、提交或远程；修改前的 HEAD、状态与补丁在 `_snapshot/` 中保留。
- 大型 `work/`、`3090结果/`、缓存未复制；原文件仍留在 Active。需用旧模型时先审查其来源清单。
- 复制清单：[copied_files.json](_snapshot/copied_files.json)；未复制清单：[excluded_files.json](_snapshot/excluded_files.json)。
- 核验结果：[VALIDATION.md](_snapshot/VALIDATION.md)；创建清单：[created_files.json](_snapshot/created_files.json)。
- `references/` 与原 `provenance/verification/` 是参考和历史证据，后续新增实验应另建输出与来源身份。

工作区总规则中生产目录的旧命名约定不由本次备份自动改写。本目录按本次用户指令建立为独立 CH4 开发准备副本；
本次用户另行明确授权在此副本开展第四章研发；旧仓库与第三章原目录仍受保护，不自动同步，不操作 GitHub。
