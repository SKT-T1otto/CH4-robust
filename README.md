# CH4-robust：多AUV动力学不确定性下的鲁棒强化学习

这是独立的第四章研究项目，包含完整任务环境、地图/规划、强化学习底座和SIRSA A+B实现，可独立克隆开发。源自正在演化的第三章快照；本仓库不自动同步第三章，也不依赖工作区的其他项目目录。

## 从这里开始

- [第四章入口与快照边界](CH4_START_HERE.md)
- [A+B v2实现、对照、消融及运行命令](docs/chapter4/implementation_ab_v2_20261113/README.md)
- [2026-11-15机制验证报告](docs/chapter4/implementation_ab_v2_20261113/VALIDATION.md)
- [34页v2研究方案](docs/chapter4/research_plan_v2_20261113/output/pdf/CH4_SIRSA_AB_详细研究方案_20261113_v2.pdf)
- [v1实现与原SIRSA适配边界](docs/chapter4/README.md)

## 完整代码范围

| 路径 | 内容 |
|---|---|
| `core/` | 原四艇环境、动力学、任务判定、地图和公共规划接口 |
| `chapter3_bser/` | 第四章所需的冻结强化学习/任务运行底座及其父版本实现 |
| `chapter4_interval_robust/` | 独立SAC、v1区间方法、v2结构化集合A与时序一致辨识B |
| `configs/`、`checks/` | 原底座与第四章配置、对照/消融、机制检查和夹具 |
| `cli.py`、`ch4_cli.py`、`tools/`、`scripts/` | 底座和第四章运行、审计与验证入口 |
| `provenance/`、`_snapshot/`、`verification/` | 不可变父版本映射、来源演化和验证器依赖的证据 |
| `docs/`、`references/legacy_ch4_source/` | 研究/实现说明、原计划及只读旧CH4参考；后者不进入运行依赖 |

## 本地或Linux启动

使用独立Python环境；已验证环境为Python 3.10.20、NumPy 2.2.6和PyTorch 2.11.0。依赖清单见`requirements.txt`及`requirements-ch4.txt`。Linux须按本机CUDA/CPU环境安装对应PyTorch，不自动安装原SIRSA的TensorFlow/MuJoCo研究栈。

```text
git clone https://github.com/SKT-T1otto/CH4-robust.git
cd CH4-robust
python -B ch4_cli.py source-verify --execute --config configs/ch4/experiment_ab_v2.json
python -B ch4_cli.py matrix --config configs/ch4/experiment_ab_v2.json
python -B ch4_cli.py train --config configs/ch4/experiment_ab_v2.json --method ab_full --output work/ch4/manual_full_001
```

最后一条仅显示计划。各执行命令必须显式加`--execute`；机制练习再加`--mechanism`。正式研究训练须先取得当前源码、当前平台的原生验证收据。Linux可使用`python3`，完整命令见v2实现说明。

```text
python -B ch4_cli.py verify --execute --config configs/ch4/experiment_ab_v2.json --output work/ch4/manual_verify_n0_001
python -B ch4_cli.py verify --execute --config configs/ch4/experiment_ab_v2_n1.json --output work/ch4/manual_verify_n1_001
```

每个输出目录必须不存在。`work/`、缓存和检查点不随源码上传；克隆后应重新验证，旧Windows收据不能代替Linux验收。

## 当前研究状态

v2采用21D联合参数、165D合法历史输入、33D带候选执行器记忆的Critic上下文。A+B、共享×时序四格和Width/C/Memory消融已实现；N0/N1各20项机制检查、v1的22项回归检查及受限端到端恢复练习通过。

尚未完成长训练、正式多种子性能、原SIRSA复现或实物验证，不能据机制通过声称算法优于基线。父快照继承的path-option provenance和pure-MARL manifest问题保留，未通过重签旧清单消除。代码来源身份不等于论文成果验收。

原SIRSA论文与固定提交归属、许可证情况见[UPSTREAM](docs/chapter4/UPSTREAM.md)。上游下载归档留在本地`work/`，不在本仓库再分发；不补造上游许可证。第三方论文全文也不作为本次代码上传内容。

[继承的第三章README原文](docs/chapter4/github_release_20261115/README_INHERITED_ACTIVE.md)仅作来源历史，当前第四章操作请使用上述入口。
