# SIRSA A+B v2代码实现与手动运行

本次实现位于`chapter4_interval_robust/ab_v2/`，显式选择`configs/ch4/experiment_ab_v2.json`或`experiment_ab_v2_n1.json`。默认入口和原v1配置仍保留，未覆盖旧模型、快照、原PDF或验证结果。第三章原目录不是运行依赖或输出目录。

2026-11-15续接完成的源码版本、实际检查及短管线记录见[VALIDATION.md](VALIDATION.md)。旧v5-v8记录保留，不能把旧源码收据当作当前源码的训练准入。

## 实现内容与身份

| v2方案主题 | 实际模块 | 本次行为 |
|---|---|---|
| 参数与schema | parameters.py、configuration.py | 原9D物理适配；每车6D/联合21D；真实/候选33D上下文；165D Actor、705D双Q |
| A | estimation.ConditionalEstimator | 共享g分层，按艇保留个体解释；每个延迟均有候选；不足支持先局部再提议，空集诚实退回先验 |
| B | estimation、runtime.Intervention | 用已发命令索引候选延迟/迟滞；有限窗初态不确定性；记录决策前真实滤波状态，仅训练诊断可读 |
| 历史和信息边界 | History、PolicyInput、runtime | 8×13三态status；-1补帧、0已观察但排除辨识、1辨识有效；当前输入只含已结束转移 |
| 学习 | learning.py | 随机SAC、精确tanh密度、事实双Q/target/熵温度、候选记忆风险查询，整行所有场景支持门控 |
| 朴素参照 | identifier.py | 6D独立监督ensemble；均值/散度是未校准启发式；非原SIRSA复现 |
| 对照/消融 | 两份v2配置 | D2、UDR、Point、Fixed-set、naive；S0T0/S1T0/S0T1/S1T1；A/B/C/Width/Memory开关 |
| 运行与恢复 | runner、checkpoint、cli | 计划优先、显式执行、新目录、同源码/配置回合边界完整恢复；拒绝v1文件 |
| 机制与统计 | validation、ab_v2_tests、metrics | 原生检查、实际低尾、预动作框覆盖/条件支持、恢复竞争终点、按训练种子汇总 |

物理固定项：垂向共享增量0、限速尺度1、噪声尺度是协议已知常量。N0=0，N1=0.01，分别训练/评估；没有宣称识别原9个独立参数。水平g、阻力、能力、净垂向偏置、迟滞、整数延迟沿用PDF候选域。原基础流场与随机流在新增扰动关闭后保留。

## 重要实施选择与方案差异

1. **回放使用冻结场景银行。** 每个事实决策前，从当时完整条件池抽N=32组联合(theta21,推断z12)，保存33D银行、有效性、来源哈希与S0局部先验回退数。每次Actor更新重用这一有限银行，不把盒边界重抽样冒充原条件集合。该设计降低回放内存；目标是固定Monte Carlo近似，不能宣称与每次重采样相同。
2. B按合法8步窗重建每个候选的滤波响应。窗口初态中点与衰减半径进入经验容差；参数突变不读取真实事件，而由公开残差空集触发窗口清除、候选重建和暖机。命令时间戳仍保留。不是严格区间可达性或覆盖定理。
3. Fixed-set的公开先验是声明的有效集合；未训练ensemble和数据不足的在线集合无效。输入`valid`不等于统计置信水平。
4. 保护首版仅支持所有组共同关闭；`finite_probe`被拒绝，防止沿用v1不匹配的预测器。新增候选保护需独立研发。
5. 主留出域支持流场、响应失配；当前三项延迟mask只覆盖0/1/2，不把3步延迟实验伪装成已支持schema。
6. 当前任务时限沿用父终止。SAC本身对collector truncation保留bootstrap；采集器若仅截断且未获得真实下一context会拒绝，而非伪造标签。一般中途collector恢复尚不支持。
7. `identify`已实现同一合法原生轨迹上的四组辨识重放。独立校准工具、ACDA/传统观测器完整基线、自动统计置信集合和正式性能矩阵尚未实现或执行。

## 对照的实际开关

S0不汇总其他艇的辨识约束；S1公共g需各有数据艇分别有解释。T0只在辨识残差中假设当步命令立即作用；T1按候选delay/lag传播。真实执行器、Actor历史、风险物理模型和候选预算保持。S0风险场景每次只抽一个公共g，并对没有该g支持的艇使用声明的局部先验与重建记忆，回退数可见。

启用局部补提议时，四格每个窗口都执行同样的两遍物理预测；无稀疏池时第二遍不改变候选。重置后不足8帧的窗口执行真实但掩码排除的补齐预测，避免历史变短使某组累计预算减少；这些调用不推进滤波状态，也不增加数据支持。`prediction_calls`包含补齐开销，`padding_prediction_calls`单报补齐，`evidence_prediction_calls`单报实际历史。相同轨迹上的总预测次数因此可核对相等，实际耗时仍需报告。有限窗初态带来的保守残差容差在四格保持相同，T0只改变均值预测的执行时序。

`width_off`仅修改Actor所见六个宽度，Critic输入与银行原样保留；`c_off`仅alpha=1；`memory_off`仅屏蔽Actor历史，不取消估计器B。所有性能比较均须各自重训。别名`ab_full`和`s1t1`、`a_off`和`s0t1`、`b_off`和`s1t0`只是便于实验命名，不应重复计为独立算法证据。

## 可复制命令

Windows进入本目录；Linux进入迁移后的同项目根目录，使用`python3`替代`python`即可。此处python应是包含原环境依赖和PyTorch的独立环境，不自动安装。

```powershell
Set-Location -LiteralPath 'E:\gym\code\WORKSPACE\AUV-Thesis-CH4-IntervalRobust'
```

只显示计划：

```text
python -B ch4_cli.py matrix --config configs/ch4/experiment_ab_v2.json
python -B ch4_cli.py train --config configs/ch4/experiment_ab_v2.json --method ab_full --output work/ch4/manual_ab_full_001
```

原生机制验证，输出必须不存在；N1需独立验证：

```text
python -B ch4_cli.py verify --execute --config configs/ch4/experiment_ab_v2.json --output work/ch4/manual_ab_verify_001
python -B ch4_cli.py verify --execute --config configs/ch4/experiment_ab_v2_n1.json --output work/ch4/manual_ab_n1_verify_001
```

受限端到端机制练习，不作性能比较：

```text
python -B ch4_cli.py train --execute --mechanism --config configs/ch4/experiment_ab_v2.json --method ab_full --output work/ch4/manual_ab_mechanism_001
python -B ch4_cli.py evaluate --execute --mechanism --config configs/ch4/experiment_ab_v2.json --method ab_full --split dev --weights work/ch4/manual_ab_mechanism_001/checkpoint_00002.pt --output work/ch4/manual_ab_mechanism_eval_001
```

同轨迹辨识检查：

```text
python -B ch4_cli.py identify --execute --mechanism --config configs/ch4/experiment_ab_v2.json --split calib --output work/ch4/manual_ab_identify_001
```

未来人工正式研发模板（本次不执行；仍需开发集校准阈值、任务能力和预算）：

```text
python -B ch4_cli.py train --execute --config configs/ch4/experiment_ab_v2.json --method ab_full --seed 7291 --validation work/ch4/manual_ab_verify_001/validation.json --run-label ab_development_001 --output work/ch4/manual_ab_full_001
python -B ch4_cli.py evaluate --execute --config configs/ch4/experiment_ab_v2.json --method ab_full --split dev --weights work/ch4/manual_ab_full_001/checkpoint_00100.pt --run-label ab_development_001 --output work/ch4/manual_ab_full_dev_001
python -B ch4_cli.py summarize --execute --config configs/ch4/experiment_ab_v2.json --inputs work/ch4/manual_ab_full_dev_001 --output work/ch4/manual_ab_summary_001
```

最终测试另需`--split test --ack-final-test --run-label <预先冻结标签>`。评估不能改变网络、噪声协议、公开先验、训练种子、时限或冻结合同；可以选dev/test、已声明物理域和扰动协议。恢复只在相同源码和有效配置下使用`--resume`，输出另选新目录。

来源记录使用已有source-record/source-verify；只有实际已审查改动后才追加，不通过重签修复父失败。v1/v2检查点都绑定精确当前来源，代码升级后不能将旧机制权重当作当前权重。

## 验证、限制与成果边界

最终执行证据见本目录`VALIDATION.md`及其链接；新旧来源演化见`provenance/ch4/source_v0005.json`起。首次轴顺序错误等开发失败日志完整保留在独立目录，未覆盖。代码完成与机制通过不等于A+B增益、名义能力、原SIRSA复现或创新成立。

所有阈值、候选数和匹配半径仍未正式校准；有限候选可能漏真值，剩余参数可能弱可辨识。Q支持只是经验门，不是外推证书；scene bank的Q低尾不是精确轨迹CVaR。当前简化三维平移物理没有实物标定。原父源码两项来源失败继续保留，原始清单与_snapshot不改写。

风险支持门比较参数、滤波记忆、公开状态、阶段/活动状态、历史摘要、集合摘要及动作邻域，再检查双Q分歧。`risk.radii.set=0.4`仍是待校准候选；任何一个场景不受支持则整行回退，不能删去困难场景再计算原测度低尾。分析型Critic测试可检查风险分支确实使用低尾和整行回退，但不代替真实任务的支持覆盖。

恢复统计把预定任务超时视为观察截止，碰撞/失败则单列竞争终点；这不改变原环境的terminated或Bellman bootstrap语义。汇总拒绝混合不同训练扰动协议，并只在同权重、训练种子、场景和域下配对名义回报差。当前输出是描述性统计，不提供已校准显著性或正式性能验收。
