# Linux 复现 · 冻结 MADDPG 残差幅度实验

在第四章仓库根目录、原先运行名义 MADDPG 的 Python 环境中执行。标准比较包含 32 个名义开发场景，每场使用三种幅度各运行一次，共 96 个完整回合。所有组均使用修复后的执行艇导航、第 16 回合冻结 Actor、同一组初始条件、原任务成功和障碍碰撞判据。每回合最多 400 步。各组闭环轨迹可以不同。

| 倍率 | 执行动作 |
|---|---|
| 0 | 四艇的学习残差精确置零，对应修复导航的 D2 参照 |
| 0.5 | 四艇的归一化 Actor 输出逐分量乘 0.5 |
| 1 | 四艇使用完整的冻结 Actor 输出 |

倍率作用在原有门控之前。地图、制导先验、执行器限幅、目标运动、奖励和成功条件沿用原配置。全程使用单线程 CPU、关闭探索，优化器更新次数为零。

## 运行

```bash
git pull --ff-only origin main
python -B -m ch4_nominal_baseline_v2.amplitude --check
python -B -m ch4_nominal_baseline_v2.amplitude --execute
```

默认从原 Linux 训练输出读取以下两个文件。它们保存在跑过 pilot 的机器上。

```text
~/ch4-nominal-dev/pilot_2729/attempt_3117d9f11bbf4d93a1be926b36397e7d/boundaries/000016/committed.json
~/ch4-nominal-dev/pilot_2729/attempt_3117d9f11bbf4d93a1be926b36397e7d/boundaries/000016/state.pt
```

文件放在其他位置时，传入含这两个文件的目录。可先验证身份，再正式运行。只有这两个检查点文件是仓库以外的实验输入。

```bash
python -B -m ch4_nominal_baseline_v2.amplitude --check --checkpoint /path/to/boundaries/000016
python -B -m ch4_nominal_baseline_v2.amplitude --execute --checkpoint /path/to/boundaries/000016
```

默认结果目录是 `work/ch4/nominal_amplitude_ep16/`。程序为每个场景的每种倍率分别保存结果。在同一台机器上重新执行同一命令，会校验已有记录并补跑缺项。若需先跑半幅组，可追加 `--scales 0.5`，随后运行默认完整命令。用 `--indices 4 15` 可选定开发场景。所有分批命令使用同一输出目录即可合并，旧导航和其他机器的记录不参与配对。如需新开独立试验，用 `--output work/ch4/新的目录名`。

## 查看结果

`plan.json` 记录三种倍率、权重摘要、代码、场景和平台身份。`scene_004_scale_0.5.json` 等文件记录原生终止原因、交接、最长连续保持、实际门控后的归一化残差幅度和轨迹摘要。96 条记录齐全后生成 `summary.json`。

| 汇总字段 | 回答的问题 |
|---|---|
| `by_scale` | 各倍率的成功数、总成功率、障碍碰撞数、超时数、交接数、交接后完成率和平均惩罚完成步数 |
| `paired.1_to_0.5` | 残差减半后，哪些场景新增成功、丢失成功、新增碰撞、消除碰撞 |
| `full_scale_gains_kept_at_half` | 完整 MADDPG 相对 D2 独有的成功，有哪些在半幅时保住 |
| `full_scale_losses_recovered_at_half` | D2 成功而完整 MADDPG 失败的场景，有哪些在半幅时恢复 |

若碰撞减少，同时任务成功或交接也减少，应一并报告这项取舍。32 场属于开发验证集，结论仅对应本次冻结权重。
