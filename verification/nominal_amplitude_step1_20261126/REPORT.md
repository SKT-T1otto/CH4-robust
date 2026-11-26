# 冻结残差幅度实验 · 实现验证

本次发布提供第四章 32 场开发集的 Linux 评估入口。验证在 Windows 的 AUV 环境完成，正式的三倍率 96 回合成绩由 Linux 手动运行生成。

| 验证项 | 结果 |
|---|---|
| 原权重 | 从历史第 16 回合、种子 2729 的已提交检查点核对训练来源和文件摘要，并核验冻结网络摘要 `0ffac82f732f2d977146f823c95225ed6f3fe2c26bc7a08bb19fd50c21c7c195`。单独导出 Git 暂存树，在干净检出中从外部路径核验同一检查点成功。 |
| 三档动作 | 四艇同观测下，三档输出准确对应零残差、原输出的一半、原输出。原生三步前缀核对了零档和全档与相应参照的轨迹摘要，半档逐步核对了经原门控的实际残差。 |
| 真实完整任务 | 使用冻结历史权重、倍率 0、验证场景 016，在第 20 步原生碰撞终止。所有 240 个开门动作分量均为零。全回合轨迹摘要与既有修复导航 D2 结果相同，为 `e3aea7630feb8aa6fd1de78e504528d0f983e28cece27fbbf29944bf9e884988`。 |
| 汇总和续跑 | 用确定性测试数据核对部分运行、补齐 96 项后才汇总、完整后重复运行、逐场配对、识别被修改的回执及错误权重边界。共 4 项自动测试通过。 |
| 受保护资产 | 对照已有内容清单逐一核验文件集合与 SHA-256。受保护 CH3 共 369 项、CH4 共 103 项、CH5 共 318 项，全部匹配。 |

本轮新增持久文件为 `ch4_nominal_baseline_v2/amplitude.py`、`ch4_nominal_baseline_v2/tests/test_amplitude.py`、`ch4_nominal_baseline_v2/AMPLITUDE.md`、本报告，以及本地 `work/ch4/nominal_amplitude_step1_verification_20261126/` 下的 `plan.json` 和 `scene_016_scale_0.json`。随代码发布的 `ch4_nominal_baseline_v2/__init__.py` 和 `ch4_nominal_baseline_v2/navigation.py` 来自前一轮已验证的本地实现。

复查命令为 `python -B -m unittest discover -s ch4_nominal_baseline_v2/tests -p test_amplitude.py -v`。完整执行说明见[Linux 运行说明](../../ch4_nominal_baseline_v2/AMPLITUDE.md)。
