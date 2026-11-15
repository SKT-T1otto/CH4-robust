# 第四章：区间条件多智能体鲁棒学习

后续SIRSA A+B v2代码现通过显式`--config configs/ch4/experiment_ab_v2.json`选择，参见[实施、对照与运行说明](implementation_ab_v2_20261113/README.md)。下文保留v1说明；v2的输入、参数、检查点及验证收据具有独立身份，不能混用。

本目录说明 `AUV-Thesis-CH4-IntervalRobust` 中新增的独立 CH4 实现。入口是 `ch4_cli.py`，运行模块是 `chapter4_interval_robust/`，配置是 `configs/ch4/experiment.json`。它采用独立 PyTorch 多智能体 SAC：四个随机 Actor、集中式联合动作双 Critic、在线公开历史集合估计与有经验支持门槛的低回报尾部目标。

这里的 `sirsa_naive` 是参考 SIRSA 思想重新实现的监督 ensemble 朴素多智能体适配。原 SIRSA 的单智能体 TensorFlow/softlearning 实验尚未复现；接口检查、原生环境机制检查和少量更新均不能替代原工作复现、名义任务能力或正式鲁棒性能验收。当前性能状态仍为 `REVIEW_REQUIRED`，`performance_passed=false`。本次未启动正式训练，也没有 GitHub 写入。

## 阅读顺序与保护范围

先读本副本根目录 `AGENTS.md`、`CH4_START_HERE.md`，再读 [DESIGN.md](DESIGN.md)、[INFORMATION.md](INFORMATION.md)、[LIMITS.md](LIMITS.md) 和 [UPSTREAM.md](UPSTREAM.md)。实际检查结果、平台身份及执行量以本次 [VALIDATION.md](VALIDATION.md) 和其引用的 JSON/日志为准。

本次新 CH4 模块依据用户的后续明确授权开发，不改写根 AGENTS 的历史文本。`core/`、`chapter3_bser/`、原配置及来源证据保持原身份；Active 和旧 CH3/CH4/CH5 不是新运行的输出位置。新实现只在自己创建的环境实例上接入物理扰动，不将旧 CH4、下载的 SIRSA 或历史结果导入运行链。第三章继续独立演化；本目录没有自动同步。

快照继承的两项来源失败仍保留：`path_option_identity` 的 provenance 不匹配，以及 `pure-MARL source manifest/inventory mismatch`。新增 CH4 来源链只记录本次增量和冻结父文件，不能把父快照重新签成通过。

## 命令约定

以下命令在本项目根目录执行。Windows PowerShell 可先运行：

```powershell
Set-Location -LiteralPath 'E:\gym\code\WORKSPACE\AUV-Thesis-CH4-IntervalRobust'
```

Linux 请先进入迁移后的本项目根目录；下方所有 `python -B ch4_cli.py ...` 命令同时适用于 Windows/Linux。Linux 若解释器名是 `python3`，将命令开头的 `python` 替换为 `python3`。示例使用 Python 的 `-B` 参数避免创建 `.pyc` 缓存；它不会减少训练预算或改变学习语义。使用已准备好的独立环境；依赖候选记录在 `requirements-ch4.txt`，这里的文档与验证流程不自动安装依赖。

除 `matrix` 外，所有命令默认只打印计划。加 `--execute` 才执行；`matrix` 即使传入 `--execute` 也只列出矩阵，不启动子进程。输出必须是本副本 `work/ch4/` 下**从未存在的新目录**，重复运行应改名；不要清理或覆盖旧目录。下面的 `_001` 是示例身份，执行前确认尚不存在。

### 1. 查看计划与实验矩阵

```text
python -B ch4_cli.py matrix
python -B ch4_cli.py train --method full --output work/ch4/manual_full_001
python -B ch4_cli.py evaluate --method nominal --protocol nominal --split dev --output work/ch4/manual_nominal_001
```

`matrix` 列出方法、四种协议、三个场景划分、三个物理域和当前候选预算，不表示这些组合已训练或评估。

### 2. 记录、核验增量来源

```text
python -B ch4_cli.py source-record --execute --reason "Reviewed independent CH4 source and frozen parent boundary"
python -B ch4_cli.py source-verify --execute
```

只有新实现尚无来源记录，或源码确有已经审查的增量时才运行 `source-record`；没有变化会拒绝重复记录。它按序新增 `provenance/ch4/source_vNNNN.json`，不使用 `--output`，不覆盖父记录。`source-verify` 核对来源链、当前增量文件、冻结父文件和快照证据；它不宣称父来源失败已解决。修改受来源管理的源码、配置、检查或依赖清单后，需要先追加真实原因的来源记录，再重新验证。

### 3. 在当前平台执行受限验证

```text
python -B ch4_cli.py verify --execute --output work/ch4/manual_verify_001
```

结果为 `validation.json`、`tests.log` 和 `mechanism_ledger.json`。退出码为零、`passed=true` 且原生环境检查成立才满足准入的一部分。正式训练还核对当前来源、`platform.system()` 和验证附件哈希。Windows 的收据不能直接供 Linux 训练使用；Linux 迁移后应重新运行本节。验证不启动正式训练，具体物理步数和每模型更新数看 ledger。

本次交付证据归档位置为 `work/ch4/validation_windows_v4/validation.json`、`work/ch4/end_to_end_full_v4/`、`work/ch4/end_to_end_eval_v4/` 和 `work/ch4/end_to_end_resume_v4/`；各项最终执行状态以本次 VALIDATION 及其原始记录为准，目录名称本身不表示验收通过。后续手动示例仍使用新的 `manual_*` 目录，避免覆盖上述证据。

可单独运行端到端机制练习；`--mechanism` 将训练/评估配置固定为 2 回合 × 最多 16 步、训练最多 4 次完整多智能体更新，并使用对应的小模型设置，不作性能比较。评估机制检查点时同样必须传 `--mechanism`，否则模型/预算合同不匹配：

```text
python -B ch4_cli.py train --execute --mechanism --method full --protocol piecewise --output work/ch4/manual_mechanism_001
python -B ch4_cli.py evaluate --execute --mechanism --method full --protocol piecewise --split dev --weights work/ch4/manual_mechanism_001/checkpoint_00002.pt --output work/ch4/manual_mechanism_eval_001
```

### 4. 手动正式研究运行

以下是未来人工启动的命令模板，本次没有执行。默认训练 100 回合、每回合最多 400 步；该预算仍是候选，名义成功门槛尚未冻结。先检查计划与本机最新验证，再决定是否执行。实际实验应预先冻结种子、预算、检查点规则、比较标签和测试协议。

```text
python -B ch4_cli.py train --execute --method full --domain in_domain --protocol episode --validation work/ch4/manual_verify_001/validation.json --run-label ch4_candidate_001 --output work/ch4/manual_full_001
python -B ch4_cli.py evaluate --execute --method full --protocol nominal --split dev --weights work/ch4/manual_full_001/checkpoint_00100.pt --run-label ch4_candidate_001 --output work/ch4/manual_full_nominal_dev_001
python -B ch4_cli.py evaluate --execute --method full --protocol episode --split dev --weights work/ch4/manual_full_001/checkpoint_00100.pt --run-label ch4_candidate_001 --output work/ch4/manual_full_episode_dev_001
python -B ch4_cli.py evaluate --execute --method nominal --protocol nominal --split dev --run-label ch4_candidate_001 --output work/ch4/manual_reference_dev_001
```

学习方法必须显式提供本副本生成的匹配 `--weights`；不会用随机初始化权重冒充受训方法。评估必须保留检查点的训练种子、场景种子块、时限、参考配置、方法和学习/估计/风险/保护等合同；可选择开发/最终测试划分、预定义物理域和扰动协议，并改变 `evaluation_episodes`。`nominal` 为无学习的 OLD/D2 公开参考，不需要权重。各学习消融必须独立训练，不用推理时开关代替重训。评估固定 Actor/identifier 权重，只递推公开历史及集合。

训练种子默认 7291；可用 `--seed` 覆盖，不必修改受来源管理的配置文件。不同种子要使用各自的新训练目录和检查点，评估时传回该检查点同一训练种子。以下仍是只打印计划的例子：

```text
python -B ch4_cli.py train --method full --seed 7292 --output work/ch4/manual_full_seed7292_001
python -B ch4_cli.py evaluate --method full --seed 7292 --split dev --weights work/ch4/manual_full_seed7292_001/checkpoint_00100.pt --output work/ch4/manual_full_seed7292_dev_001
```

训练只能使用 `--split train` 和 `--domain in_domain`。开发/测试场景种子块与训练分开；`--domain heldout_flow`、`heldout_response` 仅供评估，保持原公开先验，改变环境真实采样域，故意检验先验失配：

```text
python -B ch4_cli.py evaluate --execute --method full --protocol episode --domain heldout_flow --split dev --weights work/ch4/manual_full_001/checkpoint_00100.pt --run-label ch4_candidate_001 --output work/ch4/manual_full_flow_dev_001
python -B ch4_cli.py evaluate --execute --method full --protocol piecewise --domain heldout_response --split dev --weights work/ch4/manual_full_001/checkpoint_00100.pt --run-label ch4_candidate_001 --output work/ch4/manual_full_response_dev_001
```

最终测试必须先冻结选择规则和标签，显式确认；不得看测试结果后再调参：

```text
python -B ch4_cli.py evaluate --execute --method full --protocol episode --split test --ack-final-test --run-label ch4_frozen_test_001 --weights work/ch4/manual_full_001/checkpoint_00100.pt --output work/ch4/manual_full_test_001
```

### 5. 汇总与回合边界恢复

```text
python -B ch4_cli.py summarize --execute --inputs work/ch4/manual_full_nominal_dev_001 work/ch4/manual_full_episode_dev_001 work/ch4/manual_full_flow_dev_001 --output work/ch4/manual_summary_001
```

汇总按 `method:split:protocol:domain` 分组，核对来源、比较合同哈希、实际检查点训练回合数、先验/采样域、动作权限、保护配置和比较标签，拒绝同种子同场景重复计数。比较合同冻结时限、参考/场景设置、估计/风险等配置及训练扰动协议，允许汇总多个训练种子，不把其他设置差异混成种子波动。名义差值只配对相同检查点、场景、训练种子、物理域；不会把不同方法/域合并成一个成功率。单训练种子下的场景波动不等于独立训练重复。

若某次正式训练在第 50 回合检查点后中断，可在**相同源码与完全相同有效配置**下从新的输出目录继续原 100 回合预算：

```text
python -B ch4_cli.py train --execute --method full --domain in_domain --protocol episode --resume work/ch4/manual_full_001/checkpoint_00050.pt --validation work/ch4/manual_verify_001/validation.json --run-label ch4_candidate_001 --output work/ch4/manual_full_resume_001
```

该例以第 50 回合检查点实际存在为前提。恢复只支持完整回合结束、下一回合复位之前的边界；保存网络、target、优化器、温度、回放、经验支持库和随机状态。环境、估计器历史、执行器队列及迟滞在下一回合重新初始化，不支持回合中途恢复，也不接受历史第三章权重、下载的 pickle 或跨来源重标权重。不能通过恢复命令更改原总预算。

恢复目录会新建承接的旧 `episode_*.json`，使其单独汇总得到完整训练历史；旧场景/逐步轨迹仍在原目录。`inherited_physical_steps` 与 `physical_steps_executed_this_run`、`updates_executed_this_run` 区分承接量和本次实际执行量。汇总时选恢复后的完整目录，不把旧目录前缀再次并入同组。

## 输出解释

`run.json`/`config.json` 记录来源、平台、方法、控制权限和配置；`scene_*.json` 记录场景；`transitions.jsonl` 包含公开转移及明确命名的特权诊断；`updates.jsonl` 记录 loss、温度、尾部支持/回退数量；`episode_*.json` 和 `summary.json` 报告原成功/碰撞、原团队回报、经验覆盖、恢复删失和耗时。恢复汇总给出事件数、未恢复率和已恢复事件的条件平均步数；窗口不能跨下一次参数变化，无事件时比率为 `null`。`checkpoint_*.pt` 及同名 `.json` 有自己的来源/哈希合同。不能从有梯度、loss 下降或检查通过推断已收敛、已安全或已达到论文性能。
