# 第四章独立名义 MADDPG

本目录包含独立名义基线的配置、几何登记、网络构造、第3步只读任务运行器、第4步学习更新和第5步探索采集。网络模块构造四个分布式Actor、四个集中式单Critic及目标网络。训练调度、完整训练状态恢复与任务性能验收按后续步骤实施。

任务配置引用本工程的 `configs/b2.json` 并校验文件SHA256。新算法拥有独立身份 `ch4.maddpg.single_critic.v1`。相对水速名义点为主训练和准入域，原生环境为回归参照。主域固定线性阻力、公共流幅值0.10 m/s、四艇阻力和水平推力倍率1、推进记忆系数0.15、零离散延迟及N0条件。

场景清单位于 `configs/ch4/nominal_maddpg_scenes.json`，包含训练1000场、验证32场、名义准入100场、失配开发24场和失配确认候选256场。最后一组仅登记候选几何，最终确认样本量在失配开发之后锁定。原D2已查看场景纳入历史排重。

源码记录覆盖当前核心环境、继承任务装配代码、配置、新模块、实验脚本和已消费的历史JSON。来源完整性与既有第三章来源校验的结果分别记录。新记录保留旧来源校验失败的事实。

从第四章项目根目录执行核验。

```text
python -B -m unittest discover -s checks/chapter4 -p test_nominal_maddpg.py -v
python -B -m chapter4_interval_robust.nominal_maddpg --regenerate-sample
```

首次建库和来源登记使用以下命令，已有目标文件会被拒绝覆盖。

```text
python -B -m chapter4_interval_robust.nominal_maddpg --build-scenes --workers 4
python -B -m chapter4_interval_robust.nominal_maddpg --register-source --regenerate-sample --workers 4 --report verification/nominal_maddpg_steps12_20261125/registration.json
```

核验逐场重算全部物理指纹，并检查跨组和历史重复。`--regenerate-sample` 按每组首、中、末三个固定索引重新调用生成器，共复核15场。`--regenerate` 可额外执行全部场景的独立生成复核。

第1、2步报告记录当时的配置、静态任务契约、网络构造、场景物理指纹和源码一致性验收。其配置保留当时的零回合、零更新状态，第3步使用下述独立运行协议。

## 第3步只读运行验收

`configs/ch4/nominal_maddpg.json` 保留第1、2步原始登记状态。第3步的回合范围、归档校准引用和零更新条件另存于 `configs/ch4/nominal_maddpg_runtime.json`。

`NominalRuntime.advance()` 推进一次原生闭环任务，将执行门之后的归一化残差、原团队奖励、刷新导航后的下一观测和共享终止标记直接写入外部传入的 `NominalReplay`。返回值保持真实转移字段，物理命令、迟滞、推力、阶段和任务结果保存在 `last_record`。Replay采样采用独立随机流。

验收包含8个完整只读回合。零残差分别重放历史超时、碰撞、成功校准轨迹，并重复超时场景。冻结初始化策略在成功校准场景和首个验证场景上各自独立复位重放两次。每对重放的一侧额外抽样Replay，用于检查采样次数对任务轨迹的影响。校准样例按历史终止类型选取，仅作运行器验收。

```text
python -B -m unittest discover -s checks/chapter4 -p test_nominal_runtime.py -v
python -B -m chapter4_interval_robust.nominal_maddpg.validate_runtime
python -B -m chapter4_interval_robust.nominal_maddpg.validate_runtime --register-source --execute --workers 4
```

来源演化追加新记录，原始来源与验收文件保留各自身份。第3步报告验证运行器和真实转移数据流，优化器更新及名义性能仍由后续阶段验收。

## 第4步学习更新与有界验收

`learner.py` 的 `SingleCriticLearner` 接收真实 `ExecutionSample`，验证四艇共享原团队奖励、共享终止标记、当前/下一步执行门和均匀样本权重。一次 `update(sample)` 使用同一批次先计算目标，再更新四个单Critic、具有有效样本的Actor，最后软更新所有目标网络。Actor的其他艇动作来自Replay中实际执行的归一化残差。仅开放门样本参与本艇Actor损失，无有效样本时不调用Adam.step，因此已有动量与优化器计数不变。关闭门不妨碍Critic学习原任务价值。终止样本直接以奖励为目标，不调用目标网络；非终止样本即使所有下一步动作门关闭仍自举价值。

采用原配置的MSE、Adam、gamma=0.95、tau=0.005、Actor学习率0.001和Critic学习率0.0005，无双Critic、奖励改标签、目标截断或目标动作噪声。Actor/Critic采用独立优化器，目标网络不接收梯度。更新必须在环境交互的no_grad上下文之外调用。数值异常会使调用方失败退出；当前更新接口不承诺优化器故障后的整轮事务回滚，也不提供训练恢复。

`configs/ch4/nominal_maddpg_learner.json` 单独登记第4步机制验收权限、数值容差和封存输入哈希，保留第1、2步配置和第3步运行协议原文。测试读取第3步五份日志的1,561条真实转移，含成功、碰撞和原生400步超时，并含冻结初始化策略的非零残差。它们只用于临时模型的有限机制检查，不作为未来训练数据或性能证据。每个临时模型至多四轮四艇更新，不保存权重或训练检查点，不读取名义准入/失配结果。

```text
python -B -m unittest checks.chapter4.test_nominal_learner -v
python -B -m chapter4_interval_robust.nominal_maddpg.validate_learner
python -B -m chapter4_interval_robust.nominal_maddpg.validate_learner --register-source --execute
```

`--register-source` 只在源码有新增变化时使用，追加来源记录；重复执行验收须用新的 `--output` 目录。默认不执行更新。NumPy参考计算独立核对TD目标、目标动作、MSE和Actor损失；独立优化器与直接公式实现连续四轮更新，逐轮核对参数、Adam状态及计数。验收覆盖学习实现，不表示已完成名义训练或达到任务成功率门槛。

## 第5步策略与探索采集

`policy.py` 的训练策略持有学习器的实时引用；评价策略深复制当前网络、冻结梯度并禁止探索。`noise.py` 使用独立CPU生成器实现OU噪声，归一化Actor输出依次加噪、限幅，再交给原生执行门、残差合成和物理过程。OU补充参数固定为theta=0.15、dt=1（每个决策步）、均值0。sigma沿用0.20、0.02下限及每团队环境步0.00002衰减，约9000步到达下限；回合复位只清推进探索过程的OU记忆，不复位噪声时钟或随机流。关闭探索时不抽样、不改变OU记忆，但训练决策时钟仍前进。评价对象没有训练噪声流。

`train_config.py` 从已封存场景表核对完整场景身份和物理指纹，提供初始化、场景排序、探索和Replay独立种子，以及私有场景排序器。环境外生事件沿用按场景/步索引的既有事件流。当前第5步协议只允许validation组评价，名义准入和失配组保持关闭。

`NominalRuntime` 默认仍是零探索的regression模式；train模式必须使用训练策略与`TrainingReplay`，evaluation模式必须使用冻结副本和独立Replay。训练Replay拒绝评价场景、改标签场景和缺少来源的写入；每个环形槽记录场景ID、种子、分组、物理指纹、完整场景哈希、回合号、时间步和写入编号。外部前缀截断使用`truncate()`，返回success=None、完成代价=None，并排除任务统计，不修改原转移done标记。学习更新始终在运行器的no_grad调用之外。

```text
python -B -m unittest checks.chapter4.test_nominal_exploration -v
python -B -m chapter4_interval_robust.nominal_maddpg.validate_collection
python -B -m chapter4_interval_robust.nominal_maddpg.validate_collection --register-source --execute --output verification/my_fresh_collection_check
```

有界集成验收最多8个前缀、每段32步、每个临时模型最多4轮更新。当前六段验收包含：两段同初态、同噪声的24步采集（各在第8步之后更新一次），其中一段插入独立8步评价；两段8步零探索一致性检查；一段显式关闭部分执行门的4步接口夹具。最后一项是人为门控测试，不作为自然任务表现。首次训练片段未自然出现关闭门的事实与失败记录保留。验证插入评价前后完整训练状态不变，以及下一步实际残差确实使用更新后的Actor。它不是正式训练，不保存模型权重，也不提供完整任务成功率结论。

## 第6步调度、边界恢复和评价审计

本步另建 `configs/ch4/nominal_maddpg_execution.json`，保持此前配置、零更新来源记录及验收原文不变。新的来源版本包含本步模块和测试。初始配置仍为batch=128、Replay容量500000、warmup=256团队环境步、每4步触发2轮更新、Actor/Critic学习率0.001/0.0005、Adam和tau=0.005。时钟包含关闭执行门的真实团队转移。第256步第一次触发，在当步真实转移入Replay之后依次完成两轮；第260步再触发两轮。每轮四个Critic各一步、Actor仅有效门样本更新，最后统一软更新目标网络。日志分别保存rounds、target_updates、actor_steps、actor_skips和critic_steps。

`trainer.py` 将每次尝试写入新目录，保存逐步实际残差、门、原奖励、终止和每轮采样索引、损失及计数。初始边界与原生终止后全部应执行更新完成的边界可以提交。`checkpoint.py` 保存网络、目标网络、Adam矩、Replay有效槽和环形指针、门与场景来源、抽样随机流、场景排列/游标/随机流、OU记忆/时钟/随机流、全局Python/NumPy/Torch CPU随机状态及累计计数。初始化种子与各独立随机流种子保留。环境事件按原有场景种子和时间步索引生成；边界无活动环境，检查点明确记录 `active_environment_saved=False`。

检查点目录独占创建，`state.pt`写完并同步后，最后写入`committed.json`作为提交标记。没有完整标记的目录不能恢复；字节哈希、状态哈希、源码、原配置、执行配置、场景表、算法身份和CPU运行环境都必须一致。恢复创建新尝试目录，先验证身份，随后加载到新的对象。禁止给活动回合或外部截断的状态保存可续跑检查点；优化器中途报错同样终止当前尝试。恢复会丢弃自上一个边界以来的未提交转移和更新，不恢复活动环境。

`report.py` 从提交标记和事件记录审计完成回合。只有开始、没有提交边界的回合记为中断，日志尾部被崩溃截断也不视为任务失败。未完成回合的success与完成代价均为null，不进入完整任务成功率。恢复之前的历史累计回合与当前尝试分别计数。`evaluate.py` 只接受冻结副本，使用独立环境和独立Replay，并在包括异常的路径上检查训练完整状态及活动环境身份不变。评价不抽训练Replay、不推进训练OU或场景排列；当前仅允许validation组。检查点选择必须包含同一场景表的全部32个完整验证场景，取平均惩罚完成步数最小者，同值取更早边界；短段评价没有选择资格。

```text
python -B -m unittest checks.chapter4.test_nominal_training_state -v
python -B -m chapter4_interval_robust.nominal_maddpg.validate_training_state --register-source --execute --output verification/new_training_state_check
python -B -m chapter4_interval_robust.nominal_maddpg.trainer
python -B -m chapter4_interval_robust.nominal_maddpg.trainer --execute --output verification/new_bounded_attempt --episodes 1 --step-limit 32
python -B -m chapter4_interval_robust.nominal_maddpg.trainer --execute --resume verification/new_bounded_attempt/boundaries/000000 --output verification/new_resumed_attempt --step-limit 32
python -B -m chapter4_interval_robust.nominal_maddpg.report verification/new_bounded_attempt
```

当前执行登记只授权有界机制运行，命令行默认不执行，不能通过增大命令行预算绕过8个前缀、每段32步、每个模型4轮更新的限制。正式训练仍需后续独立执行登记。本步不缩短warmup来制造“学起来了”的结果。

验收范围刻意分开：带非零Adam状态的连续/恢复比较使用明确标记的合成调度回合夹具，运行真实学习器、OU、Replay、调度器和保存恢复；第256步后保存，两条分支在第260步后比较动作、下一批索引、损失、参数和全状态，分别最多4轮。它不计作水下机器人任务表现。另外，真实环境分别从初始边界运行两段8步，其中恢复分支插入4步独立评价；单元测试另在具有256条合成Replay和两轮Adam状态的训练器旁运行2步真实评价，验证非空状态也不改变。它们没有原生完整任务成绩。因此，本步证明保存恢复的机制性质，尚不证明真实长任务经过多回合学习后的端到端复现，更不证明名义任务达标。

## 第7步 Linux 有限开发训练

在第四章仓库根目录、已安装 requirements-ch4.txt 所列 NumPy 和 PyTorch 的 Python 环境中执行。训练和评价沿用单线程 CPU。输出目录可放在有足够空间的本地磁盘，每个回合保留含有效 Replay 的边界检查点。下列命令使用用户目录下的同一研究目录。无 `--execute` 时仅打印登记方案。

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -B -m chapter4_interval_robust.nominal_maddpg.development pilot --execute --output "$HOME/ch4-nominal-dev"
```

命令先在同一组 32 个 validation 场景、同一名义相对水速动力学上，分别运行 D2 和与 pilot 同初始化的冻结 Actor。评价关闭探索，每艇只读取原 28 维观测。两组参照均完整提交后，才从 1,000 个 train 场景中按私有排列采集 64 个回合。第 16、32、48、64 回合完成后，对冻结权重再次评价相同的 32 场。命令中断后，在同一机器、同一环境使用原命令续跑。已完成的验证组有结果收据，中断的验证组重新运行。活动训练回合从上一个已提交边界重新开始。

结果在 `pilot_2729/summary.json`。其中 references 给出两个参照，validation_trend 给出四次整组验证，training_intervals 按区间显示更新轮数、角色覆盖与耗时，selected_checkpoint 指向验证平均惩罚完成步数最低的训练边界，同值取较早者。成功时使用实际完成步数，失败记 500。64 回合的健康判断来自持续更新、异常记录、合成后命令的变化以及交接后执行艇实际参与 Actor 更新的样本数。该样本数按抽样使用次数累计，允许同一转移多轮被抽中。可同时查看交接后开启执行门的原始环境步数。逐回合统计进入边界历史；逐步日志含梯度范数、Q、TD 目标与实际各艇更新行数。计时分别覆盖场景复位与运行器构造、真实环境及 Replay 写入、Replay 抽样及更新，不含检查点 I/O。残差幅度和分量饱和率以开门分量为分母，物理变化为同状态同推进记忆下实发命令及推力与零残差参照的 L2 差（m/s²）。数值异常中止该尝试，记录在其 failure.json 中。

若 pilot 的更新与执行阶段覆盖健康，且验证表现仍需改善，可用以下命令手动启动已登记的小范围搜索。该命令需要同一研究目录中的 pilot 完成收据。每个组合从头训练，开发种子 17239、29537，均为最多 200 回合，每 16 回合及第 200 回合统一验证。四组只改变 Actor 学习率与 OU 衰减。按同一检查点两个开发种子的平均验证代价选配置，代价相同按配置表 A、B、C、D 的次序，再取较早回合。详细得分写入 search_selection.json。训练入口只采集 train 和评价 validation。

```bash
python -B -m chapter4_interval_robust.nominal_maddpg.development search --execute --output "$HOME/ch4-nominal-dev"
```
