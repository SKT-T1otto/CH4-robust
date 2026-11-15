# SIRSA依据、继承边界与可检验假设

日期：2026-11-13。以下是待实现、待实验的v2候选设计，不代表已完成复现或证明创新。仅复核所列原文的方法章节，未声称穷尽相关工作。

## 文献定位

SIRSA[1]§3、§5（所存PDF第3—5页）已经包含：多集合鲁棒性、ensemble输出中心/尺度、历史驱动递归集合更新、集合条件Actor、上下文条件标量Q、SAC与集合内均匀采样的低尾Q近似。其§6.3（第7页）还测试回合内每50步切换参数。因此，区间输入、CVaR、SAC、在线适配和分段变化均应明确继承。原文称集合为ℓ1-ball；盒状候选集合属于适配，不能重命名为原方法。

UP-OSI[2]第3—4页已有状态动作短历史辨识及估计错误轨迹再训练；RMA[3]第3—4页已有特权训练、合法历史部署，并讨论物理参数不可辨识。EPOpt[4]第3—4页已有模型采样、最差比例轨迹优化及目标域分布更新。ACDA[5]第4—7页已有时间戳、动作缓存、延迟假设下命令回溯和执行转移重构。故“加历史队列/处理延迟”本身不足以形成强贡献。

TC-RMDP[6]第3—4页说明时序相关扰动和有界漂移的合理性；其收缩保证不能直接移植到隐藏参数与有限窗口。IDTD[7]第2—4页以参数敏感度及Schur补讨论混淆：预测误差低不代表参数真实，也不代表多车历史必然能分辨流、阻力和能力。

建议主贡献为A+B：在共享流场、个体响应与未知执行时序混淆下，构造条件一致集合；C保留SIRSA风险原则并作团队适配。

## 候选模型与集合

主域采用21D约化参数：共享水平流场系数g_h，加四车各5维η_i=(阻力尺度δ_i、能力κ_i、净垂向偏置b_i、迟滞保留系数ρ_i、整数延迟d_i)。限速尺度固定1、动作噪声固定已知，垂向公共系数固定名义值0，不再分别辨识不可分离的公共垂向项与个体偏置。须保持既有名义基流和随机流；此约化不能被宣传成完整水动力模型。

合法命令u为保护/合成后发往执行器的命令；候选滤波状态按现有ρ语义递推：

\[
m^i_t=\rho_i m^i_{t-1}+(1-\rho_i)u^i_{t-d_i},\quad
x^i_{t+1}=F_0(x^i_t,\kappa_i m^i_t,g_h,\delta_i,b_i)+w^i_t.
\]

复位、启动填充与参数变化时记忆是否延续必须固定。33D诊断/训练上下文为21D参数与12D滤波记忆，不能冒充旧输入身份。

对有效、未饱和且无覆盖/碰撞异常的历史，记C_i(g,η_i,m_i;H_i)为残差容差内的候选一致约束。分层维护：

\[
G_t=\{g:\forall i,\exists(\eta_i,m_i)\ C_i(g,\eta_i,m_i;H_i)\},\quad
\Xi_t=\{(g,\eta_{1:4},m_{1:4}):g\in G_t,\ \forall i\ C_i\}.
\]

先候选g，再保留各车条件候选；未知延迟的可行分支取并集，不能错误取交。公共参数通过位置相关流场映射产生不同车流速。有限网格漏解、噪声尾部及变化点都可能造成空集；扩张/重置是工程处理，不是统计覆盖证明。

## 风险查询与历史支持

同一场景仅采一个g，各车从其条件候选采η_i，并携带该分支重建的m_i；不能拼接事实真实滤波状态与另一组参数。令q_t是声明的有限候选采样测度，取N个场景，按团队Q从低到高选k=max(1,floor(βN))，优化低尾均值，加入SAC熵项[1,8]。β与熵温度分开命名，并设最低有效尾部样本量。此项继承原理，非新CVaR定义。

训练Q只能使用事实转移的真实参数/记忆标签；替换参数后仍保留原奖励和后继状态不合法。拟用Q(H_t,Ξ_t,a_t,c,m_t)：Actor所需集合/估计器摘要若不能由有限H恢复，Q必须显式包含，不能仅靠33D使过程“自动Markov化”。Actor与部署估计器不读真实c、m。

候选残差一致仅排除明显矛盾，不能保证Q在反事实(H,Ξ,c,m,a)处有训练支持。须用同协议多场景事实轨迹训练，检查支持距离、Q分歧及预测残差；支持不足时扩大采样/回退，并记录未支持查询。门控仍是经验措施，不能证明无外推偏差、严格CVaR或最坏情形安全。

## H1—H3与否证

- H1：相同经验联合覆盖率下，条件共享集合比独立估计更窄、个体偏差更小；互补激励不足时不应虚假收缩。比较同信息权限、同预算，并报告失覆盖。
- H2：候选延迟/迟滞联合记忆比动作错位更新减少“时序变化被认成流场/能力变化”，改善变化后恢复；单因素、混合因素及无变化对照均需检验。
- H3：A+B所得集合支持的团队低尾策略，相对同架构均值目标及朴素多智能体SIRSA，改善受扰低尾与任务成功；同时报告名义损失、干预、开销及未恢复样本。效果不显著或被保护器主导即不支持假设。

## 原始文献

1. Annie Xie, Shagun Sodhani, Chelsea Finn, Joelle Pineau, Amy Zhang. **Robust Policy Learning over Multiple Uncertainty Sets**. ICML, 2022. https://proceedings.mlr.press/v162/xie22c.html
2. Wenhao Yu, Jie Tan, C. Karen Liu, Greg Turk. **Preparing for the Unknown: Learning a Universal Policy with Online System Identification**. RSS, 2017. https://arxiv.org/abs/1702.02453
3. Ashish Kumar, Zipeng Fu, Deepak Pathak, Jitendra Malik. **RMA: Rapid Motor Adaptation for Legged Robots**. RSS, 2021. https://arxiv.org/abs/2107.04034
4. Aravind Rajeswaran, Sarvjeet Ghotra, Balaraman Ravindran, Sergey Levine. **EPOpt: Learning Robust Neural Network Policies Using Model Ensembles**. ICLR, 2017；预印本2016. https://arxiv.org/abs/1610.01283
5. John Wikman, Alexandre Proutiere, David Broman. **Adaptive Reinforcement Learning for Unobservable Random Delays**. ICML, 2026；预印本2025. https://proceedings.mlr.press/v306/wikman26a.html
6. Adil Zouitine, David Bertoin, Pierre Clavier, Matthieu Geist, Emmanuel Rachelson. **Time-Constrained Robust MDPs**. 2024. https://arxiv.org/abs/2406.08395
7. Sangwoo Shin, Ashvin Anilkumar, Ryan Gao, Josiah P. Hanna. **Informationally Decoupled Trajectory Design for Sim-to-Real System Identification**. 2026预印本. https://arxiv.org/abs/2610.10905
8. Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine. **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**. ICML, 2018. https://arxiv.org/abs/1801.01290
