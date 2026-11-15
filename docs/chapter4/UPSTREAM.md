# SIRSA 上游身份与本地适配

研究起点为 Xie 等人的 *Robust Policy Learning over Multiple Uncertainty Sets*，ICML 2022。原论文与官方代码来源是 [PMLR 论文页](https://proceedings.mlr.press/v162/xie22c.html) 和 [anxie/sirsa 官方仓库](https://github.com/anxie/sirsa)。本文件的实现核对依据是本地固定版本和 `provenance/ch4/upstream_sirsa.json`，不声称已经运行原论文实验。

## 已固定并下载的参考

| 项目 | 固定值 |
| --- | --- |
| commit | `ed550ca2611468bb45154b735ee89ecd611d23a2` |
| 下载 URL | `https://codeload.github.com/anxie/sirsa/zip/ed550ca2611468bb45154b735ee89ecd611d23a2` |
| ZIP SHA-256 | `2ba550370e4fd2e1db4c910078aa55d18af4c92807234e6c5f99ac3139b7a038` |
| 本地参考 | `work/ch4/upstream_sirsa_pinned_v2/sirsa-ed550ca2611468bb45154b735ee89ecd611d23a2/` |
| 下载收据 | `work/ch4/upstream_sirsa_pinned_v2/source_receipt.json` |
| 代码级来源映射 | `provenance/ch4/upstream_sirsa.json`，含已审查文件的 blob 身份 |
| 原工作复现 | `original_reproduction_passed=false` |
| 原栈安装/训练 | 本次没有安装，没有启动 |

下载目录仅作为只读研究参考；不修改、不 import、不把原 TensorFlow 代码复制进新运行链。官方代码的 TensorFlow/softlearning/MuJoCo 依赖不加入第三章环境，也不由 CH4 verify 或 train 自动安装。

`setup.py` 声明 MIT，但固定树中没有单独的 LICENSE 文件。当前本地研究引用不代表再分发许可已审查；打包或向外再分发上游源码前仍为 `REVIEW_REQUIRED`，不得补造 LICENSE 后宣称已解决。

## 方法思想与实际差异

| 方面 | 上游/研究起点 | 当前 AUV 实现 |
| --- | --- | --- |
| 学习后端 | TensorFlow/softlearning，单智能体 SAC | 新 PyTorch 四 Actor、集中式联合动作双 Critic |
| 集合表示 | 原论文使用 L1 球术语与中心/尺度条件；固定官方 SAC 代码有逐坐标均匀中心±尺度采样 | 显式每车九维盒状外包络，两个共享场参数及四组个体参数 |
| 参数监督 | 历史条件残差 context ensemble、真实参数监督 | 独立实现 role/prior/history 条件监督 ensemble；均值±标准差启发式范围 |
| 先验递推 | 上游具有递归先验路径 | 当前每次从完整公开先验估计窗口，避免无支持递归缩窄；明确适配差异 |
| 新集合估计 | 上游 ensemble 作为参照 | full 用有限粒子共享/个体检验、整数延迟与迟滞记忆、经验残差与重置 |
| 风险效用 | 参数场景采样的 SAC 尾部目标思想 | 同一物理世界联合采样，有经验参数邻域支持门槛，不足回事实效用 |
| 动作 | 原任务连续随机策略 | 参考平移高斯+tanh 的3D绝对归一化命令，单次物理能力映射 |
| 任务 | 上游连续控制设置 | 当前四车搜索、发现、交接、执行任务与原团队奖励 |
| 对照身份 | 原 SIRSA | `sirsa_naive` 仅为新朴素多智能体适配；不能用其检查结果替代原复现 |

参考原论文不意味着当前盒状集合享有其理论结论；官方代码的抽样细节也不能用来把论文的集合术语偷偷改写成相同定义。A/B/C 的经验收益与原创性仍需要独立对照证据。

## 重新准备参考与后续复现

参考已经下载，不需要为了运行 AUV 再下载。若需在另一副本准备固定上游，先看计划，再明确执行，始终使用新目录：

```text
python -B ch4_cli.py upstream-reference --output work/ch4/manual_upstream_001
python -B ch4_cli.py upstream-reference --execute --output work/ch4/manual_upstream_001
```

命令只下载固定 ZIP、检查解包路径并写下载收据/参考说明，不启动训练，不安装依赖，不修改 GitHub。新下载后应核对上述已知 SHA-256，再用于后续阅读。

真正的原工作复现应在独立环境中另行审查上游依赖、任务许可、固定种子与预算，遵循该版本自己的 parser/variants；不将随手缩短的 AUV 检查称为原方法成功。已生成的 `REFERENCE_COMMANDS.txt` 记录参考环境准备方式，它不是本次已执行记录。原轻量设置中的 Reacher2D、history_length=5、latent_dim=2、ensemble_size=4、cvar_alpha=0.25、cvar_samples=50 仅为上游核对起点，未宣称在本平台安装可用或获得匹配成绩。

原方法复现、AUV 朴素适配与 A/B/C 改进应保持三个独立实验身份；不继承旧单智能体成绩，不重标历史检查点，不把本次来源下载收据当作性能收据。
