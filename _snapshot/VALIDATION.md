# CH4 开发快照核验报告

生成时间：2026-11-12T22:23:52.407873+08:00

快照完整性结果：**PASS**。

原目录：`E:\gym\code\WORKSPACE\AUV-Thesis-Active`；新目录：`E:\gym\code\WORKSPACE\AUV-Thesis-CH4-IntervalRobust`。
原分支：`main`；HEAD：`18ea616ae256d2bc93f1bd640fe6bd46b5af8180`。

保留 1355 个 Active 文件（含未提交/未跟踪的源码与配置），
另有 57 个旧 CH4 参考源文件、5 个研究参考文件；
合计复制 1417 个文件、167,812,790 字节。

逐文件源/副本 SHA256 一致；独立文件检查失败数 0；
源文件捕获前后变化数 0；原 Git 状态、HEAD、工作树及暂存补丁未变化：True。
这是复制期间的稳定性核验，不是文件系统原子快照，也不锁定用户之后的修改。

已对 308 个 Python 源文件进行无导入、无 pyc 生成的编译检查；错误数 0。
两个当前清单的文件依赖完整性已检查；未更改任何原始来源清单。

- `root_is_local`: 原目录退出码 0，副本退出码 0，迁移等价 True。
- `path_option_identity`: 原目录退出码 1，副本退出码 1，迁移等价 True。
- `independent_source_verifier`: 原目录退出码 1，副本退出码 1，迁移等价 True。

两项现有来源验证器在原目录与副本中均失败，错误一致：

- `path_option_identity`：`chapter3_bser/learning/provenance.py` 与已审查 path-option 清单不一致。
- `independent_source_verifier`：`pure-MARL source manifest/inventory mismatch`。

本次保留上述原有状态，未修改实现或重新签署清单。详细错误见 JSON。
复制完整性通过不代表已有研究代码的全部来源验证、原生后端验收或性能验收通过。

CH3/CH4/CH5 的全部 790 个文件在复制前后逐文件计算 SHA256；差异 0。
检查包括历史模型、实验结果、缓存和 Git 元数据，详见 legacy_before/after 与 legacy_integrity_report.json。

没有运行训练、评估 rollout、环境初始化、依赖安装、检查点加载、Git 提交或推送。
没有实现第四章算法，也没有改变奖励、观测、动作、任务成功、目标运动、扰动或通信语义。

未复制 `.git/`、缓存、`work/`、`3090结果/`。所有未复制文件仍保留在原目录。
未审阅的运行产物标为 REVIEW_REQUIRED；路径、大小及理由见 excluded_files.json。
本副本是开发源码与证据快照，不是所有原始运行结果、历史和检查点的灾备副本。

所有创建文件列于 created_files.json；该清单记录自身路径但不计算自身递归哈希。
