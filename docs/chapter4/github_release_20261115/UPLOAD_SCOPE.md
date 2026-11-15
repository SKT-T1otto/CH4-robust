# CH4-robust首次上传范围

用户于2026-11-15明确授权将当前第四章完整代码及强化学习底座上传至：

https://github.com/SKT-T1otto/CH4-robust.git

此前不允许Git初始化/提交/上传的会话限制，由本次明确指定的新仓库上传授权在此范围内取代。仅本第四章目录初始化独立`.git`，不使用第三章或父目录的仓库，不向其他远程写入。

## 包含

- 完整`core/`环境、动力学、任务/地图/规划与公共接口。
- 完整`chapter3_bser/`，作为此第四章副本自带的强化学习和任务运行底座；不访问第三章原目录作为运行依赖。
- 完整`chapter4_interval_robust/`，包含v1与独立SIRSA A+B v2及其入口。
- `configs/`、`checks/`、`tools/`、`scripts/`和依赖清单。
- `provenance/`、`_snapshot/`、`verification/`及验证器引用的来源/证据归档。大于20MB的两个归档分别约22MB和43MB，属于本地验证所需资产，原字节保留。
- 项目文档、研究方案、旧CH4源参考与已有上游许可证说明。现有`.gitattributes`保留精确文件字节，避免CRLF转换改变身份。

## 本地保留，不上传

- `work/`、`outputs/`、缓存、虚拟环境、IDE配置和临时检查点/权重。
- `work/ch4/upstream_sirsa_pinned_v2/`里的官方SIRSA下载归档和代码；其再分发许可尚未审查，不捆绑发布。
- 第三方参考论文`references/research/1-223081.pdf`全文；引用和作者自有研究方案保留。
- 凭据、私钥和`.env`等身份资料。没有读取或打印Git凭据。

Git候选文件及嵌套来源ZIP已做高置信凭据模式与模型文件检查：见`preflight_scan.json`。这是发布范围检查，不等同完整安全审计。

## 当前科研状态

源码身份仍是CH4 source v9，未因发布更改算法或重签父版本。N0/N1机制与v1回归记录保持；正式性能、原SIRSA复现、Linux/真实AUV验证尚未验收。继承的两项父来源校验失败仍保留。

根README改为第四章入口，继承的原README逐字节保存在`README_INHERITED_ACTIVE.md`。`.gitignore`只控制版本追踪，没有删除本地资产。

本次没有运行训练或加载检查点。上传前的受保护文件哈希对比见`before.json`与`after.json`；提交文件范围和哈希见`upload_manifest.json`。
