# 第四章SIRSA A+B详细方案 v2.0

日期：2026-11-13。**本次仅创建研究方案文档；v2算法尚未实施或训练。**旧方案与项目源码不覆盖。

- [最终34页PDF](output/pdf/CH4_SIRSA_AB_详细研究方案_20261113_v2.pdf)
- [可编辑完整正文](CH4_SIRSA_AB_详细方案_v2.md)
- [代码迁移审查](support/parameter_code_review.md)、[文献与机制审查](support/sirsa_foundation_review.md)、[实验设计审查](support/experiment_review.md)

方案将A+B设为核心候选贡献：共享流场条件下的个体可行解释，以及命令延迟/迟滞一致的联合更新。团队低尾目标明确承接SIRSA，不另称新提出CVaR。

关键决策：主域21D参数；165D合法历史输入；705D Critic候选；同一估计器下共享×时序2×2消融；宽度消融仅改Actor，C-off仅改alpha。所有数值均为候选开发设置，正式结果、创新性和原SIRSA复现仍待验收。

全文详细说明了参数不可辨识性、逐文件迁移、新旧schema不兼容、条件联合场景与滤波记忆、事实回放、对照权限、校准和数据划分、恢复竞争事件、覆盖与候选支持的区别及G0-G5验收。

页码导航：参数与可辨识性8-9页；A+B 11-14页；schema与学习15-18页；代码修改19-21页；对照/消融22-24页；辨识/闭环/统计25-27页；验收和机制检查28-29页；运行模板30页。

本次没有执行研发模板、安装原SIRSA依赖、加载历史模型、启动训练或向GitHub写入。现有v1入口与未来v2模板在PDF内分开标记。

生成源：[support/plan_content.py](support/plan_content.py)、[support/build_pdf.py](support/build_pdf.py)。版式、内容与完整性报告见[support/layout_report.json](support/layout_report.json)、[support/delivery_report.json](support/delivery_report.json)、[support/integrity_after.json](support/integrity_after.json)。全部新文件见[support/created_files.json](support/created_files.json)。

分项审查草稿用于追踪讨论；主文统一了三态历史status、尾样本N=32/64/128、联合框覆盖与同g候选支持、整批风险支持门和竞争终点统计。若草稿细节与主文有差异，以v2主文为准。
