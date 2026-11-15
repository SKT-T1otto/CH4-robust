"""Verify already completed artifacts and audit the resumed task; no training."""
import hashlib,importlib.util,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def new(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)
def main():
    spec=importlib.util.spec_from_file_location('task_audit',HERE/'audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    baseline=read(HERE/'resume_before_20261115.json');current=audit.protected()
    changes={k:[n for n in sorted(v.keys()|current[k].keys()) if v.get(n)!=current[k].get(n)] for k,v in baseline['protected'].items()}
    new(HERE/'resume_after_20261115.json',{'protected':current,'changes':changes,'client_date':'2026-11-15'})
    if any(changes.values()):raise AssertionError('Protected change observed; inspect without rolling back: '+repr(changes))
    results={}
    for key,name in [('N0','ab_v2_final_n0_20261115_002'),('N1','ab_v2_final_n1_20261115_002'),('v1_regression','ab_v2_v1_regression_20261115_002')]:
        folder=ROOT/'work/ch4'/name;r=read(folder/'validation.json')
        assert r['passed'] and r['failures']==r['errors']==r['skipped']==0
        for leaf,sha in r['artifacts'].items():assert hashlib.sha256((folder/leaf).read_bytes()).hexdigest()==sha
        ledger=read(folder/'mechanism_ledger.json')
        if key!='v1_regression':
            assert all(m['updates']<=4 and m['physical_steps']<=256 and m['episodes']<=8 for m in ledger['models'].values())
        results[key]={'path':str((folder/'validation.json').relative_to(ROOT)),'tests':r['tests'],'source':r['source'],'passed':True}
    assert len({r['source']['sha256'] for r in results.values()})==1
    pipeline=read(HERE/'pipeline_20261115_002.json');assert pipeline['passed'] and pipeline['physical_and_optimizer_resume_equal']
    assert pipeline['source']==results['N0']['source'] and all(s['exit_code']==0 for s in pipeline['steps'])
    # Produce the final human-readable report before hashing the whole delivery.
    train=pipeline['train'];evaluation=pipeline['evaluation'];resume=pipeline['resume']
    lines=['# SIRSA A+B v2 实施与受限验证报告','',
        '日期：2026-11-15。源码增量版本v9；原SIRSA复现、正式性能与论文创新性均未验收。','',
        '## 完成内容','',
        '已实现21D参数/33D执行器上下文、165D公开历史输入、705D团队双Q；条件分层集合A、候选延迟/迟滞时序B、冻结联合场景银行、SAC、监督ensemble参照、匹配的共享×时序四格、Width/C/Memory消融，以及独立运行与恢复入口。','',
        '原9D物理仅作显式适配；固定不可辨识垂向分量、限速及已知噪声。动作、任务奖励、目标/障碍过程、交接与成功标准保持父合同。','',
        '## 实际检查','',
        '| 检查 | 结果 | 证据 |','|---|---|---|']
    for name,result in results.items():
        target=Path(result['path']).as_posix()
        lines.append(f"| {name} | {result['tests']}项通过，0失败/错误/跳过 | [{target}](../../../{target}) |")
    lines+=['',
        '环境：Windows、Python 3.10.20、NumPy 2.2.6、PyTorch 2.11.0+cu126；显式禁用CUDA，PyTorch单线程。没有启动第三章任务或使用训练GPU。','',
        '机制覆盖：名义成对转移/随机流、D2零残差参考等价、原生动力学与N1噪声、延迟/迟滞/复位、结构不可辨识等价对、空集/缺数据/历史三态、同一公共g及个体响应、Actor类型和真值隔离、SAC终止/截断与Polyak、分析型Q的低尾/整行回退、监督参照更新、仅Actor宽度消融、完整检查点恢复及v1拒绝。','',
        '## 端到端机制练习','',
        f"- 新临时模型训练：{train['total_physical_steps']}物理步，{train['total_updates']}次完整更新，2回合×16步。",
        f"- 冻结权重dev评估：{evaluation['physical_steps_executed_this_run']}步，{evaluation['total_updates']}次更新；另有同一权重的名义配对评估32步。",
        f"- 回合边界恢复：新执行{resume['physical_steps_executed_this_run']}步、{resume['updates_executed_this_run']}次更新；最终网络、target、优化器、温度、回放、支持库、随机流和第二回合物理轨迹与连续运行逐项相等。",
        '- 同轨迹辨识：2回合×16步、四格共128条记录；四组总物理预测调用数一致，补齐/有效历史调用分列。没有进行四格策略训练矩阵。',
        '- 外部汇总完成；名义差值仅按同权重/训练种子/场景/域配对，2个dev场景匹配。','',
        '[端到端机器记录](pipeline_20261115_002.json)；各命令stdout/stderr与新work目录均保留。只加载本轮临时模型自己的检查点，没有搜索或加载历史模型。','',
        '## 结果的实际边界','',
        f"短训练成功率{train['success_rate']:.3f}，碰撞率{train['collision_rate']:.3f}，联合框覆盖{train['joint_box_coverage']:.4f}，集合有效率{train['valid_rate']:.4f}；风险支持行数{train['risk_supported_rows']}，回退行数{train['risk_fallback_rows']}。",
        f"短dev评估成功率{evaluation['success_rate']:.3f}，联合框覆盖{evaluation['joint_box_coverage']:.4f}，集合有效率{evaluation['valid_rate']:.4f}。",
        '这些仅是短管线诊断；没有证明名义任务能力、A+B收益、风险目标有效训练、收敛或相对基线优势。分析型Q检查是构造用例，不是任务性能。经验集合和Q支持门不构成统计置信或严格最坏情形保证。','',
        '## 开发失败与修正','',
        '- v5首次检查暴露条件池与原生物理核的轴顺序问题，修正后v6通过；旧失败日志保留。',
        '- v7补了历史年龄、空集复位窗口和固定先验有效性；v8增加集合摘要支持、协议汇总合同与严格边界检查。',
        '- v8端到端CLI命令均成功且恢复已相等，但同轨迹预算断言失败：不同重置历史长度导致累计预测不同。保留[pipeline_20261115.json](pipeline_20261115.json)及其原始结果；v9增加掩码补齐调用并单列有效/补齐成本。没有关闭断言、删除失败或修改旧结果。','',
        '## 待手动研究验收','',
        '开发/校准场景的残差与风险支持阈值、候选预算、名义最低能力、多训练种子与冻结测试；原SIRSA独立复现、ACDA/传统观测器强对照、Linux原生准入及真实AUV验证仍待完成。有限保护当前只允许共同关闭；不宣称安全证书。','',
        '继承的path-option provenance与pure-MARL manifest问题继续为失败，旧清单与_snapshot未重签。完整性和文件清单见[after.json](after.json)、[delivery_20261115.json](delivery_20261115.json)。','',
        '没有Git初始化、add、commit、push、remote修改或GitHub写入；没有长训练或正式实验矩阵。所有本次终端任务结束后交付。']
    (HERE/'VALIDATION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    previous_argv=sys.argv
    try:
        sys.argv=[str(HERE/'audit.py'),'after'];audit.main()
    finally:sys.argv=previous_argv
    full=read(HERE/'after.json')
    assert not full['deleted'] and all(not v for v in full['protected_changes'].values())
    # The initial audit helper was created before its own before-inventory.
    created=sorted(set(full['created'])|{'docs/chapter4/implementation_ab_v2_20261113/audit.py',
                   'docs/chapter4/implementation_ab_v2_20261113/delivery_20261115.json'})
    report={'client_date':'2026-11-15','passed':True,'source':results['N0']['source'],
        'mechanism_checks':results,'pipeline_report':str((HERE/'pipeline_20261115_002.json').relative_to(ROOT)),
        'checkpoint_restore_exact':True,'factorial_same_trajectory_budget_equal':True,
        'legacy_protected_counts':{k:len(v) for k,v in current.items()},'protected_unchanged_entire_task':True,
        'protected_unchanged_resumed_turn':True,'modified':full['modified'],'created':created,'deleted':[],
        'bootstrap_created':['docs/chapter4/implementation_ab_v2_20261113/audit.py'],
        'historical_models_loaded':False,'formal_training_started':False,'github_writes':False,
        'original_sirsa_reproduced':False,'performance_passed':False,
        'remaining':['Calibrate residual/support thresholds and candidate budgets on separate calibration data.',
                     'Establish nominal task ability before formal multiseed robustness comparisons.',
                     'Original SIRSA reproduction, ACDA/observer strong comparisons, native Linux and real-vehicle evidence remain unaccepted.',
                     'Inherited parent provenance checks remain failed; no old manifests were re-signed.'],
        'file_inventory_note':'after.json records file hashes except its own self-hash; this delivery file is listed separately.'}
    new(HERE/'delivery_20261115.json',report)
    print(json.dumps({k:report[k] for k in ('passed','source','legacy_protected_counts','protected_unchanged_entire_task','checkpoint_restore_exact','github_writes','performance_passed')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
