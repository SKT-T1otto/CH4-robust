"""Finalize CH4 evidence only; never edits algorithm inputs or protected trees."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from chapter4_interval_robust.dev_audit import AUDIT,files,protected,write_new
from chapter4_interval_robust.provenance import identity,sha
from chapter4_interval_robust import checkpoint


def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))


def equal(a,b,path='state'):
    if torch.is_tensor(a):
        torch.testing.assert_close(a,b,rtol=0,atol=0,msg=path)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert type(a)==type(b) and len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    else:assert a==b,(path,a,b)


def main():
    source=identity();validation=read('work/ch4/validation_windows_v4/validation.json')
    assert validation['passed'] and validation['source']==source and validation['tests']==22
    for name,h in validation['artifacts'].items():assert sha(ROOT/'work/ch4/validation_windows_v4'/name)==h
    summaries={name:read('work/ch4/'+folder+'/summary.json') for name,folder in
               [('train','end_to_end_full_v4'),('evaluate','end_to_end_eval_v4'),
                ('resume','end_to_end_resume_v4'),('heldout','end_to_end_heldout_v4'),
                ('nominal','end_to_end_nominal_eval_v4')]}
    assert all(r['source']==source for r in summaries.values())
    original=checkpoint.load(ROOT/'work/ch4/end_to_end_full_v4/checkpoint_00002.pt')
    resumed=checkpoint.load(ROOT/'work/ch4/end_to_end_resume_v4/checkpoint_00002.pt')
    equal(original['learner'],resumed['learner'])
    equal(original['numpy_global_rng'],resumed['numpy_global_rng'])
    equal(original['python_global_rng'],resumed['python_global_rng'])
    def trace(folder):
        return [json.loads(s) for s in (ROOT/'work/ch4'/folder/'transitions.jsonl').read_text(encoding='utf-8').splitlines()]
    equal([r for r in trace('end_to_end_full_v4') if r['episode']==1],trace('end_to_end_resume_v4'),'resumed_physical_trace')
    assert (ROOT/'work/ch4/end_to_end_eval_v4/updates.jsonl').stat().st_size==0
    assert summaries['train']['total_updates']==4 and summaries['train']['total_physical_steps']==32
    assert summaries['resume']['physical_steps_executed_this_run']==16 and summaries['resume']['updates_executed_this_run']==1
    regrouped=read('work/ch4/resume_summary_v4/summary.json')['full:train:episode:in_domain']
    assert regrouped['episodes']==2 and regrouped['mean_return']==summaries['resume']['mean_return']
    # Exercise the actual CLI rejection path without constructing an environment.
    command=[sys.executable,'-B','ch4_cli.py','evaluate','--mechanism','--seed','7292',
             '--weights','work/ch4/end_to_end_full_v4/checkpoint_00002.pt',
             '--output','work/ch4/expected_seed_rejection_v4','--execute']
    rejected=subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
    assert rejected.returncode!=0 and 'evaluation changed frozen contract: seed' in rejected.stderr
    pipeline=dict(source=source,client_date='2026-11-13',resumed_model_optimizer_replay_rng_identical=True,
                  resumed_physical_trace_identical=True,evaluation_updates=0,seed_change_rejected=True,
                  rejection_error=rejected.stderr.splitlines()[-1],
                  resumed_summary_matches_external_summary=True,
                  runs={k:{n:r[n] for n in ('episodes','total_physical_steps','physical_steps_executed_this_run',
                       'total_updates','updates_executed_this_run','risk_supported_updates','success_rate',
                       'joint_coverage','interval_valid_rate','recovery_event_count','recovery_unrecovered_rate')}
                        for k,r in summaries.items()},
                  only_new_mechanism_checkpoints_loaded=True,long_training_started=False,git_writes=False)
    write_new(AUDIT/'pipeline_checks.json',pipeline)
    inherited={}
    for name,code in {
        'path_option_identity':'from chapter3_bser.path_option.source import identity; print(identity())',
        'independent_source_verifier':'from pathlib import Path; from tools.verify_source import verify; print(verify(Path.cwd()))',
    }.items():
        probe=subprocess.run([sys.executable,'-B','-c',code],cwd=ROOT,capture_output=True,text=True)
        inherited[name]=dict(exit_code=probe.returncode,error=probe.stderr.splitlines()[-1] if probe.stderr else None)
    write_new(AUDIT/'inherited_source_checks.json',inherited)
    print('Verifying complete protected legacy trees and Active source inventory...',flush=True)
    before=read('docs/chapter4/development_audit/before.json');current=protected()
    changes={k:sorted(n for n in current[k].keys()|v.keys() if current[k].get(n)!=v.get(n)) for k,v in before['protected'].items()}
    legacy_ok=all(not changes[k] for k in ('CH3','CH4','CH5'))
    protected_text='全部受保护文件未变化。' if all(not v for v in changes.values()) else '存在外部变化，逐项见 after.json；本任务所有写入均限定在第四章副本。'
    report=f'''# 第四章实现与验证交付记录

日期：2026-11-13。源码身份：`{source['sha256']}`，独立来源版本 v{source['version']}。

已实现独立多智能体 SAC、共享/个体经验集合、时序一致辨识、受支持的集合尾部效用、匹配消融、名义/分段/漂移扰动、留出域、训练评估与回合边界恢复入口。详细模块、信息权限、复用范围和命令分别见 README、DESIGN、INFORMATION、LIMITS、UPSTREAM。

## 本次实际证据

| 检查 | 结果与范围 |
| --- | --- |
| Windows 原生机制套件 | 22 项通过，0 失败/错误/跳过；Python 3.10.20、NumPy 2.2.6、PyTorch 2.11.0+cu126，CPU 单线程 |
| 新增扰动关闭 | 对照状态、奖励、终止及 Python/NumPy/Torch 随机流完全一致 |
| 原 D2 零残差参考 | 绝对命令转换后在 2e-6 声明容差内一致；未读取旧模型 |
| 因素与时序 | 实际 core 验证流场、垂向、阻力、浮力、能力、限速、延迟、迟滞、命令噪声；验证复位和正确延迟假设 |
| SAC/风险/信息 | 检查终止和截断 bootstrap、Polyak 更新、尾部方向和梯度、经验支持门、Actor 类型边界、未训练估计回退 |
| 完整短训练 | 2 回合 × 16 物理步，共 4 次完整更新，日志与回合边界检查点保存成功 |
| 冻结权重评估 | dev 分段扰动 2 回合 × 16 步，学习更新为 0 |
| 回合边界恢复 | 承接 16 步，新执行 16 步、1 次更新；最终模型、优化器、回放、随机流和第二回合物理轨迹与连续运行完全一致 |
| 留出域与名义对照 | 各 1 回合 × 16 步；留出执行器延迟为 3、公开先验最大为 2，路径可执行且失配按原样报告 |
| 评估/汇总合同 | 变更训练 seed 被实际 CLI 拒绝；恢复前记录未丢失，外部汇总与恢复运行一致 |
| 受保护目录 | CH3/CH4/CH5 共 790 文件逐文件 SHA256 比对；Active 为排除运行输出与缓存的源码清单。{protected_text} |

证据：`work/ch4/validation_windows_v4/{{validation.json,tests.log,mechanism_ledger.json}}`；
`work/ch4/end_to_end_{{full,eval,resume,heldout}}_v4/` 与 `end_to_end_nominal_eval_v4/`；
`docs/chapter4/development_audit/pipeline_checks.json`、`after.json`。

## 不可据此声称通过的事项

这些是机制和管线检查。短训练任务成功率为 0，联合覆盖为 {summaries['train']['joint_coverage']:.5f}，风险支持更新为 0；短 dev 评估的集合有效率仅 {summaries['evaluate']['interval_valid_rate']:.5f}。
覆盖、有效率、任务能力和尾部学习均未达到正式结论所需证据。高覆盖也可能来自全先验回退。
没有执行完整研究训练或最终测试；正式预算、校准与最低名义成功门槛为 REVIEW_REQUIRED。

| PDF 阶段 | 当前身份 |
| --- | --- |
| G0 合同与事实 | 已形成可审阅实现与信息/动作/参数合同；研究阈值仍待校准 |
| G1 接入一致性 | Windows 有限原生验证通过；Linux 待重新验证 |
| G2 原 SIRSA 复现 | 固定上游提交及归档已保存；原 TF/MuJoCo 依赖未安装，原结果未复现 |
| G3 朴素多智能体适配 | 代码与入口已实现；名义任务能力和长程训练待验证 |
| G4 A/B/C 必要性 | 模块、候选消融和检查已实现；独立重训、校准与收益归因待验证 |
| G5 正式结果 | 未开始，不能宣称鲁棒性能提升、创新成立或全局安全保证 |

继承的两个原来源失败仍在：`chapter3_bser/learning/provenance.py` 与 path-option 清单不一致，以及 pure-MARL source inventory mismatch。
原清单和 `_snapshot` 保留；新增四版来源演化未将这些失败重签为通过。初次本任务测试中的 float32 预期表示错误已修正，早期失败日志保留。

没有 Git 初始化、add、commit、push、remote 修改或 GitHub 写入。唯一外部代码操作是只读获取固定 SIRSA 参考；没有导入它执行 AUV 方法。
文件新增/修改/删除清单及最终哈希见 `development_audit/after.json`；该清单记录自身路径，不递归计算自身哈希。
'''
    write_path=ROOT/'docs/chapter4/VALIDATION.md'
    with write_path.open('x',encoding='utf-8',newline='\n') as f:f.write(report)
    project=files(ROOT)
    final=dict(client_date='2026-11-13',source=source,protected_changes=changes,
               protected_counts={k:len(v) for k,v in current.items()},legacy_unchanged=legacy_ok,
               active_source_unchanged=not changes['AUV-Thesis-Active'],
               modified=sorted(n for n,h in before['project'].items() if project.get(n)!=h),
               deleted=sorted(n for n in before['project'] if n not in project),
               created=sorted(set(n for n in project if n not in before['project'])|
                              {'chapter4_interval_robust/dev_audit.py','docs/chapter4/development_audit/after.json'}),
               hashes=project,self_manifest_hash=None,formal_training_runs=False,mechanism_training_runs=True,
               github_writes=False,source_verification_passed=True,checks_passed=22,
               note='Read-only helpers reused from dev_audit; this finalizer distinguishes bounded mechanism training from formal training.')
    write_new(AUDIT/'after.json',final)
    assert not final['deleted'] and legacy_ok
    assert identity()==source
    actual=files(ROOT)
    assert set(actual)==set(project)|{'docs/chapter4/development_audit/after.json'}
    assert all(actual[k]==h for k,h in project.items())
    print(json.dumps(dict(source=source,checks_passed=22,legacy_unchanged=legacy_ok,
                         active_source_unchanged=final['active_source_unchanged'],protected_changes=changes,
                         modified=final['modified'],created_files=len(final['created']),deleted=final['deleted'],
                         resumed_states_and_trace_identical=True,github_writes=False),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
