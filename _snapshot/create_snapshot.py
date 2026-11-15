"""One-shot, copy-only CH4 snapshot. Does not train, install, commit or edit inputs."""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


TARGET = Path(__file__).resolve().parents[1]
WORKSPACE = TARGET.parent
SOURCE = WORKSPACE / 'AUV-Thesis-Active'
META = TARGET / '_snapshot'
INCLUDED_ROOTS = {
    'core', 'chapter3_bser', 'configs', 'checks', 'tools', 'scripts',
    'docs', 'provenance', 'verification',
}
CACHES = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}


def now():
    return datetime.now().astimezone().isoformat()


def sha(path):
    h = hashlib.sha256()
    with open(longpath(path), 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def longpath(path):
    value = str(Path(path).absolute())
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        return '\\\\?\\UNC\\' + value[2:] if value.startswith('\\\\') else '\\\\?\\' + value
    return value


def stat(path):
    return os.stat(longpath(path))


def copy_once(path, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if sha(path) != sha(dest):
            raise RuntimeError('Partial snapshot differs from input; review required: ' + str(dest))
        return
    shutil.copy2(longpath(path), longpath(dest))


def write_once(path, value):
    if path.exists():
        if path.read_bytes() != value:
            raise RuntimeError('Snapshot capture changed on resumption: ' + str(path))
        return
    path.write_bytes(value)


def write_json(name, value):
    with (META / name).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def git(*args):
    env = dict(os.environ, GIT_OPTIONAL_LOCKS='0')
    return subprocess.run(
        ['git', '-c', 'core.fsmonitor=false', '-c', 'core.quotepath=false',
         '-C', str(SOURCE), *args], check=True, capture_output=True, env=env,
    ).stdout


def inventory(root, skip_git=False):
    result = {}
    for base, dirs, files in os.walk(root, followlinks=False):
        dirs.sort()
        if skip_git:
            dirs[:] = [d for d in dirs if d != '.git']
        for name in dirs + files:
            if (Path(base) / name).is_symlink():
                raise RuntimeError('Symlink requires review: ' + str(Path(base) / name))
        for name in sorted(files):
            path = Path(base) / name
            result[path.relative_to(root).as_posix()] = path
    return result


def selected(name):
    parts = Path(name).parts
    return not (set(parts) & CACHES) and (len(parts) == 1 or parts[0] in INCLUDED_ROOTS)


def legacy_inventory():
    result = {}
    for folder in ('CH3', 'CH4', 'CH5'):
        for rel, path in inventory(WORKSPACE / folder).items():
            result[folder + '/' + rel] = {'bytes': stat(path).st_size, 'sha256': sha(path)}
    return result


def run_probe(root, code):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PYTHONUTF8='1')
    # This task only invokes the standard-library source verifiers, never a trainer/environment.
    env.pop('PYTHONPATH', None)
    p = subprocess.run([sys.executable, '-B', '-c', code], cwd=root, env=env,
                       capture_output=True, text=True, encoding='utf-8', timeout=60)
    result = {'exit_code': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}
    if p.returncode == 0:
        try:
            result['result'] = json.loads(p.stdout)
        except json.JSONDecodeError:
            pass
    return result


def main():
    started = now()
    assert TARGET.name == 'AUV-Thesis-CH4-IntervalRobust'
    assert SOURCE.is_dir() and not (WORKSPACE / 'AUV-Thesis-activate').exists()
    assert TARGET.resolve().parent == WORKSPACE.resolve()
    existing = inventory(TARGET)
    # Resumption only accepts this script's partial first capture. No copied file is overwritten.
    resuming = (META / 'legacy_before.json').is_file()
    if set(existing) != {'_snapshot/create_snapshot.py'} and not resuming:
        raise RuntimeError('Refusing to overwrite or reuse an existing snapshot')
    if resuming and (META / 'validation_report.json').exists():
        raise RuntimeError('A completed snapshot cannot be rerun or overwritten')

    source_git = {
        'root': git('rev-parse', '--show-toplevel').decode().strip(),
        'branch': git('branch', '--show-current').decode().strip(),
        'head': git('rev-parse', 'HEAD').decode().strip(),
        'latest_commit': git('log', '-1', '--format=%H%n%ad%n%s', '--date=iso-strict').decode('utf-8'),
        'status': git('status', '--short', '--untracked-files=all').decode('utf-8'),
    }
    diff_before = git('diff', '--binary', '--no-ext-diff')
    staged_before = git('diff', '--cached', '--binary', '--no-ext-diff')
    if resuming:
        print('Resuming partial capture; preserving original before-hash record...', flush=True)
        before = json.loads((META / 'legacy_before.json').read_text(encoding='utf-8'))['files']
    else:
        print('Hashing complete legacy trees before copying...', flush=True)
        before = legacy_inventory()
        write_json('legacy_before.json', {'time': now(), 'files': before})

    all_source = inventory(SOURCE, skip_git=True)
    copied_source = {name: path for name, path in all_source.items() if selected(name)}
    print(f'Capturing {len(copied_source)} active project files...', flush=True)
    source_before = {name: {'bytes': stat(path).st_size, 'sha256': sha(path)}
                     for name, path in copied_source.items()}
    if (META / 'source_git.json').exists():
        if json.loads((META / 'source_git.json').read_text(encoding='utf-8')) != source_git:
            raise RuntimeError('Source Git capture changed while resuming')
    else:
        write_json('source_git.json', source_git)
    write_json('source_files_before.json', {'time': now(), 'files': source_before})
    write_once(META / 'source_worktree.patch', diff_before)
    write_once(META / 'source_index.patch', staged_before)
    write_once(META / 'source_git_status.txt', source_git['status'].replace('\n', os.linesep).encode('utf-8'))

    rows = []
    for name, path in copied_source.items():
        dest = TARGET / name
        copy_once(path, dest)
        rows.append({'source': str(path), 'destination': name, **source_before[name],
                     'role': 'ACTIVE_WORKTREE_SNAPSHOT'})

    # A separate, visibly labelled reference tree; never wired into runtime imports.
    legacy_count = 0
    for name, path in inventory(WORKSPACE / 'CH4', skip_git=True).items():
        if Path(name).parts[0] == 'data' or set(Path(name).parts) & CACHES:
            continue
        if path.suffix not in ('.py', '.md', '.bat', '.patch') and path.name != '.gitignore':
            continue
        relative = 'references/legacy_ch4_source/' + name
        dest = TARGET / relative
        copy_once(path, dest)
        rows.append({'source': str(path), 'destination': relative,
                     'bytes': stat(path).st_size, 'sha256': sha(path),
                     'role': 'LEGACY_REFERENCE_ONLY_NOT_RUNTIME'})
        legacy_count += 1

    research = [
        ('docs/1-223081.pdf', 'references/research/1-223081.pdf'),
        ('docs/CH4_IntervalRobust_SIRSA_研究方案_20261112_v1.pdf',
         'references/research/CH4_IntervalRobust_SIRSA_研究方案_20261112_v1.pdf'),
        ('docs/CH4_IntervalRobust_SIRSA_20261112_v1_support/extracted_text.txt',
         'references/research/plan_support/extracted_text.txt'),
        ('docs/CH4_IntervalRobust_SIRSA_20261112_v1_support/source_context.json',
         'references/research/plan_support/source_context.json'),
        ('docs/CH4_IntervalRobust_SIRSA_20261112_v1_support/test_report.json',
         'references/research/plan_support/test_report.json'),
    ]
    for origin, relative in research:
        path, dest = WORKSPACE / origin, TARGET / relative
        copy_once(path, dest)
        rows.append({'source': str(path), 'destination': relative,
                     'bytes': stat(path).st_size, 'sha256': sha(path),
                     'role': 'RESEARCH_REFERENCE_NOT_ACCEPTANCE'})

    exclusions = []
    for name, path in all_source.items():
        if name in copied_source:
            continue
        cache = bool(set(Path(name).parts) & CACHES)
        try:
            excluded_bytes = stat(path).st_size
            missing = False
        except FileNotFoundError:
            excluded_bytes, missing = None, True
        exclusions.append({
            'source': str(path), 'relative_path': name, 'bytes': excluded_bytes,
            'classification': 'REGENERABLE_CACHE' if cache and not missing else 'REVIEW_REQUIRED',
            'reason': ('Interpreter/test cache, omitted from development snapshot' if cache else
                       'Retained at source; raw results/work/staging are not required for source snapshot. '
                       'Review provenance and use-case before future reuse; no deletion authorized.'),
            'retained_original': not missing,
            'missing_at_inventory_stat': missing,
            'original_was_not_deleted_or_changed_by_snapshot': True,
        })
    exclusions.append({'source': str(SOURCE / '.git'), 'relative_path': '.git/',
                       'classification': 'REPOSITORY_METADATA',
                       'reason': 'No linked worktree, history, remote or source index copied; HEAD/status/patches recorded.',
                       'retained_original': True})
    write_json('excluded_files.json', {'files': exclusions})
    write_json('copied_files.json', {'time': now(), 'files': rows})

    # Verify bytes twice at input and once at output, and detect shared files.
    current = {n: p for n, p in inventory(SOURCE, skip_git=True).items() if selected(n)}
    source_after = {name: {'bytes': stat(path).st_size, 'sha256': sha(path)}
                    for name, path in current.items()}
    write_json('source_files_after.json', {'time': now(), 'files': source_after})
    source_changed = sorted(n for n in source_before.keys() | source_after.keys()
                            if source_before.get(n) != source_after.get(n))
    copy_errors = []
    shared_files = []
    for row in rows:
        dest = TARGET / row['destination']
        if stat(dest).st_size != row['bytes'] or sha(dest) != row['sha256']:
            copy_errors.append(row['destination'])
        if os.path.samefile(row['source'], dest):
            shared_files.append(row['destination'])

    print('Running copy-only syntax and source-identity verification...', flush=True)
    syntax_errors = []
    syntax_count = 0
    for name in copied_source:
        if not name.endswith('.py'):
            continue
        syntax_count += 1
        try:
            # compile() handles source encoding without creating pyc files or importing modules.
            compile((TARGET / name).read_bytes(), str(TARGET / name), 'exec')
        except (SyntaxError, ValueError, UnicodeError) as exc:
            syntax_errors.append({'path': name, 'error': str(exc)})
    probes = {
        'root_is_local': 'import json; from project_paths import ROOT; print(json.dumps({"root":str(ROOT)}))',
        'path_option_identity': 'import json; from chapter3_bser.path_option.source import identity; print(json.dumps(identity(),sort_keys=True))',
        'independent_source_verifier': 'import json; from pathlib import Path; from tools.verify_source import verify; print(json.dumps(verify(Path.cwd()),sort_keys=True))',
    }
    probe_results = {}
    for name, code in probes.items():
        source_probe = run_probe(SOURCE, code)
        target_probe = run_probe(TARGET, code)
        if name == 'root_is_local':
            equal = (source_probe.get('result', {}).get('root') == str(SOURCE)
                     and target_probe.get('result', {}).get('root') == str(TARGET))
        else:
            equal = (source_probe['exit_code'] == target_probe['exit_code']
                     and source_probe['stdout'] == target_probe['stdout'])
            if source_probe['exit_code']:
                # Traceback absolute paths differ after relocation; the error itself must agree.
                equal = equal and source_probe['stderr'].splitlines()[-1:] == target_probe['stderr'].splitlines()[-1:]
        probe_results[name] = {'source': source_probe, 'copy': target_probe, 'equivalent_after_relocation': equal}

    closure = {}
    for name in ('provenance/path_option_manifest_v5.json', 'provenance/pure_marl_revision3_manifest.json'):
        data = json.loads((TARGET / name).read_text(encoding='utf-8'))
        missing = []
        for path in data['files']:
            local = (TARGET / path).resolve()
            if not local.is_relative_to(TARGET.resolve()) or not local.is_file():
                missing.append(path)
        closure[name] = {'file_count': len(data['files']), 'missing': missing}

    print('Hashing complete legacy trees after copying...', flush=True)
    after = legacy_inventory()
    differences = sorted(n for n in before.keys() | after.keys() if before.get(n) != after.get(n))
    write_json('legacy_after.json', {'time': now(), 'files': after})
    legacy_summary = {}
    for folder in ('CH3', 'CH4', 'CH5'):
        files = {n: item for n, item in before.items() if n.startswith(folder + '/')}
        legacy_summary[folder] = {
            'files': len(files), 'bytes': sum(i['bytes'] for i in files.values()),
            'unchanged': not any(n.startswith(folder + '/') for n in differences),
        }
    write_json('legacy_integrity_report.json', {
        'method': 'Full recursive inventory with SHA256 of every file, before and after; includes checkpoints/results/git/caches.',
        'repositories': legacy_summary, 'changed_files': differences,
        'passed': not differences,
    })
    final_status = git('status', '--short', '--untracked-files=all').decode('utf-8')
    git_unchanged = (git('rev-parse', 'HEAD').decode().strip() == source_git['head']
                     and final_status == source_git['status']
                     and git('diff', '--binary', '--no-ext-diff') == diff_before
                     and git('diff', '--cached', '--binary', '--no-ext-diff') == staged_before)
    verification = {
        'started_at': started, 'finished_at': now(), 'source': str(SOURCE), 'destination': str(TARGET),
        'source_confirmed': True, 'source_head': source_git['head'],
        'source_worktree_copied_including_uncommitted_files': True,
        'active_files': len(copied_source), 'copied_files': len(rows),
        'copied_bytes': sum(r['bytes'] for r in rows), 'legacy_reference_files': legacy_count,
        'source_selected_files_unchanged': not source_changed, 'source_changed_files': source_changed,
        'source_git_unchanged': git_unchanged, 'copy_hash_errors': copy_errors,
        'shared_hardlinks_or_symlinks': shared_files, 'syntax_files_checked': syntax_count,
        'syntax_errors': syntax_errors, 'source_probes': probe_results,
        'manifest_closure': closure, 'legacy_unchanged': not differences,
        'training_started': False, 'packages_installed': False, 'algorithm_semantics_changed': False,
        'checkpoint_or_results_modified': False, 'git_commit_or_push': False,
        'is_complete_result_or_checkpoint_backup': False,
        'ch3_final_acceptance_claimed': False, 'ch4_implementation_or_performance_claimed': False,
    }
    verification['snapshot_passed'] = (
        not source_changed and git_unchanged and not copy_errors and not shared_files
        and not syntax_errors and not differences
        and all(v['equivalent_after_relocation'] for v in probe_results.values())
        and all(not v['missing'] for v in closure.values())
    )
    write_json('validation_report.json', verification)

    probe_lines = []
    for name, results in probe_results.items():
        probe_lines.append(f"- `{name}`: 原目录退出码 {results['source']['exit_code']}，副本退出码 {results['copy']['exit_code']}，迁移等价 {results['equivalent_after_relocation']}。")
    probe_text = '\n'.join(probe_lines)
    report = f'''# CH4 开发快照核验报告

生成时间：{verification['finished_at']}

快照完整性结果：**{'PASS' if verification['snapshot_passed'] else 'REVIEW_REQUIRED'}**。

原目录：`{SOURCE}`；新目录：`{TARGET}`。
原分支：`{source_git['branch']}`；HEAD：`{source_git['head']}`。

保留 {len(copied_source)} 个 Active 文件（含未提交/未跟踪的源码与配置），
另有 {legacy_count} 个旧 CH4 参考源文件、{len(research)} 个研究参考文件；
合计复制 {len(rows)} 个文件、{verification['copied_bytes']:,} 字节。

逐文件源/副本 SHA256 一致；独立文件检查失败数 {len(shared_files)}；
源文件捕获前后变化数 {len(source_changed)}；原 Git 状态、HEAD、工作树及暂存补丁未变化：{git_unchanged}。
这是复制期间的稳定性核验，不是文件系统原子快照，也不锁定用户之后的修改。

已对 {syntax_count} 个 Python 源文件进行无导入、无 pyc 生成的编译检查；错误数 {len(syntax_errors)}。
两个当前清单的文件依赖完整性已检查；未更改任何原始来源清单。

{probe_text}

现有来源验证器若原目录已失败，副本保留同一结果；详细错误见 JSON。
复制完整性通过不代表已有研究代码的全部来源验证、原生后端验收或性能验收通过。

CH3/CH4/CH5 的全部 {len(before)} 个文件在复制前后逐文件计算 SHA256；差异 {len(differences)}。
检查包括历史模型、实验结果、缓存和 Git 元数据，详见 legacy_before/after 与 legacy_integrity_report.json。

没有运行训练、评估 rollout、环境初始化、依赖安装、检查点加载、Git 提交或推送。
没有实现第四章算法，也没有改变奖励、观测、动作、任务成功、目标运动、扰动或通信语义。

未复制 `.git/`、缓存、`work/`、`3090结果/`。所有未复制文件仍保留在原目录。
未审阅的运行产物标为 REVIEW_REQUIRED；路径、大小及理由见 excluded_files.json。
本副本是开发源码与证据快照，不是所有原始运行结果、历史和检查点的灾备副本。

所有创建文件列于 created_files.json；该清单记录自身路径但不计算自身递归哈希。
'''
    (META / 'VALIDATION.md').write_text(report, encoding='utf-8')

    start_here = f'''# 第四章开发快照：Interval Robust

这是按用户 2026-11-12 的明确请求，从 **AUV-Thesis-Active** 当前工作树创建的独立文件副本。
目录实际名称是 `AUV-Thesis-Active`，不是 `AUV-Thesis-activate`。

## 现在从哪里工作

- 第三章继续在 `{SOURCE}` 修改和验证。
- 第四章后续在 `{TARGET}` 开发；两者没有共享文件、符号链接或自动同步。
- 本次只备份和核验，不实现算法；当前第三章尚在修改，此快照也不宣称第三章定稿或性能验收完成。
- 根目录 README.md/AGENTS.md 与原项目逐字节保留。旧文档里的绝对路径仍可能指向 Active，执行命令前应先确认当前路径及输出路径。

## 已保留

1. `core/`、`chapter3_bser/` 的当前实现（包括 path_option、safe_rl、reach_avoid 和未提交的 baselines）。
2. `configs/`、`checks/`、`tools/`、`scripts/`、`requirements.txt` 和原入口文件。
3. `docs/`、`provenance/` 和 `verification/`；包括验证器引用的历史源码证据压缩包，原有来源身份不改写。
4. `references/research/` 中的第四章详细 PDF 方案、可检索全文、原参考论文 `1-223081.pdf` 和方案生成时的检查记录。
5. `references/legacy_ch4_source/` 中的旧 CH4 文本/代码，只供后续比较、迁移和重写参考，不导入当前运行链。

## 旧 CH4 的候选参考

- `registry/rbe_disturbance.py`、`utils/rbe_sampler.py`：扰动参数、采样和边界实验组织。
- `utils/reb_dataset.py`、`utils/reb_model.py`、`utils/rbe_metrics.py`：数据记录、边界模型与指标思路。
- `registry/ch4_stress_tiers.py`、`tools/eval_ch4_recoverable_boundary_grid.py`：分级扰动和边界评测组织。
- 旧 `env.py`、`train.py`、MADDPG 与配套工具：仅用于核对既有假设；尚未完成对当前代码的兼容性审查。

这些候选均未接入、未宣称可直接运行，也不继承旧模型与性能结论。旧数据/权重没有复制或重新命名。

## 后续开发顺序

先保留本快照的来源边界并确认第三章公共接口，再按 PDF 方案审查区间参数定义、历史窗口估计、
训练所用策略后端及风险目标，形成第四章自己的来源演化与实验配置。
当前快照同时保留第三章不同研究分支，复制完成不代表已经确定第四章最终底层策略。
第三章后续必要修复应逐项审查并迁移；不要把整个 Active 再覆盖到本目录。

后续运行前仍须完成所选路径的来源校验和原生执行准入；本次只做静态核验。
`work/` 是本副本自己的新输出位置；本次没有启动运行或建立指向旧输出的链接。

## 快照与保留范围

- 来源 HEAD：`{source_git['head']}`，分支 `{source_git['branch']}`；实际复制内容还包含当时未提交和未跟踪文件。
- `.git/` 未复制，没有创建分支、提交或远程；修改前的 HEAD、状态与补丁在 `_snapshot/` 中保留。
- 大型 `work/`、`3090结果/`、缓存未复制；原文件仍留在 Active。需用旧模型时先审查其来源清单。
- 复制清单：[copied_files.json](_snapshot/copied_files.json)；未复制清单：[excluded_files.json](_snapshot/excluded_files.json)。
- 核验结果：[VALIDATION.md](_snapshot/VALIDATION.md)；创建清单：[created_files.json](_snapshot/created_files.json)。
- `references/` 与原 `provenance/verification/` 是参考和历史证据，后续新增实验应另建输出与来源身份。

工作区总规则中生产目录的旧命名约定不由本次备份自动改写。本目录按本次用户指令建立为独立 CH4 开发准备副本；
本次未对旧仓库和既有运行行为作生产修改。
'''
    (TARGET / 'CH4_START_HERE.md').write_text(start_here, encoding='utf-8')
    work = TARGET / 'work'
    work.mkdir(exist_ok=False)
    (work / 'README.md').write_text(
        '此目录为 CH4 副本独立的新运行输出位置。备份时未复制原 work，也未启动任何实验。\n'
        '后续手动研究运行须使用新子目录和独立来源身份，不覆盖任何历史实验。\n', encoding='utf-8')

    created = []
    for name, path in inventory(TARGET).items():
        created.append({'path': name, 'absolute_path': str(path),
                        'bytes': stat(path).st_size, 'sha256': sha(path),
                        'role': 'COPIED_INPUT' if name in {r['destination'] for r in rows} else 'CREATED_METADATA'})
    created.append({'path': '_snapshot/created_files.json',
                    'absolute_path': str(META / 'created_files.json'),
                    'role': 'SELF_INVENTORY', 'sha256': None,
                    'note': 'Self path recorded; recursive self hash intentionally omitted.'})
    write_json('created_files.json', {'time': now(), 'file_count': len(created), 'files': created})
    # Last read-only self-check. No output file is created after created_files.json.
    actual = inventory(TARGET)
    assert set(actual) == {r['path'] for r in created}
    for row in created:
        if row['sha256'] is not None:
            assert sha(TARGET / row['path']) == row['sha256'], row['path']
    print(json.dumps({
        'snapshot_passed': verification['snapshot_passed'], 'target': str(TARGET),
        'copied_files': len(rows), 'created_files_total': len(created),
        'copied_bytes': verification['copied_bytes'], 'legacy': legacy_summary,
        'source_probes': {n: {'source_exit': v['source']['exit_code'], 'copy_exit': v['copy']['exit_code'],
                              'equivalent': v['equivalent_after_relocation']} for n, v in probe_results.items()},
        'validation_report': str(META / 'VALIDATION.md'),
    }, ensure_ascii=False, indent=2), flush=True)
    if not verification['snapshot_passed']:
        raise SystemExit('Snapshot requires review; no input file was changed or deleted.')


if __name__ == '__main__':
    main()
