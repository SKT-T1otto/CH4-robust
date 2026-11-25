"""Append current source records while retaining the inherited validation failures."""
import sys
from datetime import datetime, timezone
from pathlib import Path

from chapter3_bser.learning.checkpoint import digest
from chapter4_interval_robust.provenance import identity as parent_identity
from chapter4_interval_robust.provenance import record as record_parent
from chapter4_interval_robust.provenance import write_new

from .config import ROOT, file_sha, local_path, read_json


SOURCE_FOLDER = ROOT / "provenance/ch4/nominal_maddpg"
SUFFIXES = {".py", ".json", ".yaml", ".yml"}


def inventory(root=ROOT):
    files = {}
    directories = ("core", "chapter3_bser", "configs", "chapter4_interval_robust/nominal_maddpg")
    for relative in directories:
        for path in sorted((root/relative).rglob("*")):
            if path.is_file() and path.suffix in SUFFIXES and "__pycache__" not in path.parts:
                files[path.relative_to(root).as_posix()] = file_sha(path)
    fixed = ("project_paths.py", "requirements.txt", "requirements-ch4.txt",
             "chapter4_interval_robust/nominal_maddpg/README.md",
             "chapter4_interval_robust/__init__.py", "chapter4_interval_robust/scenarios.py",
             "chapter4_interval_robust/provenance.py", "chapter4_interval_robust/configuration.py",
             "chapter4_interval_robust/parameters.py", "checks/chapter4/test_nominal_maddpg.py",
             "checks/__init__.py", "checks/chapter4/__init__.py")
    for relative in fixed:
        path = root/relative
        if path.is_file():
            files[relative] = file_sha(path)
    for path in sorted((root/"checks/chapter4").glob("test_nominal_*.py")):
        files[path.relative_to(root).as_posix()] = file_sha(path)
    for path in sorted((root/"experiments").rglob("*")):
        if path.is_file() and path.suffix in (".py", ".json") and "__pycache__" not in path.parts:
            files[path.relative_to(root).as_posix()] = file_sha(path)
    return dict(sorted(files.items()))


def compare_inventory(expected, actual):
    changed = [name for name in sorted(expected.keys() | actual.keys()) if expected.get(name) != actual.get(name)]
    if changed:
        raise ValueError("registered source/input inventory differs: " + repr(changed[:12]))


def verify_record(record, actual):
    if record.get("schema") != "auv.ch4.nominal_maddpg.source.v1":
        raise ValueError("unexpected source record schema")
    if record.get("sha256") != digest({k:v for k,v in record.items() if k != "sha256"}):
        raise ValueError("source record digest differs")
    compare_inventory(record["files"], actual)


def append_source(config_path, *, stage=2, reason=None):
    records = sorted(SOURCE_FOLDER.glob("source_v[0-9][0-9][0-9][0-9].json"))
    old = read_json(records[-1]) if records else None
    actual = inventory()
    if old is not None and old["files"] == actual:
        raise ValueError("no source evolution to register")
    reason = reason or "Add independent nominal single-critic MADDPG contracts, geometry registry and steps 1-2 checks"
    try:
        parent = parent_identity()
    except ValueError as exc:
        if "source changed after source record" not in str(exc):
            raise
        parent = record_parent(reason)
    data = dict(schema="auv.ch4.nominal_maddpg.source.v1", created_utc=datetime.now(timezone.utc).isoformat(),
                config_sha256=file_sha(config_path), files=actual, parent_source=parent,
                previous_source_sha256=None if old is None else old["sha256"],
                registered_steps=list(range(1, stage+1)), reason=reason,
                inherited_source_checks_passed=False, completed_steps=[1, 2], training_enabled=False,
                task_rollouts=0, optimizer_updates=0, nominal_performance_verified=False)
    data["sha256"] = digest(data)
    write_new(SOURCE_FOLDER/f"source_v{len(records)+1:04d}.json", data)
    return data


def verify_source():
    paths = sorted(SOURCE_FOLDER.glob("source_v[0-9][0-9][0-9][0-9].json"))
    if not paths:
        raise ValueError("nominal source inventory not registered")
    previous = None
    for path in paths:
        record = read_json(path)
        if (record.get("previous_source_sha256") != previous
                or record["sha256"] != digest({k:v for k,v in record.items() if k != "sha256"})):
            raise ValueError("invalid nominal source evolution chain")
        previous = record["sha256"]
    verify_record(record, inventory())
    parent = parent_identity()
    if parent != record["parent_source"]:
        raise ValueError("Chapter 4 parent source identity changed")
    return record


def imported_local_files(record):
    matched = {}
    prefixes = ("core", "chapter3_bser", "chapter4_interval_robust", "experiments", "project_paths")
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        filename = vars(module).get("__file__")
        if not isinstance(filename, (str, Path)) or str(filename).startswith("<"):
            continue
        path = Path(filename).resolve()
        if not path.is_relative_to(ROOT):
            if any(name == p or name.startswith(p + ".") for p in prefixes):
                raise ValueError("import escaped the independent project: " + name)
            continue
        relative = path.relative_to(ROOT).as_posix()
        if relative not in record["files"] or file_sha(local_path(relative)) != record["files"][relative]:
            raise ValueError("imported local file is absent from source identity: " + relative)
        matched[name] = relative
    return matched
