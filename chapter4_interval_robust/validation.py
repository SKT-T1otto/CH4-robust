"""Focused CH4 tests; creates an independently hashed native validation receipt."""
import importlib
import platform
import unittest
from .configuration import ROOT,new_output
from .provenance import identity,sha,write_new


def execute(output):
    source=identity();folder=new_output(output)
    suite=unittest.defaultTestLoader.discover(str(ROOT/'checks/chapter4'),pattern='test_*.py')
    with (folder/'tests.log').open('x',encoding='utf-8') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    module=importlib.import_module('test_ch4')
    ledger=module.LEDGER
    write_new(folder/'mechanism_ledger.json',ledger)
    current=identity()
    report=dict(schema='auv.ch4.validation.v1',source=source,platform=platform.system(),
                python=platform.python_version(),tests=result.testsRun,
                failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
                passed=result.wasSuccessful() and not result.skipped and current==source,
                native_environment_checked=bool(ledger.get('nominal_pair_passed')),
                native_linux_accepted=platform.system()=='Linux' and result.wasSuccessful(),
                artifacts={name:sha(folder/name) for name in ('tests.log','mechanism_ledger.json')},
                long_training_started=False,performance_passed=False,original_sirsa_reproduced=False)
    write_new(folder/'validation.json',report)
    return dict(output=str(folder),**report)
