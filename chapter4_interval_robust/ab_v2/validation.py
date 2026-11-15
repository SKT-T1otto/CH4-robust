"""Native v2 mechanism receipt, tied to the selected config and source."""
import importlib.util,json,platform,unittest
from ..configuration import ROOT,new_output
from ..provenance import identity,sha,write_new,digest
from .configuration import frozen_contract

def execute(c,output):
    if not output:raise ValueError('fresh verification output required')
    source=identity();folder=new_output(output)
    spec=importlib.util.spec_from_file_location('ch4_ab_v2_checks',ROOT/'checks/chapter4/ab_v2_tests.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.CONFIG_PATH=ROOT/'configs/ch4'/('experiment_ab_v2_n1.json' if c['noise_protocol']=='N1' else 'experiment_ab_v2.json')
    module.BASE_CONFIG=c
    module.OUTPUT=folder
    suite=unittest.defaultTestLoader.loadTestsFromModule(module)
    with (folder/'tests.log').open('x',encoding='utf-8') as log:result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    write_new(folder/'mechanism_ledger.json',module.LEDGER)
    report=dict(schema='auv.ch4.ab_validation.v2',source=source,noise_protocol=c['noise_protocol'],config_sha256=digest(c),
        configuration_contract_sha256=digest(frozen_contract(c,False)),
        platform=platform.system(),python=platform.python_version(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        skipped=len(result.skipped),passed=result.wasSuccessful() and not result.skipped and source==identity(),
        native_environment_checked=module.LEDGER['native_environment_checked'],
        artifacts={n:sha(folder/n) for n in ('tests.log','mechanism_ledger.json')},performance_passed=False,original_sirsa_reproduced=False,long_training_started=False)
    write_new(folder/'validation.json',report);return dict(output=str(folder),**report)
