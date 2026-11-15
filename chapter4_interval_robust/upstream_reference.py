"""Prepare a pinned original SIRSA checkout as data; never launch its trainer."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile
from .configuration import new_output
from .provenance import write_new

COMMIT='ed550ca2611468bb45154b735ee89ecd611d23a2'
URL=f'https://codeload.github.com/anxie/sirsa/zip/{COMMIT}'


def prepare(output):
    output=new_output(output)
    archive=output/'official_sirsa.zip'
    request=urllib.request.Request(URL,headers={'User-Agent':'CH4-source-research/1.0'})
    with urllib.request.urlopen(request,timeout=60) as response,archive.open('xb') as f:
        total=0
        while True:
            block=response.read(1024*1024)
            if not block:break
            total+=len(block)
            if total>64*1024*1024:raise ValueError('unexpectedly large source archive')
            f.write(block)
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            destination=(output/entry.filename).resolve()
            if not destination.is_relative_to(output):raise ValueError('unsafe archive path')
            if entry.is_dir():destination.mkdir(parents=True,exist_ok=True)
            else:
                destination.parent.mkdir(parents=True,exist_ok=True)
                with destination.open('xb') as f:f.write(z.read(entry))
    source=output/f'sirsa-{COMMIT}'
    receipt=dict(repository='https://github.com/anxie/sirsa',commit=COMMIT,url=URL,
                 archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 source=str(source),declared_license='MIT in setup.py; no standalone LICENSE in tree',
                 redistribution_review='REVIEW_REQUIRED',training_started=False,dependencies_installed=False,
                 original_reproduction_passed=False)
    write_new(output/'source_receipt.json',receipt)
    commands=f'''Original upstream reference only; separate from the AUV implementation.
cd "{source}"
conda env create -n ch4-sirsa-original -f environment.yml
conda activate ch4-sirsa-original
python -m pip install -e .

The original TensorFlow/softlearning/MuJoCo stack is NOT installed by CH4.
Freeze and review the original reference training budget before manual execution.
Original algorithm options: SAC, Reacher2D, observed context set, predicted context,
history_length=5, latent_dim=2, ensemble_size=4, cvar_alpha=.25, cvar_samples=50.
Use upstream examples.development.main and its supported parser/variants to set
the separately reviewed budget; do not claim AUV checks reproduce this experiment.
No trainer was launched by this preparation command.
'''
    (output/'REFERENCE_COMMANDS.txt').write_text(commands,encoding='utf-8')
    return receipt
