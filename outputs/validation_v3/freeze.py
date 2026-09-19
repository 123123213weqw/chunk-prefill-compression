#!/usr/bin/env python3
"""Unlock held-out evaluation only after complete, unchanged dev calibration."""
import argparse
import hashlib
import json
from pathlib import Path
from data import read
from report import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--dataset-dir', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    ds = Path(a.dataset_dir)
    expected = {r['sample_id'] for r in read(ds / 'development.jsonl')}
    if len(expected) != 96:
        raise ValueError('expected 96 frozen development samples')
    manifests, summaries, runtime = [], [], []
    for path, phase in [(Path(a.calibration), 'calibration'), (Path(a.baseline), 'baseline')]:
        m = json.loads((path / 'manifest.json').read_text())
        rows = read(path / 'results.jsonl')
        status = json.loads((path / 'status.json').read_text())
        if status['status'] != 'COMPLETE' or status['completed'] != 96 or len(rows) != 96 or {r['sample_id'] for r in rows} != expected:
            raise ValueError('requires complete 96-sample dev runs; smoke cannot unlock test')
        if m['phase'] != phase or m['dataset_sha256'] != hashlib.sha256((ds / 'development.jsonl').read_bytes()).hexdigest():
            raise ValueError('wrong phase or development dataset')
        if any(r['split'] != 'development' for r in rows):
            raise ValueError('development only')
        manifests.append(m)
        summaries.append(summarize(path))
        runtime.append(json.loads((path / 'runtime.json').read_text()))
    if manifests[0]['code_hashes'] != manifests[1]['code_hashes']:
        raise ValueError('code changed between calibration and baseline')
    if manifests[0]['model'] != manifests[1]['model'] or runtime[0]['tokenizer_hashes'] != runtime[1]['tokenizer_hashes']:
        raise ValueError('model/tokenizer mismatch')
    if summaries[0]['gates']['short_readout_ge95'] != 'PASS' or summaries[0]['gates']['short_control_ge95'] != 'PASS':
        raise ValueError('short readout/control gate failed; test remains locked')
    if summaries[1]['gates']['long_readout_per_length_ge90'] != 'PASS':
        raise ValueError('long Full readout gate failed; test remains locked')
    lock = {'code_hashes': manifests[0]['code_hashes'], 'test_sha256': hashlib.sha256((ds / 'test.jsonl').read_bytes()).hexdigest(),
            'model': manifests[0]['model'], 'ratios': [2, 4, 8], 'random_seeds': [0, 1, 2, 3, 4],
            'calculation_primary': summaries[0]['gates']['short_calculation_ge90'] == 'PASS',
            'note': 'If calculation_primary=false, math results are diagnostic only; no filtering failed examples.',
            'dev_summary_hashes': [hashlib.sha256((Path(x) / 'summary.json').read_bytes()).hexdigest() for x in (a.calibration, a.baseline)]}
    output = Path(a.output)
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(lock, indent=2)+'\n')
    print('FROZEN', output)


if __name__ == '__main__':
    main()
