"""Negative tests: frozen held-out guard must fail BEFORE model loading/writes."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class LockTest(unittest.TestCase):
    def test_test_split_requires_lock_before_loading_model(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ds = td / 'test.jsonl'
            ds.write_text(''.join(json.dumps({'pair_id': 'pair', 'sample_id': str(m), 'member': m, 'split': 'test'})+'\n' for m in (0, 1)))
            result = subprocess.run([sys.executable, str(ROOT/'run.py'), '--model', '/nonexistent-model',
                                     '--dataset', str(ds), '--out', str(td/'out'), '--phase', 'suite'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('held-out test requires frozen development lock', result.stderr)
            self.assertFalse((td/'out').exists())

    def test_single_member_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ds = td / 'dev.jsonl'
            ds.write_text(json.dumps({'pair_id': 'p', 'sample_id': 's', 'member': 0, 'split': 'development'})+'\n')
            result = subprocess.run([sys.executable, str(ROOT/'run.py'), '--model', '/nonexistent-model',
                                     '--dataset', str(ds), '--out', str(td/'out'), '--phase', 'calibration'], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('input must contain complete pairs', result.stderr)
            self.assertFalse((td/'out').exists())

    def test_smoke_cannot_unlock(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            ds, cal = td/'data', td/'cal'
            ds.mkdir(); cal.mkdir()
            (ds/'development.jsonl').write_text(''.join(json.dumps({'sample_id': str(i)})+'\n' for i in range(96)))
            (cal/'manifest.json').write_text('{}')
            (cal/'results.jsonl').write_text(''.join(json.dumps({'sample_id': str(i)})+'\n' for i in range(4)))
            (cal/'status.json').write_text(json.dumps({'status': 'COMPLETE', 'completed': 4}))
            result = subprocess.run([sys.executable, str(ROOT/'freeze.py'), '--calibration', str(cal), '--baseline', str(cal),
                                     '--dataset-dir', str(ds), '--output', str(td/'LOCK.json')], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires complete 96-sample dev runs', result.stderr)
            self.assertFalse((td/'LOCK.json').exists())


if __name__ == '__main__':
    unittest.main()
