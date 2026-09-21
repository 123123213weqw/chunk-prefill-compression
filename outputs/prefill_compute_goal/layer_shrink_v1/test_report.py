"""Audit tests; RUN_DIRECTORY enables tamper tests on copies of a completed run."""
import copy,json,os,shutil,tempfile,unittest
from pathlib import Path
from report_experiment import audit,close,coverage,render,EXPECTED,CANDIDATES

class TestAuditHelpers(unittest.TestCase):
    def test_exact_coverage(self):coverage([{'k':1},{'k':2}],('k',),[(1,),(2,)])
    def test_duplicate_rejected(self):
        with self.assertRaises(AssertionError):coverage([{'k':1},{'k':1}],('k',),[(1,),(2,)])
    def test_missing_rejected(self):
        with self.assertRaises(AssertionError):coverage([{'k':1}],('k',),[(1,),(2,)])
    def test_unexpected_rejected(self):
        with self.assertRaises(AssertionError):coverage([{'k':3}],('k',),[(1,)])
    def test_bad_recomputed_number_rejected(self):
        for a,b in ((.1,.2),(float('nan'),1)):
            with self.assertRaises(AssertionError):close(a,b)
    def test_report_no_success_claim_from_audit_pass(self):
        scores={m:dict(readout_correct=2,control_correct=4,n=4,field_correct=8,fields=14,
                      pairs_both_correct=1,pairs=2,format_failures=0,token_caps=0,templates={'toy':{'correct':2,'n':4}}) for m in EXPECTED}
        scores['native_full']['readout_correct']=4
        a={'scores':scores,'identity_gates':16,'shape_audits':24,'linear_flop_ratios':{m:[.8] for m in CANDIDATES},
           'all_trial_boundaries_isolated':False,'timing_summary':[]}
        s=render(a);self.assertIn('未达到',s);self.assertIn('仅为诊断值',s)

@unittest.skipUnless(os.environ.get('RUN_DIRECTORY'),'requires completed real run for tamper tests')
class TestCompletedRunAudit(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'copy';shutil.copytree(os.environ['RUN_DIRECTORY'],self.path)
    def mutate(self,name,fn):
        p=self.path/name;x=json.loads(p.read_text());fn(x);p.write_text(json.dumps(x))
        with self.assertRaises(AssertionError):audit(self.path)
    def test_intact(self):self.assertEqual(audit(self.path)['status'],'PASS')
    def test_duplicate_row(self):self.mutate('correctness.json',lambda x:x.append(copy.deepcopy(x[0])))
    def test_gold_changed(self):self.mutate('correctness.json',lambda x:x[0]['gold'].update({'injected':1}))
    def test_score_changed(self):self.mutate('correctness.json',lambda x:x[0].update(correct=not x[0]['correct']))
    def test_shape_changed(self):self.mutate('shape_audits.json',lambda x:x[0]['actual_module_inputs'][0].update(tokens=1))
    def test_isolation_claim_changed(self):self.mutate('timings.json',lambda x:x[0].update(isolation_observed=not x[0]['isolation_observed']))
    def test_source_changed(self):
        with (self.path/'source/engine.py').open('a') as f:f.write('\n# tampered\n')
        with self.assertRaises(AssertionError):audit(self.path)
    def test_summary_changed(self):self.mutate('performance_summary.json',lambda x:x[0]['prefix_seconds'].update(median=999))

if __name__=='__main__':unittest.main(verbosity=2)
