"""Development matrix/statistics tests; synthetic fixtures are NOT model results."""
import collections,copy,hashlib,json,os,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from dataset_checks import validate_development
from paired_stats import comparison
from report_experiment import audit,EXPECTED,OPS,TIMED,METRICS
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'development.jsonl' if (ROOT/'development.jsonl').exists() else ROOT.parents[1]/'validation_v3/dataset/development.jsonl'
SCORER=ROOT/'methods.py' if (ROOT/'methods.py').exists() else ROOT.parents[1]/'validation_v3/methods.py'

def load():return [json.loads(l) for l in DATA.read_text().splitlines()]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False))
def fixture(root):
    """Build a clearly labeled temporary synthetic audit fixture, never a run."""
    (root/'source').mkdir()
    for p in [*ROOT.glob('*.py'),ROOT/'PROTOCOL.md',SCORER]:shutil.copy2(p,root/'source'/p.name)
    shutil.copy2(DATA,root/'source/development.jsonl')
    data=sorted([s for s in load() if s['length']==4096],key=lambda s:(s['template'],s['evidence_layout'],s['instance'],s['member']))
    selected=[s['sample_id'] for s in data if s['instance']==s['member']==0]
    manifest={'status':'completed','SYNTHETIC_UNIT_TEST_ONLY':True,'test_used':False,'shape_hooks_excluded_from_timing':True,
        'args':{'block':1024,'max_tokens':96,'warmups':1,'repeats':5,'length':4096},
        'method_specs':{m:dict(zip(('method','split_depth','keep','native'),v)) for m,v in EXPECTED.items()},'timing_methods':list(TIMED),
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'source').iterdir()},
        'dataset_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'selected_sample_ids':[s['sample_id'] for s in data],
        'sample_count':32,'pair_count':16,'identity_gate_expected':128,'timing_sample_ids':selected,
        'gpu_at_start':{'status':'OK','foreign_compute_processes':[]}}
    dump(root/'manifest.json',manifest)
    dump(root/'DATA_VALIDATION.json',{'status':'PASS','samples':96,'pairs':48,'tokenizer_revalidated':True,'gold_reparsed':True,'single_field_interventions_verified':True})
    rows=[];gates=[];shapes=[]
    for s in data:
        sid=s['sample_id'];lengths={k:s['segments'][k]['token_count'] for k in ('S','T','R')};full=sum(lengths.values())
        for name,(_,depth,keep,_) in EXPECTED.items():
            upper=full-lengths['T']+lengths['T']//64*keep;layer_lengths=[full]*depth+[upper]*(36-depth)
            ops=[{'layer':i,'op':op,'tokens':layer_lengths[i],'in_features':8,'out_features':8,'calls':6} for i in range(36) for op in OPS]
            shapes.append({'sample_id':sid,'method':name,'status':'OK','segment_lengths':lengths,'split_depth':depth,'keep':keep,
                'lower_layer_tokens':full,'upper_layer_tokens':upper,'actual_module_inputs':ops,
                'linear_flops':sum(128*r['tokens'] for r in ops),'linear_token_layer_ratio':sum(layer_lengths)/(36*full)})
            for p in s['probes']:
                if p['probe_id'] not in ('readout','control'):continue
                gold=p['gold'];rows.append({'sample_id':sid,'pair_id':s['pair_id'],'template':s['template'],'length':s['length'],
                    'evidence_layout':s['evidence_layout'],'instance':s['instance'],'member':s['member'],'method':name,'probe':p['probe_id'],
                    'status':'OK','gold':gold,'answer':json.dumps(gold),'parsed':gold,'correct':True,'format_ok':True,
                    'field_correct':len(gold),'fields':len(gold),'generated_tokens':1,'generated_ids':[1],'hit_token_cap':False,
                    'layer_lengths':layer_lengths,'cache_bytes':sum(layer_lengths)*32,'position_metadata_bytes':sum(layer_lengths)*8})
                if name in ('identity_d12','identity_d24'):
                    gates.append({'sample_id':sid,'identity_method':name,'probe':p['probe_id'],'generated_tokens_equal':True,
                                  'max_cache_relative_rms':0.,'logit_relative_rms':0.,'first_token_kl':0.})
    dump(root/'correctness.json',rows);dump(root/'identity_gates.json',gates);dump(root/'shape_audits.json',shapes)
    summary={}
    for m in EXPECTED:
        summary[m]={}
        for q in ('readout','control'):
            rr=[r for r in rows if r['method']==m and r['probe']==q]
            summary[m][q]={'correct':32,'n':32,'unavailable':0,'field_correct':sum(r['field_correct'] for r in rr),'fields':sum(r['fields'] for r in rr)}
    dump(root/'correctness_summary.json',summary)
    telemetry={'status':'OK','foreign_compute_processes':[]};trials=[];perf=[]
    for sid in sorted(selected):
        for m in TIMED:
            values={'prefix_seconds':1.,'query_ttft_seconds':.1,'total_ttft_seconds':1.1,'kv_bytes_after_Q':1000,'peak_allocated_bytes':2000,'peak_allocated_over_model_bytes':1000}
            for repeat in range(-1,5):trials.append({'sample_id':sid,'method':m,'repeat':repeat,'warmup':repeat<0,'status':'OK',
                **values,'gpu_before':telemetry,'gpu_after':telemetry,'isolation_observed':True})
            perf.append({'sample_id':sid,'method':m,'measured':5,'oom':0,**{k:{'median':v,'min':v,'max':v} for k,v in values.items()}})
    dump(root/'timings.json',trials);dump(root/'performance_summary.json',perf)

class TestMatrix(unittest.TestCase):
    def test_01_all_96_development(self):self.assertEqual(validate_development(load())['pairs'],48)
    def test_02_test_split_rejected(self):
        rows=load();rows[0]['split']='test'
        with self.assertRaises(AssertionError):validate_development(rows)
    def test_03_duplicate_rejected(self):
        rows=load();rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaises(AssertionError):validate_development(rows)
    def test_04_gold_tamper_rejected(self):
        rows=load();next(iter(rows[0]['probes']))['gold']['injected']=42
        with self.assertRaises(AssertionError):validate_development(rows)
    def test_05_bootstrap_keeps_pairs_and_counts_harm(self):
        rows=[]
        for pair in range(4):
            for member in (0,1):
                for m in ('native_full','candidate'):
                    rows.append({'sample_id':f'{pair}-{member}','pair_id':str(pair),'template':'a','length':4096,'evidence_layout':'distributed',
                        'method':m,'probe':'readout','status':'OK','correct':m=='native_full' or pair!=0})
        c=comparison(rows,'candidate');self.assertEqual(c['pairs'],4);self.assertEqual(c['method_minus_full'],-.25)
        self.assertEqual(c['full_correct_to_method_wrong'],2);self.assertEqual(c['full_wrong_to_method_correct'],0)
        self.assertEqual(c,comparison(rows,'candidate'));self.assertLessEqual(c['ci95'][0],-.25)
        self.assertGreaterEqual(c['ci95'][1],-.25)
    def test_06_oom_not_removed_from_denominator(self):
        rows=[]
        for member in (0,1):
            for m in ('native_full','candidate'):
                rows.append({'sample_id':str(member),'pair_id':'p','template':'a','length':4096,'evidence_layout':'clustered',
                    'probe':'readout','method':m,'correct':m=='native_full','status':'OK' if m=='native_full' else 'OOM'})
        c=comparison(rows,'candidate');self.assertEqual(c['n'],2);self.assertEqual(c['unavailable_method'],2);self.assertEqual(c['method_minus_full'],-1)
    def test_07_complete_audit_fixture_and_tamper_suite(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td);fixture(path);a=audit(path)
            self.assertEqual(a['sample_count'],32);self.assertEqual(a['pair_count'],16)
            self.assertEqual(a['scores']['native_full']['pairs'],16)
            self.assertEqual(a['correctness_records'],384);self.assertEqual(a['timing_trials'],168)
            result=subprocess.run([sys.executable,str(ROOT/'test_report.py')],env=dict(os.environ,RUN_DIRECTORY=str(path)),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
    def test_08_timing_selection_is_fixed_and_balanced(self):
        selected=[r for r in load() if r['instance']==0 and r['member']==0]
        self.assertEqual(len(selected),12)
        self.assertEqual(len({(r['template'],r['length'],r['evidence_layout']) for r in selected}),12)

if __name__=='__main__':unittest.main(verbosity=2)
