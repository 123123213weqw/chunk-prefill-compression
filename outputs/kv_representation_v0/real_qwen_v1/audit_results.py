"""Validate downloaded result integrity and independently recompute summary means."""
import collections,gzip,hashlib,json,math
from pathlib import Path
p=Path(__file__).resolve().parent/'runs/4080-pilot'
expected='9b87a5a2ac88ddca05ee774236b5d44e561d7cecfef2dbba50342aa9a912213d'
assert hashlib.sha256((p/'records.jsonl.gz').read_bytes()).hexdigest()==expected
manifest=json.loads((p/'manifest.json').read_text())
assert manifest['status']=='completed' and manifest['test_set_used'] is False
for name,field in [('run.py','script_sha256'),('smoke4.jsonl','dataset_sha256')]:
 assert hashlib.sha256((p/'source'/name).read_bytes()).hexdigest()==manifest[field]
metrics=['mixed_relative_l2','chunk_relative_l2','chunk_logZ_error','original_chunk_mass','mass_abs_error']
counts=collections.Counter(); sums=collections.defaultdict(lambda:collections.Counter());total=0
with gzip.open(p/'records.jsonl.gz','rt') as f:
 for line in f:
  r=json.loads(line);total+=1
  assert r['sample_id'] in manifest['samples'] and r['distance']>=1
  for metric in metrics: assert math.isfinite(r[metric]), (metric,r)
  assert 0<=r['original_chunk_mass']<=1.000001
  for scope in ['all','layer_'+str(r['layer']),r['bucket']]:
   key=(scope,r['method'],r['m']);counts[key]+=1
   for metric in metrics:sums[key][metric]+=r[metric]
assert total==manifest['observations']==261120
for r in json.loads((p/'summary.json').read_text()):
 key=(r['scope'],r['method'],r['m']);assert counts[key]==r['n']
 for metric in metrics: assert abs(sums[key][metric]/counts[key]-r[metric]['mean'])<1e-10
checks=json.loads((p/'reference_checks.json').read_text());assert len(checks)==160
assert max(c['relative_rms'] for c in checks)<.005
report={'status':'PASS','records':total,'reference_checks':len(checks),'finite_metrics':True,
        'summary_means_independently_verified':True,'source_dataset_and_records_hashes_verified':True}
(p/'AUDIT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
