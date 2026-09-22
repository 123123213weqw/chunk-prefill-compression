import argparse,json
from pathlib import Path
from transformers import AutoTokenizer
import data
from specs import SEEDS
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);a=p.parse_args();tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True);rows=[]
 for seed in SEEDS:
  data.SEED=seed;rr=data.build(tok)
  for r in rr:r.update(generator_seed=seed,original_split=r['split'],split='replication_v5')
  rows.extend(rr);print('GENERATED',seed,len(rows),flush=True)
 previous=[ROOT.parent/'long_validation_v4/fresh.jsonl',ROOT.parent/'compression_depth_v3/dataset.jsonl']
 ids={r['sample_id'] for r in rows};assert len(rows)==len(ids)==256
 for f in previous:assert ids.isdisjoint({json.loads(l)['sample_id'] for l in f.read_text().splitlines()})
 for r in rows:assert data.expected_from_text(r)==r['gold'] and len(tok.encode(r['T'],add_special_tokens=False))==r['target_tokens']
 with (ROOT/'fresh.jsonl').open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 (ROOT/'preparation.json').write_text(json.dumps({'samples':256,'seeds':SEEDS,'old_ids_disjoint':True,'gold_reparsed':True},indent=2)+'\n')
 print('DATA_PASS',flush=True)
