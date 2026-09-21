import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
import data
ROOT=Path(__file__).resolve().parent

def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--v3',required=True);a=p.parse_args();old=Path(a.v3)
 tok=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True);rows=[]
 for seed in range(20260921,20260925):
  data.SEED=seed;rr=data.build(tok)
  for r in rr:r.update(generator_seed=seed,original_split=r['split'],split='fresh_validation')
  rows.extend(rr);print('generated',seed,len(rows),flush=True)
 assert len(rows)==256 and len({r['sample_id'] for r in rows})==256
 oldrows=[json.loads(l) for l in (old/'dataset.jsonl').read_text().splitlines()];assert not ({r['sample_id'] for r in rows}&{r['sample_id'] for r in oldrows})
 for r in rows:assert data.expected_from_text(r)==r['gold'] and len(tok.encode(r['T'],add_special_tokens=False))==r['target_tokens']
 with (ROOT/'fresh.jsonl').open('x') as f:
  for r in rows:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 prior=[json.loads(l) for l in (old/'correctness.jsonl').read_text().splitlines()];prior=[r for r in prior if r['split']=='validation' and r['probe']=='task'];lookup={(r['sample_id'],r['method']):r for r in prior}
 candidates=[r for r in oldrows if r['split']=='validation' and lookup[(r['sample_id'],'full')]['correct']]
 def failed(r,n):return not lookup[(r['sample_id'],n)]['correct']
 chosen=[]
 for name in ('d27_k32','adaptive_equal','d18_k48'):
  for r in sorted(candidates,key=lambda r:r['sample_id']):
   if failed(r,name) and r not in chosen:chosen.append(r)
 cells=defaultdict(list)
 for r in sorted(candidates,key=lambda r:r['sample_id']):
  if r not in chosen:cells[(r['task'],r['target_tokens'])].append(r)
 while len(chosen)<16:
  moved=False
  for key in sorted(cells):
   if cells[key] and len(chosen)<16:chosen.append(cells[key].pop(0));moved=True
  if not moved:break
 chosen=chosen[:16]
 with (ROOT/'diagnostic.jsonl').open('x') as f:
  for r in chosen:f.write(json.dumps({**r,'split':'exposed_v3_diagnostic','selection_prior':{n:lookup[(r['sample_id'],n)]['correct'] for n in ('full','d18_k48','d27_k32','adaptive_equal')}},ensure_ascii=False)+'\n')
 # Include their paired counterfactual gold even if counterpart is not selected.
 dump(ROOT/'counterfactuals.json',{r['sample_id']:next(x['gold'] for x in oldrows if x['pair_id']==r['pair_id'] and x['member']!=r['member']) for r in chosen})
 threshold=json.loads((old/'thresholds.json').read_text());dump(ROOT/'thresholds.json',threshold)
 dump(ROOT/'preparation.json',{'fresh_n':len(rows),'diagnostic_n':len(chosen),'v3_source':str(old),'diagnostic_chunks':sum(r['target_tokens']//64 for r in chosen),'single_chunk_trials':4*sum(r['target_tokens']//64 for r in chosen),'fresh_seeds':list(range(20260921,20260925)),'gold_reparsed':True,'old_ids_disjoint':True})
 print((ROOT/'preparation.json').read_text())
if __name__=='__main__':main()
