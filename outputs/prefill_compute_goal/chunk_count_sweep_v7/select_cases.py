"""Post-hoc selection from a frozen, already-exposed v5 run. No new validation claims."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parent

def select(run):
 rows=[json.loads(l) for l in (run/'frozen_source/fresh.jsonl').read_text().splitlines()]
 def unit(r,m,b=1024):return json.loads((run/f'results/units/correctness__{r["sample_id"]}__{m}__b{b}__rnone.json').read_text())
 losses=[r for r in rows if unit(r,'full')['probes']['task']['correct'] and not unit(r,'d30_mean16')['probes']['task']['correct']]; assert len(losses)==3
 selected=[]; used={r['sample_id'] for r in losses}; matches=[]
 for loss in sorted(losses,key=lambda r:r['sample_id']):
  candidates=[r for r in rows if r['sample_id'] not in used and r['pair_id']!=loss['pair_id'] and r['task']==loss['task'] and r['target_tokens']==loss['target_tokens'] and unit(r,'full')['probes']['task']['correct'] and unit(r,'d30_mean16')['probes']['task']['correct']]
  candidates.sort(key=lambda r:(r['gold'].get('warehouse')!=loss['gold'].get('warehouse'),r['generator_seed']!=loss['generator_seed'],r['member']!=loss['member'],abs(r['instance']-loss['instance']),r['sample_id']))
  control=candidates[0];used.add(control['sample_id']);matches.append({'loss':loss['sample_id'],'control':control['sample_id'],'same_warehouse_gold':control['gold'].get('warehouse')==loss['gold'].get('warehouse'),'same_seed':control['generator_seed']==loss['generator_seed']})
  for r,role in ((loss,'loss'),(control,'matched_correct')):
   selected.append({'row':r,'role':role,'match_loss':loss['sample_id'],'historical':{f'{m}_b{b}':unit(r,m,b) for m in ('full','d30_mean16') for b in (1024,4096)}})
 return {'status':'POST_HOC_DIAGNOSTIC','source_run':'confirmatory_v5/runs/4080-20260921-v1','selection_rule':'All b1024 Full-correct/d30_mean16-wrong cases; unique controls match task/length and both correct; prioritize same warehouse gold, same seed/member, nearest instance then sample_id. Only two eligible Amber controls exist; third control does not match warehouse gold. Selected post hoc, not independent random samples.','matches':matches,'samples':selected}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--source-run',type=Path,required=True);a=p.parse_args();s=select(a.source_run);(ROOT/'selection.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n');print(json.dumps(s['matches'],indent=2))
