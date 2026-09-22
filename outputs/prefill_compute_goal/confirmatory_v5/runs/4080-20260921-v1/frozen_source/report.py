import argparse,hashlib,json,random,statistics
from collections import Counter,defaultdict
from pathlib import Path
from scoring import strict_score
from specs import METHODS,BLOCKS,REPEATS,TIMING_SEEDS
ROOT=Path(__file__).resolve().parent

def dump(p,x):
 q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');q.replace(p)
def key(j):return f"{j['phase']}__{j['sample_id']}__{j['method']}__b{j['block']}__r{j.get('repeat','none')}"
def ci(v):
 v=sorted(v);return [v[int(.025*(len(v)-1))],v[int(.975*(len(v)-1))]] if v else None

def accuracy(rr,full,bootstrap=False):
 n=len(rr);base=[r for r in rr if full[r['sample']['sample_id']]['probes']['task']['correct']]
 s={'n':n,'correct':sum(r['probes']['task']['correct'] for r in rr),'native_correct_n':len(base),'retained':sum(r['probes']['task']['correct'] for r in base),
    'lost':sum(not r['probes']['task']['correct'] for r in base),'gained':sum(r['probes']['task']['correct'] and not full[r['sample']['sample_id']]['probes']['task']['correct'] for r in rr),
    'control_correct':sum(r['probes']['control']['correct'] for r in rr)}
 s['conditional_retention']=s['retained']/len(base) if base else None
 if bootstrap and n:
  pairs=defaultdict(list)
  for r in rr:pairs[r['sample']['pair_id']].append((int(r['probes']['task']['correct']),int(full[r['sample']['sample_id']]['probes']['task']['correct'])))
  assert all(len(v)==2 for v in pairs.values());groups=list(pairs.values());rng=random.Random(20260930);deltas=[];retentions=[]
  for _ in range(2000):
   draw=[x for g in rng.choices(groups,k=len(groups)) for x in g];deltas.append(sum(a-b for a,b in draw)/len(draw));den=sum(b for a,b in draw)
   if den:retentions.append(sum(a*b for a,b in draw)/den)
  s['paired_accuracy_delta_95ci']=ci(deltas);s['paired_retention_95ci']=ci(retentions)
 return s

def timing_stats(docs):
 vals=[d['reduction'] for d in docs];s={'documents':len(vals),'median_document_reduction':statistics.median(vals) if vals else None}
 if len(vals)>=2:
  rng=random.Random(20260930);s['document_bootstrap_95ci']=ci([statistics.median(rng.choices(vals,k=len(vals))) for _ in range(2000)])
 return s

def report(out,partial=True):
 out=Path(out);manifest=json.loads((out/'manifest.json').read_text());queue=json.loads((out/'queue.json').read_text());expected={key(j):j for j in queue}
 for name,h in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h
 raw=[json.loads(l) for l in (ROOT/'fresh.jsonl').read_text().splitlines()];src={r['sample_id']:r for r in raw};units=[json.loads(p.read_text()) for p in sorted((out/'units').glob('*.json'))]
 assert len({r['unit_key'] for r in units})==len(units)
 for r in units:
  assert r['unit_key'] in expected and r['job']==expected[r['unit_key']] and r['status']=='COMPLETED'
  for k,v in r['sample'].items():assert v==src[r['job']['sample_id']][k]
  if r['job']['phase']=='correctness':
   assert r['shape']['status']=='PASS' and r['shape']['projection_input_checks']==252
   for name,p in r['probes'].items():
    assert p['gold']==(src[r['job']['sample_id']]['gold'] if name=='task' else {'label':'ready-blue'})
    assert p['correct']==strict_score(p['answer'],p['gold'])['correct']
   if r['job'].get('gate'):assert len(r['gates'])==2 and all(g['ids_equal'] and g['kv_rms']<=.005 and g['logit_rms']<=.005 and g['kl']<=.001 for g in r['gates'])
  else:
   assert r['seconds']>0 and r['diagnostics_enabled'] is False and not r['before_gpu']['foreign'] and not r['after_gpu']['foreign']
 if not partial:assert len(units)==len(expected)==3200
 cr=[r for r in units if r['job']['phase']=='correctness'];names=['full',*METHODS]
 sets=[{r['job']['sample_id'] for r in cr if r['job']['block']==b and r['job']['method']==n} for b in BLOCKS for n in names];common=set.intersection(*sets);pairs=defaultdict(list)
 for sid in common:pairs[src[sid]['pair_id']].append(sid)
 common={sid for ids in pairs.values() if len(ids)==2 for sid in ids}
 summary={'audit':'PASS_completed_units_only' if partial else 'PASS','partial':partial,'completed_units':len(units),'expected_units':len(expected),
          'phase_counts':dict(Counter(r['job']['phase'] for r in units)),'common_paired_documents':len(common),'accuracy':{},'cells':{},'cross_block':{},'timing':{}}
 lookup={(r['job']['sample_id'],r['job']['method'],r['job']['block']):r for r in cr}
 for b in BLOCKS:
  full={r['job']['sample_id']:r for r in cr if r['job']['block']==b and r['job']['method']=='full'}
  for name in names:
   rr=[lookup[(sid,name,b)] for sid in sorted(common)];summary['accuracy'][f'{name}/b{b}']=accuracy(rr,full,not partial)
   for task in ('fields','narrative','code','updates'):
    for length in (4096,16384):
     cell=[r for r in rr if r['sample']['task']==task and r['sample']['target_tokens']==length];s=accuracy(cell,full);s['native_adequate']=bool(s['n'] and s['native_correct_n']/s['n']>=.75);summary['cells'][f'{name}/b{b}/{task}/{length}']=s
 for name in names:
  comparable=[(lookup[(sid,name,BLOCKS[0])],lookup[(sid,name,BLOCKS[1])]) for sid in sorted(common)]
  summary['cross_block'][name]={'documents':len(comparable),'task_ids_changed':sum(a['probes']['task']['ids']!=b['probes']['task']['ids'] for a,b in comparable),
     'task_correctness_changed':sum(a['probes']['task']['correct']!=b['probes']['task']['correct'] for a,b in comparable)}
 times=defaultdict(list)
 for r in units:
  if r['job']['phase']=='timing' and not r['job']['warmup']:times[(r['job']['sample_id'],r['job']['method'],r['job']['block'])].append(r)
 chosen=[r for r in raw if r['generator_seed'] in TIMING_SEEDS and r['instance']==0 and r['member']==0]
 valid=[r['sample_id'] for r in chosen if all(len(times[(r['sample_id'],n,b)])==REPEATS for n in names for b in BLOCKS)]
 med={(sid,n,b):statistics.median(r['seconds'] for r in times[(sid,n,b)]) for sid in valid for n in names for b in BLOCKS}
 for name in names:
  for block in BLOCKS:
   docs=[{'sample_id':sid,'task':src[sid]['task'],'length':src[sid]['target_tokens'],'median_seconds':med[(sid,name,block)],'reduction':1-med[(sid,name,block)]/med[(sid,'full',block)],
      'peak_allocated_bytes_max':max(r['peak_allocated_bytes'] for r in times[(sid,name,block)])} for sid in valid]
   summary['timing'][f'{name}/b{block}']={'per_document':docs,'all':timing_stats(docs),'by_length':{str(L):timing_stats([d for d in docs if d['length']==L]) for L in (4096,16384)}}
  docs=[]
  for sid in valid:
   base=min(med[(sid,'full',b)] for b in BLOCKS);candidate=min(med[(sid,name,b)] for b in BLOCKS)
   docs.append({'sample_id':sid,'length':src[sid]['target_tokens'],'reduction':1-candidate/base,'candidate_block':min(BLOCKS,key=lambda b:med[(sid,name,b)]),'native_block':min(BLOCKS,key=lambda b:med[(sid,'full',b)])})
  summary['timing'][f'{name}/best_of_two_blocks']={'selection_note':'Two-point tuning on these timing documents, not universally optimal settings.','per_document':docs,'all':timing_stats(docs),'by_length':{str(L):timing_stats([d for d in docs if d['length']==L]) for L in (4096,16384)}}
 dump(out/'summary.json',summary)
 lines=['# v5 frozen-candidate replication',f"Status: {'PARTIAL' if partial else 'COMPLETE'}; units {len(units)}/{len(expected)}; shared paired correctness coverage {len(common)}/256.",'',
 '| Method/block | Task correct | Full-correct lost | Conditional retention |','|---|---:|---:|---:|']
 for name,s in summary['accuracy'].items():
  retention='NA' if s['conditional_retention'] is None else f"{100*s['conditional_retention']:.2f}%";lines.append(f"| {name} | {s['correct']}/{s['n']} | {s['lost']}/{s['native_correct_n']} | {retention} |")
 lines+=['','## Clean prefill time reduction','No decode/scoring/hooks in timing; required merge/indexing/mask/position costs included.','| Method/block | Complete timing docs | 4K reduction | 16K reduction |','|---|---:|---:|---:|']
 def pct(x):return 'NA' if x is None else f'{100*x:.2f}%'
 for name,s in summary['timing'].items():lines.append(f"| {name} | {s['all']['documents']}/16 | {pct(s['by_length']['4096']['median_document_reduction'])} | {pct(s['by_length']['16384']['median_document_reduction'])} |")
 lines+=['','## Limits','New seeds, same four synthetic templates and one model/GPU. Not new-domain or architecture generalization.',
 'Native accuracy below75% makes a cell inadequate for strong preservation claims. All cells remain reported.',
 'Equal total task counts do not imply identical correct answers or losslessness. Confidence intervals and losses/gains are in summary.json.',
 'Best-of-two timing is explicitly tuning on the timing set. Cross-block output differences are reported in summary.json.',
 'A partial report is not a completed replication.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();s=report(Path(a.run),a.partial);print(json.dumps({k:s[k] for k in ('audit','completed_units','expected_units','common_paired_documents')}))
