import argparse,hashlib,json,random,statistics
from pathlib import Path
from collections import Counter,defaultdict
from scoring import strict_score
ROOT=Path(__file__).resolve().parent

def dump(p,obj):
 tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def report(out,partial=True):
 out=Path(out);manifest=json.loads((out/'manifest.json').read_text());queue=json.loads((out/'queue.json').read_text())
 for name,expected in manifest['source_sha256'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
 def key(j):return f"{j['phase']}__{j['sample_id']}__{j['method']}__{j.get('chunk','all')}"
 expected={key(j):j for j in queue};units=[json.loads(p.read_text()) for p in sorted((out/'units').glob('*.json'))]
 assert len({r['unit_key'] for r in units})==len(units)
 raw=[json.loads(l) for name in ('fresh.jsonl','diagnostic.jsonl') for l in (ROOT/name).read_text().splitlines()];src={r['sample_id']:r for r in raw}
 for r in units:
  assert r['unit_key'] in expected and r['job']==expected[r['unit_key']] and r['status']=='COMPLETED'
  assert r['shape']['status']=='PASS' and r['shape']['projection_input_checks']==36*7
  sid=r['job']['sample_id']
  for k,v in r['sample'].items():assert v==src[sid][k]
  for name,p in r['probes'].items():
   assert p['gold']==(src[sid]['gold'] if name=='task' else {'label':'ready-blue'})
   assert p['correct']==strict_score(p['answer'],p['gold'])['correct']
  if r['job'].get('gate'):
   assert len(r['gates'])==2 and all(g['ids_equal'] and g['kv_rms']<=.005 and g['logit_rms']<=.005 and g['kl']<=.001 for g in r['gates'])
 if not partial:assert len(units)==len(expected),'Incomplete queue cannot be completed'
 fresh=[r for r in units if r['job']['phase']=='fresh'];methods=['full',*manifest['methods']]
 full={r['sample']['sample_id']:r for r in fresh if r['job']['method']=='full'}
 sets=[{r['sample']['sample_id'] for r in fresh if r['job']['method']==n} for n in methods]
 common=set.intersection(*sets) if sets else set();pairs=defaultdict(list)
 for sid in common:pairs[src[sid]['pair_id']].append(sid)
 common={sid for ids in pairs.values() if len(ids)==2 for sid in ids}
 def stats(rr):
  a=[r for r in rr if full[r['sample']['sample_id']]['probes']['task']['correct']]
  return {'n':len(rr),'correct':sum(r['probes']['task']['correct'] for r in rr),'native_correct_n':len(a),
   'lost':sum(not r['probes']['task']['correct'] for r in a),'gained':sum(r['probes']['task']['correct'] and not full[r['sample']['sample_id']]['probes']['task']['correct'] for r in rr),
   'control_correct':sum(r['probes']['control']['correct'] for r in rr)}
 summary={'audit':'PASS_completed_units_only' if partial else 'PASS','partial':partial,'completed_units':len(units),'expected_units':len(expected),
  'phase_counts':dict(Counter(r['job']['phase'] for r in units)),'fresh_native_documents':len(full),'fresh_native_correct':sum(r['probes']['task']['correct'] for r in full.values()),
  'fresh_common_paired_documents':len(common),'fresh':{},'cells':{},'diagnostics':{}}
 for name in methods:
  rr=[r for r in fresh if r['job']['method']==name and r['sample']['sample_id'] in common];s=stats(rr)
  if rr and not partial:
   grouped=defaultdict(list)
   for r in rr:grouped[r['sample']['pair_id']].append(int(r['probes']['task']['correct'])-int(full[r['sample']['sample_id']]['probes']['task']['correct']))
   v=[statistics.mean(x) for x in grouped.values()];rng=random.Random(20260925);bs=sorted(statistics.mean(rng.choices(v,k=len(v))) for _ in range(2000));s['paired_delta_95ci']=[bs[49],bs[1949]]
  summary['fresh'][name]=s
  for task,length in sorted({(r['task'],r['target_tokens']) for r in raw}):
   cell=[r for r in rr if r['sample']['task']==task and r['sample']['target_tokens']==length];c=stats(cell);c['native_adequate']=c['n']>0 and c['native_correct_n']/c['n']>=.75;summary['cells'][f'{name}/{task}/{length}']=c
 baselines={(r['sample']['sample_id'],r['job']['method']):r for r in units if r['job']['phase']=='diagnostic_baseline'}
 effects=[]
 for r in units:
  if r['job']['phase']!='intervention':continue
  j=r['job'];sid=j['sample_id'];base=baselines[(sid,'full' if 'only_chunk' in j['spec'] else j['base_method'])]
  teacher=baselines[(sid,'full')]
  # Teacher features are ordered by logical chunk_start; chunk index is relative to T.
  fs=[f for f in teacher.get('teacher_features',[]) if f['depth']==j['spec']['depth'] and f['keep']==j['spec']['keep']];fs=sorted(fs,key=lambda f:f['chunk_start'])
  effects.append({'sample_id':sid,'method':j['method'],'chunk':j['chunk'],'before_correct':base['probes']['task']['correct'],'after_correct':r['probes']['task']['correct'],
   'margin_delta':r['likelihood']['log_likelihood_margin']-base['likelihood']['log_likelihood_margin'],
   'teacher_reconstruction_error':fs[j['chunk']]['score'] if len(fs)>j['chunk'] else None})
 for name in sorted({e['method'] for e in effects}):
  es=[e for e in effects if e['method']==name];summary['diagnostics'][name]={'completed_interventions':len(es),'documents':len({e['sample_id'] for e in es}),
    'correct_to_wrong':sum(e['before_correct'] and not e['after_correct'] for e in es),'wrong_to_correct':sum(not e['before_correct'] and e['after_correct'] for e in es),
    'largest_absolute_margin_changes':sorted(es,key=lambda e:abs(e['margin_delta']),reverse=True)[:12]}
 dump(out/'summary.json',summary);dump(out/'intervention_effects.json',effects)
 lines=['# Long validation v4',f"Status: {'PARTIAL' if partial else 'COMPLETE'}. Units: {len(units)}/{len(expected)}.",
  f"Fresh shared paired coverage: {len(common)}/256 documents. Diagnostic set is previously exposed v3 data.",'',
  '| Method | Correct | Full-correct lost | Control |','|---|---:|---:|---:|']
 for name,v in summary['fresh'].items():lines.append(f"| {name} | {v['correct']}/{v['n']} | {v['lost']}/{v['native_correct_n']} | {v['control_correct']}/{v['n']} |")
 lines+=['','## Interpretation limits','Fresh seeds use the same synthetic templates: not new domains or cross-architecture generalization.',
  'Native baseline <75% makes that cell inadequate for a compression-robustness claim. See summary.json.',
  'Intervention likelihood compares exact canonical JSON sequences, not semantic correctness probabilities.',
  'Runtime is instrumented diagnostic cost, not production prefill speed. Single-chunk effects can interact.',
  'A partial report is not a completed 24-hour validation. Thresholds and candidate rules remain frozen.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--partial',action='store_true');a=p.parse_args();r=report(Path(a.run),partial=a.partial);print(json.dumps({k:r[k] for k in ('audit','completed_units','expected_units','fresh_common_paired_documents')}))
