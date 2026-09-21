"""Rebuild summary from raw records; reject incomplete or corrupted coverage."""
import argparse,hashlib,json,random,statistics
from collections import defaultdict,Counter
from pathlib import Path
from data import validate

def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def auc(scores,loss):
    a=[s for s,l in zip(scores,loss) if l];b=[s for s,l in zip(scores,loss) if not l]
    return None if not a or not b else sum((x>y)+.5*(x==y) for x in a for y in b)/(len(a)*len(b))
def quantile(v,p):
    v=sorted(v);i=(len(v)-1)*p;j=int(i);return v[j]+(v[min(j+1,len(v)-1)]-v[j])*(i-j)
def paired_ci(rows,full):
    pairs=defaultdict(list)
    for r in rows:pairs[r['pair_id']].append(int(r['correct'])-int(full[r['sample_id']]['correct']))
    assert all(len(v)==2 for v in pairs.values());v=[statistics.mean(x) for x in pairs.values()];rng=random.Random(20260920)
    samples=sorted(statistics.mean(rng.choices(v,k=len(v))) for _ in range(2000))
    return [quantile(samples,.025),quantile(samples,.975)]

def report(out):
    m=json.loads((out/'manifest.json').read_text());raw=read(out/'dataset.jsonl');validate(raw);src={r['sample_id']:r for r in raw}
    assert sha(out/'dataset.jsonl')==m['dataset_sha256']
    for name,h in m['source_sha256'].items():assert sha(out/'source'/name)==h
    rows=read(out/'correctness.jsonl');gates=read(out/'gates.jsonl');shapes=read(out/'shapes.jsonl');timings=read(out/'timings.jsonl');features=read(out/'features.jsonl')
    expected={(r['sample_id'],method,p) for r in raw for method in m[r['split']+'_methods'] for p in ('task','control')}
    keys=[(r['sample_id'],r['method'],r['probe']) for r in rows];assert len(keys)==len(set(keys)) and set(keys)==expected
    for r in rows:
        s=src[r['sample_id']]
        for k in ('pair_id','member','instance','split','task','target_tokens'):assert r[k]==s[k]
        from scoring import strict_score
        assert strict_score(r['answer'],r['gold'])['correct']==r['correct']
        assert r['gold']==(s['gold'] if r['probe']=='task' else {'label':'ready-blue'})
    assert len(gates)==128 and {(g['sample_id'],g['probe']) for g in gates}=={(s,p) for s in src for p in ('task','control')}
    assert all(g['ids_equal'] and g['kv_rms']<=.005 and g['logit_rms']<=.005 and g['kl']<=.001 for g in gates)
    threshold=json.loads((out/'thresholds.json').read_text());calids={r['sample_id'] for r in raw if r['split']=='calibration'}
    assert set(threshold['sample_ids'])==calids and threshold['uses_answers'] is False
    fcal=[r for r in features if r['sample_id'] in calids];early=m['layers']//2;late=3*m['layers']//4
    for key,depth,k,q in [('early_median',early,48,.5),('early_q25',early,48,.25),('late_q25',late,32,.25)]:
        assert abs(threshold['values'][key]-quantile([f['score'] for f in fcal if f['depth']==depth and f['keep']==k],q))<1e-12
    # Threshold file committed to the exact calibration-only JSONL prefix.
    prefix=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in fcal).encode()
    assert hashlib.sha256(prefix).hexdigest()==threshold['calibration_features_sha256']
    expected_shapes={(sid,method) for sid,method,p in expected if p=='task' and method not in ('full','identity')}
    sk=[(s['sample_id'],s['method']) for s in shapes];assert len(sk)==len(set(sk)) and set(sk)==expected_shapes
    for s in shapes:
        seg=m['token_lengths'][s['sample_id']];routes=s['routes'];L=m['layers'];name=s['method']
        assert sorted(r['chunk_start'] for r in routes)==list(range(seg['S'],seg['S']+seg['T'],64))
        for r in routes:
            if name in m['fixed']:
                spec=m['fixed'][name];assert (r['merge_depth'],r['keep'])==(spec['depth'],spec['keep'])
            elif name in ('adaptive_equal','random_equal'):assert (r['merge_depth'],r['keep']) in ((early,48),(late,32))
            else:assert (r['merge_depth'],r['keep']) in ((early,48),(late,32),(L,64))
        lengths=[seg['S']+seg['R']+sum(64 if i<r['merge_depth'] else r['keep'] for r in routes) for i in range(L)]
        assert lengths==s['expected_layer_tokens'] and len(s['actual_linear_inputs'])==7*L
        assert all(r['tokens']==lengths[r['layer']] for r in s['actual_linear_inputs'])
        if name in (*m['primary'],'adaptive_equal','random_equal'):assert s['saved_T_layer_tokens']==seg['T']*L//8
    tnames=['native_b1024','native_b4096',*m['primary'],'adaptive_equal','random_equal','adaptive_safe']
    tsids={s['sample_id'] for s in raw if s['instance']==2 and s['member']==0}
    tk=[(r['sample_id'],r['method'],r['repeat']) for r in timings]
    assert len(tk)==len(set(tk)) and set(tk)=={(sid,n,r) for sid in tsids for n in tnames for r in range(-1,5)}
    assert all(not r['before']['foreign'] and not r['after']['foreign'] and r['seconds']>0 for r in timings)
    summary={'audit':'PASS','identity_gates':len(gates),'correctness_records':len(rows),'shape_records':len(shapes),'timing_trials':len(timings),'validation':{},'by_cell':{},'timings':{},'indicator':{}}
    full={r['sample_id']:r for r in rows if r['method']=='full' and r['probe']=='task' and r['split']=='validation'}
    def stats(rr):
        base=[r for r in rr if full[r['sample_id']]['correct']]
        ps=defaultdict(list)
        for r in rr:ps[r['pair_id']].append(r['correct'])
        return {'n':len(rr),'correct':sum(r['correct'] for r in rr),'full_correct_n':len(base),'retained':sum(r['correct'] for r in base),
                'lost':sum(not r['correct'] for r in base),'gained':sum(r['correct'] and not full[r['sample_id']]['correct'] for r in rr),
                'both_pair_members_correct':sum(all(v) for v in ps.values()),'pairs':len(ps),'paired_accuracy_delta_95ci':paired_ci(rr,full)}
    for method in m['validation_methods']:
        rr=[r for r in rows if r['split']=='validation' and r['method']==method and r['probe']=='task']
        summary['validation'][method]=stats(rr)
        summary['validation'][method]['control_correct']=sum(r['correct'] for r in rows if r['split']=='validation' and r['method']==method and r['probe']=='control')
        for task in ('fields','narrative','code','updates'):
            for length in (4096,16384):
                cell=[r for r in rr if r['task']==task and r['target_tokens']==length];v=stats(cell)
                v['native_cell_adequate']=sum(full[r['sample_id']]['correct'] for r in cell)>=.75*len(cell)
                summary['by_cell'][f'{method}/{task}/{length}']=v
    ft=defaultdict(list)
    for f in features:ft[(f['sample_id'],f['depth'],f['keep'])].append(f['score'])
    for method,spec in m['fixed'].items():
        if spec.get('phase')!='ceil' or spec['keep'] not in (32,48):continue
        rr=[r for r in rows if r['split']=='validation' and r['probe']=='task' and r['method']==method and full[r['sample_id']]['correct']]
        scores=[ft[(r['sample_id'],spec['depth'],spec['keep'])] for r in rr];loss=[not r['correct'] for r in rr]
        summary['indicator'][method]={'n_full_correct':len(rr),'losses':sum(loss),'mean_score_loss_auc':auc([statistics.mean(v) for v in scores],loss),
                                      'p95_score_loss_auc':auc([quantile(v,.95) for v in scores],loss),
                                      'caution':'Document-level association; task/length can confound. No per-chunk causal identification.'}
    times=defaultdict(list)
    for r in timings:
        if not r['warmup']:times[(r['sample_id'],r['method'])].append(r['seconds'])
    for name in tnames:
        summary['timings'][name]={'per_document':[{'sample_id':sid,'task':src[sid]['task'],'length':src[sid]['target_tokens'],
             'median_seconds':statistics.median(times[(sid,name)]),'reduction_vs_full_b1024':1-statistics.median(times[(sid,name)])/statistics.median(times[(sid,'native_b1024')])} for sid in sorted(tsids)]}
        summary['timings'][name]['median_document_reduction']=statistics.median(x['reduction_vs_full_b1024'] for x in summary['timings'][name]['per_document'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Compression depth v3: '+Path(m['model']).name,'','Exploratory synthetic validation: 32 documents / 16 pairs. Same-family results do not establish generality.','',
           '| Method | Exact task | Lost / Full-correct | Control |','|---|---:|---:|---:|']
    for name,v in summary['validation'].items():lines.append(f"| {name} | {v['correct']}/{v['n']} | {v['lost']}/{v['full_correct_n']} | {v['control_correct']}/32 |")
    lines+=['','## Prefill timings','Includes router and merge overhead; excludes loading/tokenization and decode.','', '| Method | Median per-document time reduction vs native b1024 |','|---|---:|']
    for name,v in summary['timings'].items():lines.append(f"| {name} | {100*v['median_document_reduction']:.2f}% |")
    lines+=['','## Limits','Only linear projection/MLP work is exactly budget-matched, not total FLOPs or wall time.','Full accuracy below 75% in a task/length cell invalidates robustness claims there; see summary.json.','Indicator AUROC=null means losses or retained cases are absent, not a successful predictor.','No method was tuned on validation. This small study cannot establish robust universal compression.','',f"Audit PASS: {len(gates)} identity gates, {len(shapes)} shape audits, {len(timings)} timing trials."]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:summary[k] for k in ('audit','identity_gates','correctness_records','shape_records','timing_trials')}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();report(Path(a.run))
