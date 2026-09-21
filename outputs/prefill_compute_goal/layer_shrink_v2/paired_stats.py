"""Pair-cluster, stratum-preserving bootstrap; no member/trial pseudo-replication."""
import collections,random,statistics

def quantile(xs,q):
    xs=sorted(xs);pos=(len(xs)-1)*q;i=int(pos);j=min(i+1,len(xs)-1)
    return xs[i]+(xs[j]-xs[i])*(pos-i)

def comparison(rows,method,repeats=2000,seed=20260919):
    lookup={(r['sample_id'],r['method']):r for r in rows if r['probe']=='readout'}
    ids=sorted({r['sample_id'] for r in rows if r['probe']=='readout'})
    paired=collections.defaultdict(list);harm=benefit=0;full_correct=0;retained=0
    for sid in ids:
        a=lookup[(sid,'native_full')];b=lookup[(sid,method)]
        # OOM/failure is included as incorrect in all-sample deployment score,
        # while answer-only rates and unavailable counts are reported separately.
        ac=bool(a.get('correct',False));bc=bool(b.get('correct',False))
        harm+=ac and not bc;benefit+=not ac and bc;full_correct+=ac;retained+=ac and bc
        paired[a['pair_id']].append((sid,float(bc)-float(ac),(a['template'],a['length'],a['evidence_layout'])))
    strata=collections.defaultdict(list)
    for pair,items in paired.items():
        assert len(items)==2 and items[0][2]==items[1][2]
        strata[items[0][2]].append(statistics.mean(x[1] for x in items))
    rng=random.Random(seed);draws=[]
    for _ in range(repeats):
        draw=[]
        for key in sorted(strata):
            values=strata[key];draw.extend(rng.choices(values,k=len(values)))
        draws.append(statistics.mean(draw))
    return {'n':len(ids),'pairs':len(paired),'strata':len(strata),'method_minus_full':statistics.mean(x[1] for v in paired.values() for x in v),
            'ci95':[quantile(draws,.025),quantile(draws,.975)],'bootstrap_repeats':repeats,'bootstrap_seed':seed,
            'full_correct_to_method_wrong':harm,'full_wrong_to_method_correct':benefit,
            'full_correct_n':full_correct,'full_correct_retained':retained,
            'full_correct_retention':retained/full_correct if full_correct else None,
            'unavailable_full':sum(lookup[(sid,'native_full')]['status']!='OK' for sid in ids),
            'unavailable_method':sum(lookup[(sid,method)]['status']!='OK' for sid in ids)}

def strata_scores(rows):
    result=[]
    keys=sorted({(r['method'],r['length'],r['template'],r['evidence_layout'],r['probe']) for r in rows})
    for m,n,t,layout,probe in keys:
        selected=[r for r in rows if (r['method'],r['length'],r['template'],r['evidence_layout'],r['probe'])==(m,n,t,layout,probe)]
        result.append({'method':m,'length':n,'template':t,'layout':layout,'probe':probe,'n':len(selected),
                       'correct':sum(bool(r.get('correct',False)) for r in selected),'unavailable':sum(r['status']!='OK' for r in selected)})
    return result
