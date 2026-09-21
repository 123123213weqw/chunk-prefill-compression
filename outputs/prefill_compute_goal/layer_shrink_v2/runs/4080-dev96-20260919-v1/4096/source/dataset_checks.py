"""Validate development-only matrix from actual text, without opening test data."""
from collections import Counter
from data_reference import FIELDS,compute,reparse,evidence_spans,ids_hash

def validate_development(rows,enc=None):
    expected={(t,n,layout,i,m) for t in FIELDS for n in (4096,8192,16384)
              for layout in ('clustered','distributed') for i in range(4) for m in (0,1)}
    observed=[(r['template'],r['length'],r['evidence_layout'],r['instance'],r['member']) for r in rows]
    assert len(rows)==96 and set(observed)==expected and len(set(observed))==96
    assert len({r['sample_id'] for r in rows})==96
    pairs={}
    for r in rows:
        assert r['split']=='development';pairs.setdefault(r['pair_id'],[]).append(r)
        parsed=reparse(r);assert set(parsed)==set(FIELDS[r['template']])
        probes={p['probe_id']:p for p in r['probes']}
        assert set(probes)=={'readout','calculation','control'}
        assert parsed==probes['readout']['gold']
        assert {'result':compute(r['template'],parsed)}==probes['calculation']['gold']
        assert r['target_job'] not in r['segments']['S']['text']+r['segments']['R']['text']
        assert r['segments']['T']['token_count']==r['length']
        if enc is not None:
            for seg in [*r['segments'].values(),r['short_T'],*[p['Q'] for p in r['probes']]]:
                ids=enc(seg['text'])
                assert len(ids)==seg['token_count'] and ids_hash(ids)==seg['token_id_sha256']
            assert evidence_spans(enc,r['segments']['T']['text'],r['evidence'])==r['evidence_spans']
        pos=sorted(e['token_span'][0] for e in r['evidence_spans'])
        if r['evidence_layout']=='distributed':assert min(b-a for a,b in zip(pos,pos[1:]))>=.15*r['length']
    assert len(pairs)==48
    for pair in pairs.values():
        assert len(pair)==2 and sorted(r['member'] for r in pair)==[0,1]
        a,b=sorted(pair,key=lambda r:r['member'])
        for key in ('template','length','evidence_layout','instance','target_job'):assert a[key]==b[key]
        assert a['segments']['S']==b['segments']['S'] and a['segments']['R']==b['segments']['R']
        assert [p['Q'] for p in a['probes']]==[p['Q'] for p in b['probes']]
        aa=a['segments']['T']['text'].splitlines();bb=b['segments']['T']['text'].splitlines()
        assert len(aa)==len(bb) and sum(x!=y for x,y in zip(aa,bb))==1
        assert [p['token_span'] for p in a['evidence_spans']]==[p['token_span'] for p in b['evidence_spans']]
        ga=reparse(a);gb=reparse(b);assert sum(ga[k]!=gb[k] for k in ga)==1
        assert ga[a['changed_field']]!=gb[a['changed_field']] and compute(a['template'],ga)!=compute(b['template'],gb)
    return {'status':'PASS','samples':96,'pairs':48,'tokenizer_revalidated':enc is not None,
            'gold_reparsed':True,'single_field_interventions_verified':True}
