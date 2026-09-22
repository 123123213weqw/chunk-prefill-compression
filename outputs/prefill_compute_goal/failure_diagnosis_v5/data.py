"""Fresh synthetic task families. Calibration/validation split is fixed by pair.
This is an exploratory benchmark, not evidence of general natural-language ability.
"""
import hashlib,json,random,re
TASKS=('fields','narrative','code','updates')
SEED=20260920

def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def cases(task,key,instance,member):
    rng=random.Random(int(digest(f'{SEED}:{key}:{instance}')[:16],16));a=rng.randint(11,39);b=rng.randint(41,69)
    if task=='fields':
        text=[f'Record {key}: count={a+member}.\n',f'Record {key}: limit={b}.\n']
        q=f'Read count and limit of record {key}. Return JSON with integer fields count and limit.'
        gold={'count':a+member,'limit':b}
    elif task=='narrative':
        names=['Amber','Birch','Cedar','Dune'];place=names[(instance+member)%4]
        text=[f'The parcel labelled {key} was delivered to the {place} warehouse.\n',f'The parcel labelled {key} contains {b} sealed boxes.\n']
        q=f'Where was parcel {key} delivered, and how many boxes does it contain? Return JSON with fields warehouse (name only) and boxes (integer).'
        gold={'warehouse':place,'boxes':b}
    elif task=='code':
        x=a+member
        text=[f'Python function for item {key}:\ndef f_{key}(x):\n    if x >= {a+1}:\n        return {b}\n    return {a}\n',f'For item {key}, the input x is {x}.\n']
        q=f'What integer does f_{key} return for the recorded input? Return only JSON with integer field result.'
        gold={'result':b if x>=a+1 else a}
    else:
        text=[f'Event 1 for asset {key}: its location was North.\n',f'Event 2 for asset {key}: it moved to South.\n',
              f'Event 3 for asset {key}: '+('the move to South was cancelled; restore the previous location.' if member else 'the move to South was confirmed.')+'\n']
        q=f'After all recorded events, where is asset {key}? Return only JSON with field location.'
        gold={'location':'North' if member else 'South'}
    return text,q,gold

def expected_from_text(row):
    text=row['T'];k=row['key'];task=row['task']
    if task=='fields':return {f:int(re.search(r'Record '+k+': '+f+r'=(\d+)\.',text)[1]) for f in ('count','limit')}
    if task=='narrative':return {'warehouse':re.search(r'parcel labelled '+k+r' was delivered to the (\w+) warehouse',text)[1],
                                 'boxes':int(re.search(r'parcel labelled '+k+r' contains (\d+) sealed boxes',text)[1])}
    if task=='code':
        threshold,yes,no=map(int,re.search(r'def f_'+k+r'\(x\):\n    if x >= (\d+):\n        return (\d+)\n    return (\d+)',text).groups())
        x=int(re.search(r'For item '+k+r', the input x is (\d+)',text)[1]);return {'result':yes if x>=threshold else no}
    line=re.search(r'Event 3 for asset '+k+r': ([^\n]+)',text)[1]
    return {'location':'North' if 'cancelled' in line else 'South'}

def build(tok):
    enc=lambda x:tok.encode(x,add_special_tokens=False);rows=[]
    for task in TASKS:
        for target in (4096,16384):
            for instance in range(4):
                pair=digest(f'{SEED}:{task}:{target}:{instance}')[:20];key='R'+pair[:8]
                for member in (0,1):
                    lines,q,gold=cases(task,key,instance,member)
                    fillers=[]
                    for j in range(target//8):
                        other='R'+digest(f'filler:{pair}:{j}')[:8]
                        fl,_,_=cases(task,other,j,0);fillers.extend(fl)
                    def compose(n):
                        inserts={round(n*(.15+.65*i/max(1,len(lines)-1))):line for i,line in enumerate(lines)}
                        return '<|im_start|>tool\n'+''.join(inserts.get(j,'')+fillers[j] for j in range(n))+'<|im_end|>\n'
                    lo,hi=10,len(fillers)
                    while lo+1<hi:
                        mid=(lo+hi)//2
                        if len(enc(compose(mid)))<=target-16:lo=mid
                        else:hi=mid
                    t=compose(lo)
                    # Exact requested length using deterministic neutral tail padding.
                    prefix=t[:-len('<|im_end|>\n')]
                    gap=target-len(enc(t));found=None
                    for pad in range(max(0,gap-8),gap+20):
                        text=prefix+' .' *pad+'<|im_end|>\n'
                        if len(enc(text))==target:found=text;break
                    if found is None:raise ValueError('exact token fit')
                    row={'sample_id':digest(pair+str(member))[:24],'pair_id':pair,'member':member,'instance':instance,
                         'split':'calibration' if instance<2 else 'validation','task':task,'target_tokens':target,'key':key,'T':found,
                         'question':q,'gold':gold}
                    assert expected_from_text(row)==gold;rows.append(row)
    validate(rows);return rows

def validate(rows):
    assert len(rows)==64 and len({r['sample_id'] for r in rows})==64
    expected={(t,n,i,m) for t in TASKS for n in (4096,16384) for i in range(4) for m in (0,1)}
    assert {(r['task'],r['target_tokens'],r['instance'],r['member']) for r in rows}==expected
    for r in rows:
        assert r['gold']==expected_from_text(r)
        assert r['split']==('calibration' if r['instance']<2 else 'validation')
    for pid in {r['pair_id'] for r in rows}:
        a,b=sorted([r for r in rows if r['pair_id']==pid],key=lambda r:r['member'])
        assert a['question']==b['question'] and a['gold']!=b['gold'] and a['split']==b['split']
    return {'status':'PASS','samples':64,'pairs':32,'calibration':32,'validation':32,'gold_reparsed':True,
            'note':'Counterfactual facts verified; exact evidence-position equality is NOT enforced in this new corpus.'}

def encode(tok,row):
    enc=lambda s:tok.encode(s,add_special_tokens=False)
    s='<|im_start|>system\nRead the archive carefully. Keep records separate. Answer the later question using only the archive. Output only the requested JSON. /no_think<|im_end|>\n<|im_start|>user\nRead this archive and wait for a question.<|im_end|>\n'
    t=enc(row['T']);whole=len(t)//64*64
    r=enc('<|im_start|>assistant\nArchive read. Checkpoint label: ready-blue.<|im_end|>\n')
    def q(text):return enc('<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n')
    return {'source':row,'segments':{'S':enc(s),'T':t[:whole],'R':t[whole:]+r},
            'probes':[{'name':'task','ids':q(row['question']),'gold':row['gold']},
                      {'name':'control','ids':q('What is the checkpoint label? Return JSON with field label.'),'gold':{'label':'ready-blue'}}]}

if __name__=='__main__':
    import argparse
    from pathlib import Path
    from transformers import AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument('--tokenizer',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    rows=build(AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True));out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(json.dumps(validate(rows)))
