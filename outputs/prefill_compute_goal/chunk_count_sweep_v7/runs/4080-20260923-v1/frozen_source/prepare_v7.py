import hashlib,json,random
from pathlib import Path
ROOT=Path(__file__).resolve().parent
COUNTS=(0,16,32,64,96,128,160,192,224,240,256)
ORDERS=('prefix','suffix','random_A','random_B')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def order(sid,name):
 v=list(range(256))
 if name=='suffix':v.reverse()
 elif name.startswith('random_'):
  label=name.rsplit('_',1)[1];seed=int(hashlib.sha256(f'v7:{label}:{sid}'.encode()).hexdigest()[:16],16);random.Random(seed).shuffle(v)
 else:assert name=='prefix'
 return v
def make():
 s=json.loads((ROOT/'selection.json').read_text());assert len(s['samples'])==6;queue=[];orders={}
 for item in s['samples']:
  sid=item['row']['sample_id'];assert item['row']['target_tokens']==16384
  orders[sid]={name:order(sid,name) for name in ORDERS}
  for method,count in (('full',0),('all_compressed',256)):
   queue.append({'sample_id':sid,'method':method,'count':count,'unit_key':sid+'__'+method})
  for name in ORDERS:
   for count in COUNTS[1:-1]:
    subset=orders[sid][name][:count]
    queue.append({'sample_id':sid,'method':'sweep','order':name,'count':count,'chunk_indices':subset,'unit_key':f'{sid}__{name}__n{count}'})
 assert len(queue)==228 and len({q['unit_key'] for q in queue})==228
 return {'status':'READY_GPU_NOT_RUN','selection_sha256':sha(ROOT/'selection.json'),'parent_prepared_sha256':sha(ROOT/'prepared.json'),'parent_v6_summary_sha256':sha(ROOT/'PARENT_V6_SUMMARY.json'),'counts':COUNTS,'order_names':ORDERS,'orders':orders,'queue':queue,'expected_units':228}
if __name__=='__main__':
 obj=make();(ROOT/'queue_v7.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');print(json.dumps({'expected_units':228,'counts':COUNTS,'orders':ORDERS}))
