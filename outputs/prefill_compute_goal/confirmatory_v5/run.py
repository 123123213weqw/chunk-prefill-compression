import argparse,hashlib,json,os,platform,random,time,traceback
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from data import encode,expected_from_text
from specs import METHODS,BLOCKS,SEEDS,TIMING_SEEDS,WARMUPS,REPEATS
from core import gate,execute,gpu
from clean import timed
ROOT=Path(__file__).resolve().parent

def dump(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def unitkey(j):return f"{j['phase']}__{j['sample_id']}__{j['method']}__b{j['block']}__r{j.get('repeat','none')}"
def jobs(rows):
 pairs=sorted({r['pair_id'] for r in rows});random.Random(20260930).shuffle(pairs)
 for pid in pairs:
  for r in sorted([x for x in rows if x['pair_id']==pid],key=lambda r:r['member']):
   blocks=list(BLOCKS);random.Random(int(r['sample_id'][:6],16)).shuffle(blocks)
   for block in blocks:
    common={'phase':'correctness','sample_id':r['sample_id'],'block':block}
    yield {**common,'method':'full','gate':True}
    names=list(METHODS);random.Random(block+int(r['sample_id'][:6],16)).shuffle(names)
    for n in names:yield {**common,'method':n,'spec':METHODS[n]}
 chosen=[r for r in rows if r['generator_seed'] in TIMING_SEEDS and r['instance']==0 and r['member']==0];assert len(chosen)==16
 for r in sorted(chosen,key=lambda r:r['sample_id']):
  for repeat in range(-WARMUPS,REPEATS):
   combinations=[(n,b) for n in ('full',*METHODS) for b in BLOCKS];random.Random(20260930+repeat+int(r['sample_id'][:6],16)).shuffle(combinations)
   for name,block in combinations:yield {'phase':'timing','sample_id':r['sample_id'],'method':name,'block':block,'repeat':repeat,'warmup':repeat<0}

def summarize(out,partial=True):
 from report import report
 return report(out,partial)

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--hours',type=float,default=12);a.add_argument('--model',default='<MODEL_ROOT>/Qwen3-4B-Instruct-2507');a.add_argument('--resume',action='store_true');args=a.parse_args();assert 0<args.hours<=24
 out=Path(args.out);started=time.time();deadline=started+args.hours*3600;done=0;active=None
 if args.resume:assert (out/'manifest.json').exists()
 else:out.mkdir(parents=True,exist_ok=False);(out/'units').mkdir()
 try:
  rows=[json.loads(l) for l in (ROOT/'fresh.jsonl').read_text().splitlines()];assert len(rows)==256 and len({r['sample_id'] for r in rows})==256
  for r in rows:assert r['gold']==expected_from_text(r) and r['generator_seed'] in SEEDS
  source={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.jsonl')};queue=list(jobs(rows));assert len(queue)==3200 and len({unitkey(j) for j in queue})==3200
  config={'source_sha256':source,'model':args.model,'methods':METHODS,'blocks':list(BLOCKS),'expected_units':len(queue),'samples':256,'timing_samples':16,'warmups':WARMUPS,'repeats':REPEATS}
  if args.resume:
   old=json.loads((out/'manifest.json').read_text());assert old==config,'resume mismatch'
  else:dump(out/'manifest.json',config);dump(out/'queue.json',queue)
  assert torch.cuda.is_available();gpu();torch.manual_seed(20260930);torch.set_num_threads(8)
  tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();assert model.config.num_hidden_layers==36
  dump(out/'environment.json',{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'gpu':gpu(),'model_config_sha256':sha(Path(args.model)/'config.json'),'tokenizer_sha256':sha(Path(args.model)/'tokenizer.json'),'dtype':'float16'})
  eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);source_rows={r['sample_id']:r for r in rows};encoded={}
  dump(out/'status.json',{'status':'RUNNING','pid':os.getpid(),'deadline_unix':deadline,'expected_units':len(queue)})
  for job in queue:
   active=job;key=unitkey(job);dest=out/'units'/(key+'.json')
   if dest.exists():
    prev=json.loads(dest.read_text());assert prev['unit_key']==key and prev['status']=='COMPLETED';done+=1;continue
   if time.time()>=deadline:
    summarize(out,True);status={'status':'BUDGET_EXHAUSTED','completed_units':done,'expected_units':len(queue),'next_unit':job,'elapsed_seconds':time.time()-started};dump(out/'status.json',status);dump(out/'progress.json',status);return
   dump(out/'progress.json',{'status':'RUNNING','completed_units':done,'expected_units':len(queue),'current':job,'deadline_unix':deadline,'elapsed_seconds':time.time()-started})
   sid=job['sample_id'];src=source_rows[sid]
   if sid not in encoded:
    encoded[sid]=encode(tok,src);assert len(encoded[sid]['segments']['T'])==src['target_tokens']
   ex=encoded[sid];begin=time.time()
   if job['phase']=='timing':result=timed(model,ex['segments'],METHODS.get(job['method']),job['block'])
   elif job.get('gate'):result=gate(model,tok,ex,eos,block=job['block'])
   else:
    baseline=out/'units'/(unitkey({'phase':'correctness','sample_id':sid,'method':'full','block':job['block']})+'.json');assert baseline.exists()
    result=execute(model,tok,ex,job['spec'],{},eos,block=job['block'])
   dump(dest,{'unit_key':key,'status':'COMPLETED','job':job,'sample':{k:src[k] for k in ('sample_id','pair_id','member','task','target_tokens','generator_seed','instance')},'unit_seconds':time.time()-begin,**result});done+=1
   print(json.dumps({'done':done,'total':len(queue),'unit':key,'correct':result.get('probes',{}).get('task',{}).get('correct'),'seconds':result.get('seconds',round(time.time()-begin,3))}),flush=True)
   if done%128==0:summarize(out,True)
  summarize(out,False);status={'status':'COMPLETED','completed_units':done,'expected_units':len(queue),'elapsed_seconds':time.time()-started,'finished_unix':time.time()};dump(out/'status.json',status);dump(out/'progress.json',status)
 except BaseException as exc:
  status={'status':'INTERRUPTED' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'FAILED','completed_units':done,'current':active,'error':repr(exc),'traceback':traceback.format_exc()};dump(out/'status.json',status);dump(out/'progress.json',status);raise
if __name__=='__main__':main()
