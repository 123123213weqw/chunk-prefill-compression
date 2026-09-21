import argparse,hashlib,json,os,platform,random,sys,time,traceback
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from data import encode,expected_from_text
from specs import METHODS,BASES
from core import gate,execute,gpu
ROOT=Path(__file__).resolve().parent

def dump(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(l) for l in p.read_text().splitlines()]
def unitkey(job):return f"{job['phase']}__{job['sample_id']}__{job['method']}__{job.get('chunk','all')}"

def jobs(fresh,diagnostic):
 # Pair-wise deterministic shuffle; each document's identity gate precedes its lossy methods.
 pairs=sorted({r['pair_id'] for r in fresh});random.Random(20260925).shuffle(pairs)
 for pid in pairs:
  for r in sorted([r for r in fresh if r['pair_id']==pid],key=lambda r:r['member']):
   common={'phase':'fresh','sample_id':r['sample_id']}
   yield {**common,'method':'full','gate':True}
   order=list(METHODS);random.Random(int(r['sample_id'][:8],16)).shuffle(order)
   for n in order:yield {**common,'method':n,'spec':METHODS[n]}
 for r in diagnostic:
  common={'phase':'diagnostic_baseline','sample_id':r['sample_id']}
  yield {**common,'method':'full','gate':True}
  for n,spec in BASES.items():yield {**common,'method':n,'spec':spec}
 # Round robin across document, depth and intervention; deterministic ordering.
 for chunk in range(max(r['target_tokens']//64 for r in diagnostic)):
  for r in diagnostic:
   if chunk>=r['target_tokens']//64:continue
   for name,spec in BASES.items():
    for intervention in ('only_chunk','restore_chunk'):
     yield {'phase':'intervention','sample_id':r['sample_id'],'method':name+'_'+intervention,'base_method':name,'chunk':chunk,'spec':{**spec,intervention:chunk}}

def summarize(out,partial=True):
 from report import report
 report(out,partial=partial)

def main():
 a=argparse.ArgumentParser();a.add_argument('--out',required=True);a.add_argument('--hours',type=float,default=24);a.add_argument('--model',default='<MODEL_ROOT>/Qwen3-4B-Instruct-2507');a.add_argument('--resume',action='store_true');args=a.parse_args()
 out=Path(args.out);started=time.time();deadline=started+args.hours*3600
 if not 0<args.hours<=24:raise ValueError('0 < hours <=24 required')
 if args.resume:assert (out/'manifest.json').exists()
 else:out.mkdir(parents=True,exist_ok=False);(out/'units').mkdir()
 done=0;active=None
 try:
  files={p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.jsonl','.md')}
  fresh=read(ROOT/'fresh.jsonl');diag=read(ROOT/'diagnostic.jsonl');allrows={r['sample_id']:r for r in fresh+diag}
  assert len(fresh)==256 and len(allrows)==len(fresh)+len(diag)
  for r in allrows.values():assert expected_from_text(r)==r['gold']
  queue=list(jobs(fresh,diag));assert len({unitkey(j) for j in queue})==len(queue)
  config={'source_sha256':files,'model':args.model,'methods':METHODS,'expected_units':len(queue),'fresh_documents':len(fresh),'diagnostic_documents':len(diag),'hours_per_invocation':args.hours}
  if args.resume:
   old=json.loads((out/'manifest.json').read_text())
   for k in ('source_sha256','model','methods','expected_units'):assert old[k]==config[k],('resume mismatch',k)
  else:dump(out/'manifest.json',config);dump(out/'queue.json',queue)
  assert torch.cuda.is_available();gpu();torch.manual_seed(20260925);torch.set_num_threads(8)
  tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
  model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
  assert model.config.num_hidden_layers==36
  dump(out/'environment.json',{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'gpu':gpu(),
    'model_config_sha256':sha(Path(args.model)/'config.json'),'tokenizer_sha256':sha(Path(args.model)/'tokenizer.json'),'dtype':'float16','block':1024})
  eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);thresholds=json.loads((ROOT/'thresholds.json').read_text())['values'];foils=json.loads((ROOT/'counterfactuals.json').read_text())
  dump(out/'status.json',{'status':'RUNNING','pid':os.getpid(),'deadline_unix':deadline,'expected_units':len(queue)})
  # Tokenize lazily and cache CPU inputs; no validation outputs enter a router.
  encoded={};resumed=0
  for idx,job in enumerate(queue):
   key=unitkey(job);dest=out/'units'/(key+'.json');active=job
   if dest.exists():
    prev=json.loads(dest.read_text());assert prev['unit_key']==key and prev['status']=='COMPLETED';done+=1;resumed+=1;continue
   if time.time()>=deadline:
    summarize(out,partial=True);dump(out/'status.json',{'status':'BUDGET_EXHAUSTED','completed_units':done,'expected_units':len(queue),'next_unit':job,'elapsed_seconds':time.time()-started});return
   dump(out/'progress.json',{'status':'RUNNING','completed_units':done,'expected_units':len(queue),'current':job,'deadline_unix':deadline,'elapsed_seconds':time.time()-started,'resumed_units':resumed})
   sid=job['sample_id'];src=allrows[sid]
   if sid not in encoded:
    encoded[sid]=encode(tok,src);assert len(encoded[sid]['segments']['T'])==src['target_tokens']
   ex=encoded[sid];foil=None if job['phase']=='fresh' else foils[sid];begin=time.time()
   if job.get('gate'):result=gate(model,tok,ex,eos,foil)
   else:
    baseline_phase='fresh' if job['phase']=='fresh' else 'diagnostic_baseline'
    baseline=out/'units'/(unitkey({'phase':baseline_phase,'sample_id':sid,'method':'full'})+'.json');assert baseline.exists(),'identity gate required'
    result=execute(model,tok,ex,job['spec'],thresholds,eos,foil)
   record={'unit_key':key,'status':'COMPLETED','job':job,'sample':{k:src[k] for k in ('sample_id','pair_id','member','task','target_tokens','split')},'unit_seconds':time.time()-begin,**result}
   dump(dest,record);done+=1
   print(json.dumps({'done':done,'total':len(queue),'unit':key,'correct':result['probes']['task']['correct'],'seconds':round(record['unit_seconds'],2)}),flush=True)
   if done%256==0:summarize(out,partial=True)
  summarize(out,partial=False);dump(out/'status.json',{'status':'COMPLETED','completed_units':done,'expected_units':len(queue),'elapsed_seconds':time.time()-started,'finished_unix':time.time()})
 except BaseException as exc:
  dump(out/'status.json',{'status':'INTERRUPTED' if isinstance(exc,(InterruptedError,KeyboardInterrupt)) else 'FAILED','completed_units':done,'current':active,'error':repr(exc),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
