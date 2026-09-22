"""Bounded, exclusive-GPU post-hoc local-recovery experiment."""
import argparse,hashlib,json,os,platform,time,traceback
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
import core
from engine import Engine
from legacy_engine import Engine as LegacyEngine
from data import encode
ROOT=Path(__file__).resolve().parent
BASE={'kind':'fixed','depth':30,'keep':16}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def same_probes(a,b):
 for name in ('task','control'):
  assert a['probes'][name]['ids']==b['probes'][name]['ids'],f'Baseline non-reproduction: {name}'
  assert a['probes'][name]['correct']==b['probes'][name]['correct']

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--hours',type=float,default=2);a=p.parse_args();assert 0<a.hours<=2
 out=a.out;out.mkdir(parents=True,exist_ok=False);(out/'units').mkdir();start=time.time();done=0;job=None
 try:
  prep=json.loads((ROOT/'prepared.json').read_text());sel=json.loads((ROOT/'selection.json').read_text());assert prep['selection_sha256']==sha(ROOT/'selection.json');assert prep['tokenizer_sha256']==sha(Path(a.model)/'tokenizer.json')
  q=prep['queue'];dump(out/'manifest.json',{'status':'POST_HOC_DIAGNOSTIC','model':a.model,'block':1024,'base_spec':BASE,'expected_units':len(q),'source_sha256':{p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md','.jsonl')},'seed':20260930})
  dump(out/'queue.json',q);assert torch.cuda.is_available();core.gpu();torch.manual_seed(20260930);torch.set_num_threads(8)
  tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();assert model.config.num_hidden_layers==36
  dump(out/'environment.json',{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'gpu':core.gpu(),'model_config_sha256':sha(Path(a.model)/'config.json'),'tokenizer_sha256':sha(Path(a.model)/'tokenizer.json'),'dtype':'float16'})
  eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);samples={x['row']['sample_id']:x for x in sel['samples']};encoded={}
  for job in q:
   if time.time()-start>a.hours*3600:raise TimeoutError('diagnostic wall-time budget exceeded')
   dump(out/'status.json',{'status':'RUNNING','completed_units':done,'expected_units':len(q),'current':job,'elapsed_seconds':time.time()-start,'pid':os.getpid()});core.gpu()
   item=samples[job['sample_id']];row=item['row'];sid=row['sample_id'];begin=time.time()
   if sid not in encoded:encoded[sid]=encode(tok,row)
   ex=encoded[sid];foil={**row['gold'],'warehouse':'Dune' if row['gold']['warehouse']!='Dune' else 'Amber'}
   if job['method']=='full':
    result=core.gate(model,tok,ex,eos,foil=foil,block=1024);same_probes(result,item['historical']['full_b1024']);result['historical_reproduction']='PASS'
   else:
    assert (out/'units'/f'{sid}__full.json').exists()
    if job['method']=='intervention':assert (out/'units'/f'{sid}__compressed.json').exists()
    spec={**BASE,'chunk_keeps':job['chunk_keeps']};result=core.execute(model,tok,ex,spec,{},eos,foil=foil,block=1024)
    expected=[job['chunk_keeps'].get(str(i),16) for i in range(len(ex['segments']['T'])//64)]
    assert [r['keep'] for r in result['routes']]==expected
    assert result['shape']['saved_T_layer_tokens']==sum(6*(64-k) for k in expected)
    if job['method']=='compressed':
     same_probes(result,item['historical']['d30_mean16_b1024']);core.Engine=LegacyEngine
     try:legacy=core.execute(model,tok,ex,BASE,{},eos,foil=foil,block=1024)
     finally:core.Engine=Engine
     same_probes(result,legacy);assert result['shape']==legacy['shape'];assert result['likelihood']==legacy['likelihood']
     result['legacy_engine_equivalence']='PASS';result['legacy_probes']=legacy['probes'];result['historical_reproduction']='PASS'
   assert result['probes']['control']['correct'],'control probe failed'
   dump(out/'units'/(job['unit_key']+'.json'),{'status':'COMPLETED','job':job,'role':item['role'],'sample_id':sid,'foil':foil,'unit_seconds':time.time()-begin,**result});done+=1
   print(json.dumps({'done':done,'total':len(q),'unit':job['unit_key'],'correct':result['probes']['task']['correct'],'answer':result['probes']['task']['answer'],'margin':result['likelihood']['log_likelihood_margin']},ensure_ascii=False),flush=True)
  dump(out/'status.json',{'status':'COMPLETED','completed_units':done,'expected_units':len(q),'elapsed_seconds':time.time()-start,'finished_unix':time.time()})
  from report import report
  report(out)
 except BaseException as exc:
  dump(out/'status.json',{'status':'FAILED','completed_units':done,'current':job,'elapsed_seconds':time.time()-start,'error':repr(exc),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
