"""Bounded exclusive-GPU answer-blind chunk-count sweep on exposed v5 cases."""
import argparse,hashlib,json,os,platform,time,traceback
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
import core
from engine import Engine
from legacy_engine import Engine as Legacy
from data import encode
from prepare_v7 import make
ROOT=Path(__file__).resolve().parent;BASE={'kind':'fixed','depth':30,'keep':16}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def same_probes(a,b):
 for n in ('task','control'):
  assert a['probes'][n]['ids']==b['probes'][n]['ids'],f'Non-reproducing {n}'
  assert a['probes'][n]['correct']==b['probes'][n]['correct']
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--hours',type=float,default=2);a=p.parse_args();assert 0<a.hours<=2
 out=a.out;out.mkdir(parents=True,exist_ok=False);(out/'units').mkdir();started=time.time();done=0;job=None
 try:
  prepared=json.loads((ROOT/'queue_v7.json').read_text());assert prepared['queue']==make()['queue']
  assert prepared['selection_sha256']==sha(ROOT/'selection.json') and prepared['parent_prepared_sha256']==sha(ROOT/'prepared.json') and prepared['parent_v6_summary_sha256']==sha(ROOT/'PARENT_V6_SUMMARY.json')
  assert len(prepared['queue'])==228
  dump(out/'manifest.json',{'status':'POST_HOC_CHUNK_COUNT_SWEEP','model':a.model,'expected_units':228,'source_sha256':{p.name:sha(p) for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.json','.md','.jsonl')},'block':1024,'dtype':'float16','seed':20260930})
  dump(out/'queue.json',prepared['queue']);assert torch.cuda.is_available();core.gpu();torch.manual_seed(20260930);torch.set_num_threads(8)
  tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(a.model,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval();assert model.config.num_hidden_layers==36
  dump(out/'environment.json',{'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),'gpu':core.gpu(),'model_config_sha256':sha(Path(a.model)/'config.json'),'tokenizer_sha256':sha(Path(a.model)/'tokenizer.json'),'dtype':'float16'})
  eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);selected=json.loads((ROOT/'selection.json').read_text());samples={x['row']['sample_id']:x for x in selected['samples']};encoded={}
  for job in prepared['queue']:
   if time.time()-started>a.hours*3600:raise TimeoutError('sweep wall-time budget exceeded')
   dump(out/'status.json',{'status':'RUNNING','completed_units':done,'expected_units':228,'current':job,'elapsed_seconds':time.time()-started,'pid':os.getpid()});core.gpu()
   item=samples[job['sample_id']];row=item['row'];sid=row['sample_id'];begin=time.time()
   if sid not in encoded:encoded[sid]=encode(tok,row)
   ex=encoded[sid]
   if job['method']=='full':
    result=core.gate(model,tok,ex,eos,block=1024);same_probes(result,item['historical']['full_b1024']);result['historical_reproduction']='PASS'
   elif job['method']=='all_compressed':
    assert (out/'units'/f'{sid}__full.json').exists()
    result=core.execute(model,tok,ex,BASE,{},eos,block=1024);same_probes(result,item['historical']['d30_mean16_b1024']);core.Engine=Legacy
    try:legacy=core.execute(model,tok,ex,BASE,{},eos,block=1024)
    finally:core.Engine=Engine
    same_probes(result,legacy);assert result['shape']==legacy['shape'];result['legacy_engine_equivalence']='PASS';result['historical_reproduction']='PASS'
   else:
    assert job['method']=='sweep' and (out/'units'/f'{sid}__all_compressed.json').exists()
    subset=set(job['chunk_indices']);spec={'kind':'fixed','depth':30,'keep':64,'chunk_keeps':{str(i):16 for i in subset}}
    result=core.execute(model,tok,ex,spec,{},eos,block=1024)
    assert len(result['routes'])==256 and [r['keep'] for r in result['routes']]==[16 if i in subset else 64 for i in range(256)]
    assert result['shape']['compressed_chunks']==job['count'] and result['shape']['saved_T_layer_tokens']==6*48*job['count']
   assert result['probes']['control']['correct'],'control probe wrong'
   dump(out/'units'/(job['unit_key']+'.json'),{'status':'COMPLETED','job':job,'role':item['role'],'sample_id':sid,'unit_seconds':time.time()-begin,**result});done+=1
   print(json.dumps({'done':done,'total':228,'unit':job['unit_key'],'correct':result['probes']['task']['correct'],'answer':result['probes']['task']['answer']},ensure_ascii=False),flush=True)
  dump(out/'status.json',{'status':'COMPLETED','completed_units':done,'expected_units':228,'elapsed_seconds':time.time()-started,'finished_unix':time.time()})
  from report import report
  report(out)
 except BaseException as exc:
  dump(out/'status.json',{'status':'FAILED','completed_units':done,'current':job,'elapsed_seconds':time.time()-started,'error':repr(exc),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
