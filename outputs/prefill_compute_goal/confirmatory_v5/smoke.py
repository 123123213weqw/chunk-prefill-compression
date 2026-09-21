"""GPU preflight uses exposed old diagnostics; no fresh v5 answers inspected."""
import json,torch
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
from data import encode
from core import gate,execute
from clean import timed
from specs import METHODS
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(8);torch.manual_seed(20260930);path='<MODEL_ROOT>/Qwen3-4B-Instruct-2507'
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True);model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
 rows=[json.loads(l) for l in (ROOT.parent/'long_validation_v4/diagnostic.jsonl').read_text().splitlines()];eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
 for length in (4096,16384):
  r=next(r for r in rows if r['target_tokens']==length);ex=encode(tok,r);g=gate(model,tok,ex,eos,block=4096);print('GATE',length,g['gates'],flush=True)
  for name,spec in METHODS.items():
   x=execute(model,tok,ex,spec,{},eos,block=4096);print('SHAPE',length,name,x['shape']['status'],flush=True)
  for block in (1024,4096):
   for name in ('full',*METHODS):
    x=timed(model,ex['segments'],METHODS.get(name),block);print('CLEAN_SMOKE',length,name,block,x['seconds'],x['peak_allocated_bytes'],flush=True)
 print('GPU_PREFLIGHT_PASS',flush=True)
