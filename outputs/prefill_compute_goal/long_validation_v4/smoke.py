"""Real-GPU preflight on exposed v3 diagnostic data, not fresh validation."""
import json,torch
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForCausalLM
from core import gate,execute
from data import encode
from specs import METHODS
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
 torch.set_num_threads(8);torch.manual_seed(20260925);path='<MODEL_ROOT>/Qwen3-4B-Instruct-2507'
 tok=AutoTokenizer.from_pretrained(path,local_files_only=True);m=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
 rows=[json.loads(l) for l in (ROOT/'diagnostic.jsonl').read_text().splitlines()];thresholds=json.loads((ROOT/'thresholds.json').read_text())['values'];foils=json.loads((ROOT/'counterfactuals.json').read_text());eos=m.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos])
 for length in (4096,16384):
  r=next(r for r in rows if r['target_tokens']==length);ex=encode(tok,r);foil=foils[r['sample_id']]
  g=gate(m,tok,ex,eos,foil);print('GATE',length,g['gates'],flush=True)
  for method in ('d18_k48','d27_k32'):
   for op,idx in [('only_chunk',1),('restore_chunk',length//64-1)]:
    x=execute(m,tok,ex,{**METHODS[method],op:idx},thresholds,eos,foil);print('SMOKE',length,method,op,x['shape']['compressed_chunks'],x['likelihood']['log_likelihood_margin'],flush=True)
  x=execute(m,tok,ex,METHODS['d27_k32_endpoint'],thresholds,eos,foil);print('ENDPOINT',length,x['shape']['status'],flush=True)
 print('GPU_SMOKE_PASS',flush=True)
