"""No diagnostic hooks or feature collection in timed prefix execution."""
import time,torch
from transformers import DynamicCache
from engine import Engine
from core import gpu,release

@torch.inference_mode()
def native_prefix(model,segments,block):
 device=next(model.parameters()).device;cache=DynamicCache(config=model.config);logical=0
 for stage in ('S','T','R'):
  ids=segments[stage]
  for off in range(0,len(ids),block):
   part=ids[off:off+block]
   out=model.model(input_ids=torch.tensor([part],device=device),position_ids=torch.arange(logical,logical+len(part),device=device)[None],past_key_values=cache,use_cache=True)
   logical+=len(part);del out
 return cache

@torch.inference_mode()
def timed(model,segments,spec,block):
 # No model hooks, including accidental correctness hooks carried over.
 assert all(not module._forward_pre_hooks and not module._forward_hooks for module in model.modules())
 release();before=gpu();torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
 if spec is None:state=native_prefix(model,segments,block)
 else:
  engine=Engine(model,spec,record=False);state=engine.prefill(engine.new_state(),segments,block)
 torch.cuda.synchronize();seconds=time.perf_counter()-start
 allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
 if spec is not None:assert not engine.routes and not engine.features and not engine.attn_rows
 after=gpu();del state
 if spec is not None:del engine
 release();return {'seconds':seconds,'base_allocated_bytes':base,'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,'before_gpu':before,'after_gpu':after,'diagnostics_enabled':False}
