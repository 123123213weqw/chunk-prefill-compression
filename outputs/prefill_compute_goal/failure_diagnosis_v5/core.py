import gc,json,os,subprocess,time,math
from collections import Counter
import torch
from engine import Engine,ShapeAudit
from scoring import strict_score

def release():gc.collect();torch.cuda.empty_cache()
def gpu():
 raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True,timeout=20)
 foreign=[r for r in raw.splitlines() if int(r.split(',')[0])!=os.getpid()]
 if foreign:raise InterruptedError('Foreign GPU processes present; not killed: '+str(foreign))
 telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=name,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=20).strip()
 return {'foreign':foreign,'processes':raw.strip(),'telemetry':telemetry}

@torch.inference_mode()
def probe(e,s,q,eos):
 cp=s.checkpoint();ids=[]
 try:
  logits=e.forward(s,q,'Q');first=logits[0].float().cpu();assert torch.isfinite(first).all()
  for i in range(96):
   t=int(logits[0].argmax());ids.append(t)
   if t in eos:break
   if i<95:logits=e.forward(s,[t],'decode')
  return ids,first
 finally:s.restore(cp)

@torch.inference_mode()
def likelihood(e,s,q,tokens):
 cp=s.checkpoint();logp=0.
 try:
  logits=e.forward(s,q,'Q')
  for i,t in enumerate(tokens):
   logp+=float(logits[0].float().log_softmax(-1)[t])
   if i+1<len(tokens):logits=e.forward(s,[t],'decode')
 finally:s.restore(cp)
 assert math.isfinite(logp),'non-finite likelihood'
 return {'sum_logp':logp,'tokens':len(tokens),'mean_logp':logp/len(tokens)}

def score_ids(tok,ids,p,eos):
 answer=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
 return {'answer':answer,'gold':p['gold'],'ids':ids,'hit_cap':len(ids)==96 and ids[-1] not in eos,**strict_score(answer,p['gold'])}

def expected_routes(e,seg):
 if e.native:return [{'chunk_start':len(seg['S'])+j,'merge_depth':e.model.config.num_hidden_layers,'keep':64} for j in range(0,len(seg['T']),64)]
 return e.routes

def shape(e,s,h,seg):
 L=e.model.config.num_hidden_layers;routes=expected_routes(e,seg);assert len(routes)==len(seg['T'])//64
 expected=[len(seg['S'])+len(seg['R'])+sum(64 if i<r['merge_depth'] else r['keep'] for r in routes) for i in range(L)]
 counts=Counter()
 for r in h.rows:counts[(r['layer'],r['op'])]+=r['tokens']
 assert [len(x) for x in s.positions]==expected
 assert len(counts)==L*7 and all(v==expected[i] for (i,op),v in counts.items())
 changes=sum(r['keep']!=64 for r in routes)
 if 'only_chunk' in e.spec:assert changes==1
 if 'restore_chunk' in e.spec:assert changes==len(routes)-1
 s.validate()
 return {'status':'PASS','layer_lengths':expected,'linear_flops':h.linear_flops(),'kv_bytes':s.tensor_bytes(),
         'projection_input_checks':len(counts),'compressed_chunks':changes,'total_chunks':len(routes),
         'saved_T_layer_tokens':sum((L-r['merge_depth'])*(64-r['keep']) for r in routes)}

def canonical(tok,gold):
 return tok.encode(json.dumps(gold,ensure_ascii=False,separators=(',',':')),add_special_tokens=False)+[tok.convert_tokens_to_ids('<|im_end|>')]

def diag_scores(e,s,example,tok,foil):
 q=example['probes'][0]['ids'];a=likelihood(e,s,q,canonical(tok,example['probes'][0]['gold']));b=likelihood(e,s,q,canonical(tok,foil))
 return {'correct':a,'counterfactual':b,'log_likelihood_margin':a['sum_logp']-b['sum_logp']}

@torch.inference_mode()
def execute(model,tok,example,spec,thresholds,eos,foil=None,block=1024):
 release();gpu();e=Engine(model,spec,thresholds,record=True);h=ShapeAudit(model);torch.cuda.synchronize();start=time.perf_counter()
 try:s=e.prefill(e.new_state(),example['segments'],block)
 finally:h.close()
 torch.cuda.synchronize();seconds=time.perf_counter()-start;result={'shape':shape(e,s,h,example['segments']),'instrumented_prefill_seconds':seconds,'probes':{}}
 # Freeze prefix route data before probes; per-target score retained only as diagnostic.
 result['routes']=[dict(r) for r in e.routes]
 result['route_counts']={f'd{d}_k{k}':n for (d,k),n in Counter((r['merge_depth'],r['keep']) for r in e.routes).items()}
 target=spec.get('only_chunk',spec.get('restore_chunk'))
 if target is not None:result['target_route']=e.routes[target]
 for p in example['probes']:
  ids,_=probe(e,s,p['ids'],eos);result['probes'][p['name']]=score_ids(tok,ids,p,eos)
 if foil is not None:result['likelihood']=diag_scores(e,s,example,tok,foil)
 gpu();del e,s,h;release();return result

@torch.inference_mode()
def gate(model,tok,example,eos,foil=None,block=1024):
 release();gpu();a=Engine(model,{'kind':'native'},record=True);h=ShapeAudit(model)
 try:sa=a.prefill(a.new_state(),example['segments'],block)
 finally:h.close()
 result={'shape':shape(a,sa,h,example['segments']),'probes':{},'gates':[]}
 refs={p['name']:probe(a,sa,p['ids'],eos) for p in example['probes']}
 b=Engine(model,record=True,measure_depths=(18,27) if foil is not None else ());sb=b.prefill(b.new_state(),example['segments'],block);sb.validate();errs=[]
 for x,y in zip(sa.cache.layers,sb.cache.layers):
  for xx,yy in ((x.keys,y.keys),(x.values,y.values)):
   assert xx.shape==yy.shape
   errs.append(float(((xx.float()-yy.float()).square().sum()/xx.float().square().sum().clamp_min(1e-20)).sqrt()))
 for p in example['probes']:
  ia,la=refs[p['name']];ib,lb=probe(b,sb,p['ids'],eos)
  rms=float(((la-lb).square().sum()/la.square().sum().clamp_min(1e-20)).sqrt());kl=float((la.softmax(-1)*(la.log_softmax(-1)-lb.log_softmax(-1))).sum())
  g={'probe':p['name'],'kv_rms':max(errs),'logit_rms':rms,'kl':kl,'ids_equal':ia==ib}
  assert max(errs)<=.005 and rms<=.005 and kl<=.001 and ia==ib,g
  result['gates'].append(g);result['probes'][p['name']]=score_ids(tok,ia,p,eos)
 if foil is not None:
  result['teacher_features']=b.features;result['likelihood']=diag_scores(a,sa,example,tok,foil)
 gpu();del a,b,sa,sb,h;release();return result
