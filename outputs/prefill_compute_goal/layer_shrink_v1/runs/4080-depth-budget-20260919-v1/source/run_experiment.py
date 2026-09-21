#!/usr/bin/env python3
"""Identity-gated hidden-state shrinking study: correctness then timed prefill.

Only development data. Real-model execution requires CUDA; never silently falls
back to CPU. Input tokenizer hashes and source snapshots are recorded.
"""
import argparse,gc,hashlib,json,os,random,shutil,statistics,subprocess,sys,time,traceback
from collections import Counter
from pathlib import Path
import torch,transformers
from transformers import AutoModelForCausalLM,AutoTokenizer,DynamicCache
from engine import Engine,ShapeAudit

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'validation_v3'))
from methods import strict_score

from specs import DATASET_SHA256,SPECS,METHODS,IDENTITIES,CANDIDATES,TIMING_METHODS,expected_shape

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def sync():torch.cuda.synchronize()
def release():gc.collect();torch.cuda.empty_cache()
def kv_bytes(cache):return sum(l.keys.numel()*l.keys.element_size()+l.values.numel()*l.values.element_size() for l in cache.layers if l.is_initialized)

def encoded(tok,seg):
    ids=tok.encode(seg['text'],add_special_tokens=False)
    assert len(ids)==seg['token_count']
    assert hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest()==seg['token_id_sha256']
    return ids

def encode_sample(tok,s):
    if s['split']!='development':raise ValueError('formal test remains locked')
    return {'source':s,'segments':{k:encoded(tok,s['segments'][k]) for k in ('S','T','R')},
            'probes':[{**p,'ids':encoded(tok,p['Q'])} for p in s['probes'] if p['probe_id'] in ('readout','control')]}

def make_engine(model,name):
    spec=SPECS[name]
    return Engine(model,**{k:spec[k] for k in ('method','split_depth','native')})

def gpu_snapshot():
    """Boundary telemetry only; not proof against between-sample interference."""
    try:
        def query(options):
            return subprocess.check_output(['nvidia-smi',*options,'--format=csv,noheader,nounits'],text=True,timeout=10).strip()
        apps=query(['--query-compute-apps=pid,used_gpu_memory'])
        foreign=[]
        for row in apps.splitlines():
            pid,mem=row.split(',',1)
            if int(pid)!=os.getpid():foreign.append({'pid':int(pid),'memory_mib':mem.strip()})
        gpu=query(['--query-gpu=uuid,name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,clocks.mem,power.draw'])
        return {'status':'OK','foreign_compute_processes':foreign,'gpu_csv':gpu}
    except Exception as exc:return {'status':'unknown','error':repr(exc)}

@torch.inference_mode()
def probe(engine,state,p,max_tokens,eos):
    cp=state.checkpoint();generated=[]
    try:
        logits=engine.forward(state,p['ids'],'Q');first=logits[0].detach().float().cpu()
        for _ in range(max_tokens):
            token=int(logits[0].argmax());generated.append(token)
            if token in eos or len(generated)>=max_tokens:break
            logits=engine.forward(state,[token],'decode')
        return generated,first
    finally:state.restore(cp)


def shape_result(audit,lengths,method):
    expected_full,expected_upper,ratio=expected_shape(lengths,method)
    counts=Counter()
    for r in audit.rows:counts[(r['layer'],r['op'])]+=r['tokens']
    depth=SPECS[method]['split_depth']
    for i in range(36):
        for op in ['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']:
            assert counts[(i,op)]==(expected_full if i<depth else expected_upper)
    return {'actual_module_inputs':audit.rows,'linear_flops':audit.linear_flops(),
            'split_depth':depth,'keep':SPECS[method]['keep'],'segment_lengths':lengths,
            'lower_layer_tokens':expected_full,'upper_layer_tokens':expected_upper,
            'linear_token_layer_ratio':ratio}

@torch.inference_mode()
def correctness(model,tok,samples,args,out,event):
    eos=model.generation_config.eos_token_id
    eos=set(eos if isinstance(eos,list) else [eos])
    rows=[];shapes=[];gates=[]
    # Both boundary configurations must match native BEFORE lossy conditions.
    for s in samples:
        sid=s['source']['sample_id'];native=make_engine(model,'native_full')
        a=native.prefill(native.new_state(),s['segments'],args.block);a.validate()
        reference={p['probe_id']:probe(native,a,p,args.max_tokens,eos) for p in s['probes']}
        for identity_name in IDENTITIES:
            identity=make_engine(model,identity_name)
            b=identity.prefill(identity.new_state(),s['segments'],args.block);b.validate()
            cache_errors=[]
            for la,lb in zip(a.cache.layers,b.cache.layers):
                assert la.keys.shape==lb.keys.shape and la.values.shape==lb.values.shape
                for x,y in [(la.keys,lb.keys),(la.values,lb.values)]:
                    err=float(((x.float()-y.float()).square().sum()/x.float().square().sum().clamp_min(1e-20)).sqrt())
                    assert err<.005,('identity cache mismatch',identity_name,sid,err)
                    cache_errors.append(err)
            for p in s['probes']:
                ta,xa=reference[p['probe_id']];tb,xb=probe(identity,b,p,args.max_tokens,eos)
                logit_error=float(((xa-xb).square().sum()/xa.square().sum().clamp_min(1e-20)).sqrt())
                kl=float((xa.softmax(-1)*(xa.log_softmax(-1)-xb.log_softmax(-1))).sum())
                gate={'sample_id':sid,'identity_method':identity_name,'probe':p['probe_id'],
                      'max_cache_relative_rms':max(cache_errors),'logit_relative_rms':logit_error,
                      'first_token_kl':kl,'generated_tokens_equal':ta==tb}
                gates.append(gate);dump(out/'identity_gates.json',gates)
                assert logit_error<.005 and kl<.001 and ta==tb,('identity generation gate failed',gate)
            del b;release()
            event({'event':'identity_gate_pass','sample_id':sid,'method':identity_name})
        del a;release()
    for s in samples:
        sid=s['source']['sample_id']
        for method in METHODS:
            engine=make_engine(model,method);audit=ShapeAudit(model)
            try:state=engine.prefill(engine.new_state(),s['segments'],args.block)
            finally:audit.close()
            state.validate()
            shape=shape_result(audit,{k:len(v) for k,v in s['segments'].items()},method)
            shapes.append({'sample_id':sid,'method':method,**shape})
            for p in s['probes']:
                ids,first=probe(engine,state,p,args.max_tokens,eos)
                answer=tok.decode(ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                row={'sample_id':sid,'pair_id':s['source']['pair_id'],'template':s['source']['template'],'method':method,'probe':p['probe_id'],
                     'answer':answer,'gold':p['gold'],'generated_ids':ids,'generated_tokens':len(ids),
                     'hit_token_cap':len(ids)==args.max_tokens and ids[-1] not in eos,
                     'cache_bytes':state.tensor_bytes(),'position_metadata_bytes':state.position_bytes(),
                     'layer_lengths':[len(x) for x in state.positions],**strict_score(answer,p['gold'])}
                rows.append(row);event({'event':'correctness',**row})
                dump(out/'correctness.json',rows)
            del state;release()
    dump(out/'correctness.json',rows);dump(out/'shape_audits.json',shapes)
    table={}
    for method in METHODS:
        table[method]={}
        for pname in ('readout','control'):
            rr=[r for r in rows if r['method']==method and r['probe']==pname]
            table[method][pname]={'correct':sum(r['correct'] for r in rr),'n':len(rr),
                                'field_correct':sum(r['field_correct'] for r in rr),'fields':sum(r['fields'] for r in rr)}
    dump(out/'correctness_summary.json',table)
    return rows

@torch.inference_mode()
def fast_native_prefix(model,segments,block):
    """Unmodified HF model, no position-bookkeeping or shape hooks in native timing."""
    cache=DynamicCache(config=model.config);logical=0
    for stage in ('S','T','R'):
        ids=segments[stage]
        for off in range(0,len(ids),block):
            part=ids[off:off+block]
            result=model.model(input_ids=torch.tensor([part],device='cuda:0'),
                               position_ids=torch.arange(logical,logical+len(part),device='cuda:0')[None],
                               past_key_values=cache,use_cache=True)
            logical+=len(part);del result
    return cache,logical

@torch.inference_mode()
def benchmark(model,samples,args,out,event):
    methods=list(TIMING_METHODS)
    rows=[]
    # One member per pair: correctness uses both. No duplication as independent timing docs.
    selected=[s for s in samples if s['source']['member']==0]
    for s in selected:
        for repeat in range(-args.warmups,args.repeats):
            order=methods.copy();random.Random(20260908+repeat+int(s['source']['sample_id'][:4],16)).shuffle(order)
            for method in order:
                before_gpu=gpu_snapshot()
                state=cache=None;release();sync();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                try:
                    begin=time.perf_counter()
                    if method.startswith('native_b'):
                        cache,logical=fast_native_prefix(model,s['segments'],int(method.split('_b')[-1]))
                    else:
                        engine=make_engine(model,method);state=engine.prefill(engine.new_state(),s['segments'],args.block)
                        cache=state.cache;logical=state.logical_next
                    sync();prefix=time.perf_counter()-begin
                    # Same raw question and LM last-position projection across all methods.
                    question=next(p['ids'] for p in s['probes'] if p['probe_id']=='readout')
                    qstart=time.perf_counter()
                    if state is None:
                        output=model.model(input_ids=torch.tensor([question],device='cuda:0'),
                                           position_ids=torch.arange(logical,logical+len(question),device='cuda:0')[None],
                                           past_key_values=cache,use_cache=True)
                        token=int(model.lm_head(output.last_hidden_state[:,-1:])[0,0].argmax());del output
                    else:token=int(engine.forward(state,question,'Q')[0].argmax())
                    sync();qtime=time.perf_counter()-qstart
                    row={'sample_id':s['source']['sample_id'],'method':method,'repeat':repeat,'warmup':repeat<0,'status':'OK',
                         'prefix_seconds':prefix,'query_ttft_seconds':qtime,'total_ttft_seconds':prefix+qtime,
                         'first_token':token,'kv_bytes_after_Q':kv_bytes(cache),
                         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
                         'peak_allocated_over_model_bytes':torch.cuda.max_memory_allocated()-base}
                except torch.OutOfMemoryError as exc:
                    row={'sample_id':s['source']['sample_id'],'method':method,'repeat':repeat,'warmup':repeat<0,'status':'OOM','error':str(exc)}
                finally:
                    del state,cache;release()
                row.update(gpu_before=before_gpu,gpu_after=gpu_snapshot())
                row['isolation_observed']=all(t['status']=='OK' and not t['foreign_compute_processes'] for t in (row['gpu_before'],row['gpu_after']))
                rows.append(row);event({'event':'timing',**row});dump(out/'timings.json',rows)
    summary=[]
    for sid in sorted({r['sample_id'] for r in rows}):
        for method in methods:
            rr=[r for r in rows if r['sample_id']==sid and r['method']==method and not r['warmup']]
            good=[r for r in rr if r['status']=='OK']
            item={'sample_id':sid,'method':method,'measured':len(good),'oom':sum(r['status']=='OOM' for r in rr)}
            for metric in ['prefix_seconds','query_ttft_seconds','total_ttft_seconds','kv_bytes_after_Q','peak_allocated_bytes','peak_allocated_over_model_bytes']:
                if good:item[metric]={'median':statistics.median(r[metric] for r in good),'min':min(r[metric] for r in good),'max':max(r[metric] for r in good)}
            summary.append(item)
    dump(out/'performance_summary.json',summary)


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);p.add_argument('--out',required=True)
    p.add_argument('--block',type=int,default=1024);p.add_argument('--max-tokens',type=int,default=96)
    p.add_argument('--warmups',type=int,default=1);p.add_argument('--repeats',type=int,default=5);args=p.parse_args()
    if args.block!=1024 or args.max_tokens!=96:raise ValueError('v1 fixes block=1024, max_tokens=96')
    if args.warmups!=1 or args.repeats!=5:raise ValueError('v1 fixes 1 warmup and 5 repeats')
    if digest(args.dataset)!=DATASET_SHA256:raise ValueError('dataset differs from preregistered smoke4')
    if not torch.cuda.is_available():raise RuntimeError('CUDA server required; no CPU model fallback')
    torch.set_num_threads(4);torch.manual_seed(20260908)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False);(out/'source').mkdir()
    raw=[json.loads(l) for l in Path(args.dataset).read_text().splitlines()]
    if len(raw)!=4 or len({s['pair_id'] for s in raw})!=2 or any(s['split']!='development' or s['length']!=4096 for s in raw):
        raise ValueError('v1 requires frozen 4K development smoke4; no held-out evaluation')
    for file in [Path(__file__),ROOT/'engine.py',ROOT/'test_engine.py',ROOT/'specs.py',ROOT/'report_experiment.py',ROOT/'test_report.py',ROOT/'PROTOCOL.md',ROOT.parents[1]/'validation_v3/methods.py',Path(args.dataset)]:shutil.copy2(file,out/'source'/file.name)
    meta={'status':'running','args':vars(args),'scope':'depth versus budget matched-compute development pilot','method_specs':SPECS,
          'timing_methods':list(TIMING_METHODS),'gpu_at_start':gpu_snapshot(),
          'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),'dtype':'float16',
          'dataset_sha256':digest(args.dataset),'test_used':False,'shape_hooks_excluded_from_timing':True,
          'native_performance_baseline':'unmodified HF model.model, no layerwise position tracking',
          'source_sha256':{f.name:digest(f) for f in (out/'source').iterdir()}}
    dump(out/'manifest.json',meta)
    def event(row):
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        print(json.dumps(row,ensure_ascii=False),flush=True)
    try:
        tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        samples=[encode_sample(tok,s) for s in raw]
        model=AutoModelForCausalLM.from_pretrained(args.model,dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).to('cuda:0').eval()
        assert model.config.num_hidden_layers==36 and model.config.model_type=='qwen3'
        meta['model_config_sha256']=digest(Path(args.model)/'config.json');dump(out/'manifest.json',meta)
        correctness(model,tok,samples,args,out,event)
        benchmark(model,samples,args,out,event)
        meta.update(status='completed',accuracy_scope='development smoke only',performance_scope='prefill and query TTFT only; not full-answer latency')
    except Exception as exc:
        meta.update(status='failed',error=repr(exc),traceback=traceback.format_exc());dump(out/'manifest.json',meta);raise
    dump(out/'manifest.json',meta);print(json.dumps(meta),flush=True)
if __name__=='__main__':main()
