"""Finite, fail-closed CUDA experiment. Existing output directories are rejected."""
import argparse,gc,hashlib,json,os,random,shutil,statistics,subprocess,sys,time,traceback
from collections import Counter
from pathlib import Path
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM,DynamicCache
from engine import Engine,ShapeAudit,plans
from data import encode,validate
from scoring import strict_score
ROOT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def append(p,x):
    with Path(p).open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
def release():gc.collect();torch.cuda.empty_cache()
def gpu():
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
    foreign=[r for r in apps.splitlines() if int(r.split(',')[0])!=os.getpid()]
    telemetry=subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,temperature.gpu,clocks.sm,power.draw','--format=csv,noheader,nounits'],text=True).strip()
    return {'foreign':foreign,'telemetry':telemetry}
def ensure_gpu(x):
    if x['foreign']:raise RuntimeError('foreign GPU processes: '+str(x))

@torch.inference_mode()
def probe(e,state,p,eos):
    cp=state.checkpoint();ids=[]
    try:
        logits=e.forward(state,p['ids'],'Q');first=logits[0].float().cpu()
        for _ in range(96):
            t=int(logits[0].argmax());ids.append(t)
            if t in eos:break
            if len(ids)<96:logits=e.forward(state,[t],'decode')
        return ids,first
    finally:state.restore(cp)

def row(src,method,p,tokens,tok,eos):
    answer=tok.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return {**{k:src[k] for k in ('sample_id','pair_id','member','instance','split','task','target_tokens')},'method':method,'probe':p['name'],
            'status':'OK','answer':answer,'gold':p['gold'],'generated_ids':tokens,'hit_cap':len(tokens)==96 and tokens[-1] not in eos,**strict_score(answer,p['gold'])}

def audit_shape(e,s,audit,seg):
    L=e.model.config.num_hidden_layers
    expect=[len(seg['S'])+len(seg['R'])+sum(64 if i<r['merge_depth'] else r['keep'] for r in e.routes) for i in range(L)]
    actual=Counter((r['layer'],r['op']) for r in [])
    for r in audit.rows:actual[(r['layer'],r['op'])]+=r['tokens']
    for i in range(L):
        assert len(s.positions[i])==expect[i]
        for op in ('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'):assert actual[(i,op)]==expect[i]
    assert len(e.routes)==len(seg['T'])//64
    budget=sum((L-r['merge_depth'])*(64-r['keep']) for r in e.routes)
    if e.kind in ('adaptive_equal','random_equal'):assert budget==len(seg['T'])*L//8
    dense=sum(4*r['queries']*r['keys']*e.model.config.num_attention_heads*e.model.config.head_dim for r in e.attn_rows)
    return {'expected_layer_tokens':expect,'actual_linear_inputs':[{'layer':i,'op':op,'tokens':v} for (i,op),v in sorted(actual.items())],
            'linear_flops':audit.linear_flops(),'dense_attention_flops_estimate':dense,'saved_T_layer_tokens':budget,
            'kv_bytes':s.tensor_bytes(),'routes':e.routes,'status':'PASS'}

@torch.inference_mode()
def gates(model,tok,samples,eos,out,depths):
    features=[]
    for n,s in enumerate(samples):
        src=s['source'];sid=src['sample_id'];dump(out/'progress.json',{'phase':src['split']+'_identity_gates','document':n+1,'total':len(samples),'sample_id':sid})
        ensure_gpu(gpu())
        a=Engine(model,{'kind':'native'});sa=a.prefill(a.new_state(),s['segments']);sa.validate()
        refs={p['name']:probe(a,sa,p,eos) for p in s['probes']}
        b=Engine(model,record=True,measure_depths=depths);sb=b.prefill(b.new_state(),s['segments']);sb.validate()
        errors=[]
        for x,y in zip(sa.cache.layers,sb.cache.layers):
            for xx,yy in ((x.keys,y.keys),(x.values,y.values)):
                assert xx.shape==yy.shape
                errors.append(float(((xx.float()-yy.float()).square().sum()/xx.float().square().sum().clamp_min(1e-20)).sqrt()))
        for p in s['probes']:
            ta,xa=refs[p['name']];tb,xb=probe(b,sb,p,eos)
            err=float(((xa-xb).square().sum()/xa.square().sum().clamp_min(1e-20)).sqrt())
            kl=float((xa.softmax(-1)*(xa.log_softmax(-1)-xb.log_softmax(-1))).sum())
            gate={'sample_id':sid,'split':src['split'],'probe':p['name'],'kv_rms':max(errors),'logit_rms':err,'kl':kl,'ids_equal':ta==tb}
            append(out/'gates.jsonl',gate)
            assert max(errors)<=.005 and err<=.005 and kl<=.001 and ta==tb,gate
            append(out/'correctness.jsonl',row(src,'full',p,ta,tok,eos));append(out/'correctness.jsonl',row(src,'identity',p,tb,tok,eos))
        fs=[{'sample_id':sid,'split':src['split'],**f} for f in b.features];features.extend(fs)
        for f in fs:append(out/'features.jsonl',f)
        print('GATE PASS',src['split'],n+1,src['task'],src['target_tokens'],flush=True)
        del sa,sb,a,b;release()
    return features

@torch.inference_mode()
def correctness(model,tok,samples,methods,thresholds,eos,out):
    rng=random.Random(20260920)
    for n,s in enumerate(samples):
        names=list(methods);rng.shuffle(names)
        for name in names:
            src=s['source'];dump(out/'progress.json',{'phase':src['split']+'_compression','document':n+1,'total':len(samples),'method':name})
            ensure_gpu(gpu());e=Engine(model,methods[name],thresholds,record=True);h=ShapeAudit(model);state=None
            try:state=e.prefill(e.new_state(),s['segments'])
            finally:h.close()
            state.validate();shape=audit_shape(e,state,h,s['segments']);append(out/'shapes.jsonl',{'sample_id':src['sample_id'],'split':src['split'],'method':name,**shape})
            for p in s['probes']:
                tokens,_=probe(e,state,p,eos);append(out/'correctness.jsonl',row(src,name,p,tokens,tok,eos))
            print('CORRECTNESS',src['split'],n+1,name,flush=True)
            del state,e,h;release()

@torch.inference_mode()
def native_prefix(model,seg,block):
    cache=DynamicCache(config=model.config);pos=0
    for stage in ('S','T','R'):
        ids=seg[stage]
        for off in range(0,len(ids),block):
            part=ids[off:off+block]
            output=model.model(input_ids=torch.tensor([part],device='cuda'),position_ids=torch.arange(pos,pos+len(part),device='cuda')[None],past_key_values=cache,use_cache=True)
            pos+=len(part);del output
    return cache

@torch.inference_mode()
def timings(model,samples,methods,thresholds,out):
    selected=[s for s in samples if s['source']['instance']==2 and s['source']['member']==0]
    for s in selected:
        for rep in range(-1,5):
            names=list(methods);random.Random(20260920+rep+int(s['source']['sample_id'][:6],16)).shuffle(names)
            for name in names:
                src=s['source'];dump(out/'progress.json',{'phase':'timings','sample_id':src['sample_id'],'method':name,'repeat':rep})
                release();before=gpu();ensure_gpu(before);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated()
                start=time.perf_counter()
                if name.startswith('native_b'):state=native_prefix(model,s['segments'],int(name.split('_b')[1]))
                else:
                    e=Engine(model,methods[name],thresholds,record=False);state=e.prefill(e.new_state(),s['segments'])
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved();after=gpu()
                append(out/'timings.jsonl',{**{k:src[k] for k in ('sample_id','task','target_tokens')},'method':name,'repeat':rep,'warmup':rep<0,
                       'seconds':seconds,'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,'base_bytes':base,'before':before,'after':after})
                ensure_gpu(after);del state;release()
            print('TIMING',src['task'],src['target_tokens'],rep,flush=True)

def quantile(v,p):
    v=sorted(v);i=(len(v)-1)*p;lo=int(i);return v[lo]+(v[min(lo+1,len(v)-1)]-v[lo])*(i-lo)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--out',required=True);ap.add_argument('--dataset',default=str(ROOT/'dataset.jsonl'));args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    try:
        assert torch.cuda.is_available(),'CUDA required';ensure_gpu(gpu());torch.manual_seed(20260920);torch.set_num_threads(8)
        raw=[json.loads(l) for l in Path(args.dataset).read_text().splitlines()];validation=validate(raw)
        snapshot=out/'source';snapshot.mkdir()
        for p in ROOT.iterdir():
            if p.is_file() and p.suffix in ('.py','.md'):shutil.copy2(p,snapshot/p.name)
        shutil.copy2(args.dataset,out/'dataset.jsonl')
        tok=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,dtype=torch.float16,attn_implementation='sdpa').to('cuda').eval()
        fixed,early,late=plans(model.config.num_hidden_layers);depths=sorted({v['depth'] for v in fixed.values()})
        primary={f'd{early}_k48':fixed[f'd{early}_k48'],f'd{late}_k32':fixed[f'd{late}_k32']}
        methods={**fixed,**{k:{'kind':k} for k in ('adaptive_equal','random_equal','adaptive_safe')}}
        eos=model.generation_config.eos_token_id;eos=set(eos if isinstance(eos,list) else [eos]);eos.discard(None)
        samples=[encode(tok,r) for r in raw];cal=[s for s in samples if s['source']['split']=='calibration'];val=[s for s in samples if s['source']['split']=='validation']
        metadata={p.name:sha(p) for p in Path(args.model).iterdir() if p.is_file() and p.suffix in ('.json','.jinja','.txt')}
        manifest={'model':args.model,'layers':model.config.num_hidden_layers,'model_metadata_sha256':metadata,'dataset_sha256':sha(args.dataset),
                  'source_sha256':{p.name:sha(p) for p in snapshot.iterdir()},'torch':torch.__version__,'transformers':transformers.__version__,
                  'gpu':gpu(),'data_validation':validation,'fixed':fixed,'validation_methods':['full','identity',*methods],'calibration_methods':['full','identity',*primary],
                  'primary':list(primary),'seed':20260920,'dtype':'float16','attn':'sdpa','block':1024,'max_generated_tokens':96,
                  'token_lengths':{s['source']['sample_id']:{k:len(v) for k,v in s['segments'].items()} for s in samples}}
        dump(out/'manifest.json',manifest);dump(out/'status.json',{'status':'RUNNING','pid':os.getpid()})
        f=gates(model,tok,cal,eos,out,depths)
        thresholds={'early_median':quantile([r['score'] for r in f if r['depth']==early and r['keep']==48],.5),
                    'early_q25':quantile([r['score'] for r in f if r['depth']==early and r['keep']==48],.25),
                    'late_q25':quantile([r['score'] for r in f if r['depth']==late and r['keep']==32],.25)}
        dump(out/'thresholds.json',{'values':thresholds,'source_split':'calibration','sample_ids':[s['source']['sample_id'] for s in cal],
                                  'uses_answers':False,'frozen_before_validation':True,'calibration_features_sha256':sha(out/'features.jsonl')})
        correctness(model,tok,cal,primary,thresholds,eos,out)
        gates(model,tok,val,eos,out,depths)
        correctness(model,tok,val,methods,thresholds,eos,out)
        timing_methods={'native_b1024':None,'native_b4096':None,**primary,**{k:methods[k] for k in ('adaptive_equal','random_equal','adaptive_safe')}}
        timings(model,val,timing_methods,thresholds,out)
        subprocess.run([sys.executable,str(ROOT/'report.py'),'--run',str(out)],check=True)
        dump(out/'status.json',{'status':'COMPLETED','pid':os.getpid(),'finished_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    except BaseException as e:
        dump(out/'status.json',{'status':'FAILED','pid':os.getpid(),'error':repr(e),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
