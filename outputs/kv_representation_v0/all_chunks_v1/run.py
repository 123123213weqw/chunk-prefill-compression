#!/usr/bin/env python3
"""Simultaneously compress completed T chunks, one teacher layer at a time."""
import argparse,collections,gzip,hashlib,json,math,platform,time,traceback
from pathlib import Path
import torch
from compressor_source import cluster,groups

METHODS=['identity_raw','mean_no_mass_fp16','mean_mass_fp16','cluster_mass_fp16','cluster_mass_fp32']
METRICS=['relative_l2','full_logZ_error','compressed_history_mass','mass_abs_error','completed_chunks_logZ_rmse','completed_chunks_mass_l1']

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')

def assemble(k,v,prototypes,ns,nt,pos):
    """Only chunks ending strictly before this query position are summarized."""
    completed=min(max((pos-ns)//64,0),nt//64)
    stop=ns+completed*64
    pk,pv,b=prototypes
    # Future prototypes may exist offline but are neither indexed nor attended.
    keys=torch.cat([k[:ns],pk[:completed].flatten(0,1),k[stop:pos+1]])
    vals=torch.cat([v[:ns],pv[:completed].flatten(0,1),v[stop:pos+1]])
    bias=torch.cat([torch.zeros(ns),b[:completed].flatten(),torch.zeros(pos+1-stop)])
    return keys,vals,bias,completed

def reference_tests():
    torch.manual_seed(20260907)
    k=torch.randn(160,8);v=torch.randn(160,8);q=torch.randn(4,8)
    # Exact identity assembly and future perturbation must preserve visible tensors.
    ns,nt,pos=10,128,83
    p=(k[ns:ns+nt].reshape(2,64,8),v[ns:ns+nt].reshape(2,64,8),torch.zeros(2,64))
    a=assemble(k,v,p,ns,nt,pos);assert a[3]==1
    assert torch.equal(a[0],k[:pos+1]) and torch.equal(a[1],v[:pos+1])
    k2=k.clone();v2=v.clone();k2[pos+1:]+=999;v2[pos+1:]-=999
    p2=(p[0].clone(),p[1].clone(),p[2].clone());p2[0][1]+=100;p2[1][1]-=100;p2[2][1]+=100
    b=assemble(k2,v2,p2,ns,nt,pos)
    assert all(torch.equal(a[i],b[i]) for i in range(3))
    # Two compressed chunks, each containing 16 exactly duplicated groups.
    k=torch.randn(32,8).repeat_interleave(4,0);v=torch.randn(128,8)
    entries=[groups(k[i:i+64],v[i:i+64],torch.arange(64)//4,16) for i in (0,64)]
    pk=torch.cat([x[0] for x in entries]);pv=torch.cat([x[1] for x in entries]);bias=torch.cat([x[2] for x in entries])
    rawk=torch.randn(20,8);rawv=torch.randn(20,8)
    actual=(q@torch.cat([k,rawk]).T/math.sqrt(8)).softmax(-1)@torch.cat([v,rawv])
    pred=(q@torch.cat([pk,rawk]).T/math.sqrt(8)+torch.cat([bias,torch.zeros(20)])).softmax(-1)@torch.cat([pv,rawv])
    err=float((actual-pred).abs().max());assert err<1e-6
    return {'status':'PASS','identity_visible_tensor_equality':True,'future_perturbation_invariance':True,'two_chunk_exact_merge_max_abs_error':err}

@torch.inference_mode()
def run(args,out):
    dump(out/'preflight.json',reference_tests())
    parent=Path(args.teacher);manifest=json.loads((parent/'manifest.json').read_text())
    assert manifest['status']=='completed' and manifest['test_set_used'] is False
    samples=[json.loads(l) for l in (parent/'source/smoke4.jsonl').read_text().splitlines()]
    assert all(s['split']=='development' for s in samples)
    assert [s['sample_id'] for s in samples]==manifest['samples']
    meta={'status':'running','host':platform.node(),'torch':torch.__version__,
          'teacher_manifest_sha256':digest(parent/'manifest.json'),'script_sha256':digest(__file__),
          'compressor_sha256':digest(Path(__file__).with_name('compressor_source.py')),
          'samples':manifest['samples'],'layers':manifest['layers'],'methods':METHODS,'n':64,'m':32,
          'teacher_dtype':'FP16','evaluation_dtype':'CPU FP32','test_set_used':False,
          'scope':'simultaneous completed T chunk compression within individual Full-teacher layers; no cross-layer propagation',
          'teacher_tensor_sha256':{}}
    dump(out/'manifest.json',meta)
    records=[];checks=[];timings=[];storage=[]
    for sample in samples:
        sid=sample['sample_id'];file=parent/(sid+'.pt');meta['teacher_tensor_sha256'][file.name]=digest(file)
        saved=torch.load(file,map_location='cpu',weights_only=True)
        assert saved['sample_id']==sid
        ns=saved['ns'];nt=sample['segments']['T']['token_count'];assert nt==4096
        for layer,data in saved['layers'].items():
            begun=time.perf_counter()
            K,V,Q=data['k'].float(),data['v'].float(),data['q'].float()
            nq=Q.shape[0]//K.shape[0];assert nq==4 and K.shape[-1]==128
            scale=1/math.sqrt(K.shape[-1])
            for kh in range(K.shape[0]):
                k,v=K[kh],V[kh]
                values={m:[] for m in METHODS}
                for start in range(ns,ns+nt,64):
                    kc,vc=k[start:start+64],v[start:start+64]
                    begin=time.perf_counter();cm=groups(kc,vc,cluster(kc,32),32)
                    timings.append({'sample_id':sid,'layer':layer,'kv_head':kh,'chunk':(start-ns)//64,'cpu_seconds':time.perf_counter()-begin})
                    pm=groups(kc,vc,torch.arange(64)//2,32)
                    values['identity_raw'].append((kc,vc,torch.zeros(64)))
                    values['mean_no_mass_fp16'].append((pm[0].half().float(),pm[1].half().float(),torch.zeros(32)))
                    values['mean_mass_fp16'].append((pm[0].half().float(),pm[1].half().float(),pm[2]))
                    values['cluster_mass_fp16'].append((cm[0].half().float(),cm[1].half().float(),cm[2]))
                    values['cluster_mass_fp32'].append(cm)
                prototypes={name:tuple(torch.stack([e[j] for e in entries]) for j in range(3)) for name,entries in values.items()}
                for qi,pos in enumerate(data['positions']):
                    q=Q[kh*nq:(kh+1)*nq,qi]
                    truth_logits=q@k[:pos+1].T*scale;truth=truth_logits.softmax(-1)@v[:pos+1]
                    ref=data['output'][kh*nq:(kh+1)*nq,qi].float()
                    err=float(((truth-ref).square().sum()/ref.square().sum().clamp_min(1e-20)).sqrt())
                    assert err<.005,(sid,layer,kh,qi,err)
                    checks.append({'sample_id':sid,'layer':layer,'kv_head':kh,'query_position':pos,'reference_relative_rms':err})
                    fullz=truth_logits.logsumexp(-1)
                    completed=min((pos-ns)//64,nt//64);assert completed>0
                    stop=ns+completed*64
                    blockz=truth_logits[:,ns:stop].reshape(nq,completed,64).logsumexp(-1)
                    blockmass=(blockz-fullz[:,None]).exp()
                    originalmass=blockmass.sum(-1)
                    for name in METHODS:
                        ck,cv,bias,nblocks=assemble(k,v,prototypes[name],ns,nt,pos)
                        assert nblocks==completed
                        logits=q@ck.T*scale+bias
                        pred=logits.softmax(-1)@cv;pz=logits.logsumexp(-1)
                        error=((pred-truth).square().sum(-1)/truth.square().sum(-1).clamp_min(1e-20)).sqrt()
                        if name=='identity_raw':assert float(error.max())<1e-6
                        width=64 if name=='identity_raw' else 32
                        bz=logits[:,ns:ns+width*completed].reshape(nq,completed,width).logsumexp(-1)
                        mass=(bz-pz[:,None]).exp();historymass=mass.sum(-1)
                        zrmse=(bz-blockz).square().mean(-1).sqrt()
                        massl1=(mass-blockmass).abs().sum(-1)
                        for h in range(nq):
                            r={'sample_id':sid,'pair_id':sample['pair_id'],'template':sample['template'],'layer':layer,'kv_head':kh,'query_head':kh*nq+h,
                               'query_position':pos,'bucket':'question' if pos>=saved['prefix'] else 'text','completed_chunks':completed,'method':name,
                               'relative_l2':float(error[h]),'full_logZ_error':float(pz[h]-fullz[h]),
                               'compressed_history_mass':float(originalmass[h]),'mass_abs_error':float((historymass[h]-originalmass[h]).abs()),
                               'completed_chunks_logZ_rmse':float(zrmse[h]),'completed_chunks_mass_l1':float(massl1[h])}
                            assert all(math.isfinite(r[x]) for x in METRICS);records.append(r)
                        if layer==manifest['layers'][0] and kh==0:
                            rawcount=pos+1-completed*64
                            # Tensor storage estimate only, not measured allocator/device memory.
                            pwidth=4 if name=='cluster_mass_fp32' else 2
                            size=(pos+1)*128*2*2 if name=='identity_raw' else rawcount*128*2*2+completed*32*(128*2*pwidth+4)
                            storage.append({'sample_id':sid,'query_position':pos,'bucket':'question' if pos>=saved['prefix'] else 'text','method':name,
                                            'completed_chunks':completed,'raw_visible_entries':pos+1,'new_entries':len(ck),
                                            'raw_bytes_per_kv_head':(pos+1)*128*2*2,'compressed_bytes_per_kv_head':size,
                                            'byte_ratio':(pos+1)*128*2*2/size,'metadata_excluded':True})
            print(json.dumps({'event':'layer_complete','sample':sid,'layer':layer,'seconds':time.perf_counter()-begun,'records':len(records)}),flush=True)
        dump(out/'manifest.json',meta)
    with gzip.open(out/'records.jsonl.gz','wt') as f:
        for r in records:f.write(json.dumps(r)+'\n')
    dump(out/'reference_checks.json',checks);dump(out/'compression_timings.json',timings);dump(out/'storage.json',storage)
    grouped=collections.defaultdict(list)
    for r in records:
        for scope in ['all',r['bucket'],'layer_'+str(r['layer']),'sample_'+r['sample_id'],'completed_'+str(r['completed_chunks'])]:grouped[(scope,r['method'])].append(r)
    summary=[]
    for (scope,method),rr in grouped.items():
        row={'scope':scope,'method':method,'n':len(rr)}
        for metric in METRICS:
            x=torch.tensor([r[metric] for r in rr],dtype=torch.float64)
            row[metric]={'mean':float(x.mean()),'rmse':float(x.square().mean().sqrt()),'p95_abs':float(x.abs().quantile(.95)),
                         'p99_abs':float(x.abs().quantile(.99)),'max_abs':float(x.abs().max())}
        summary.append(row)
    dump(out/'summary.json',summary)
    meta.update(status='completed',observations=len(records),reference_checks=len(checks),max_reference_relative_rms=max(c['reference_relative_rms'] for c in checks),
                records_sha256=digest(out/'records.jsonl.gz'))
    dump(out/'manifest.json',meta)
    print(json.dumps(meta),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20260907)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    try:run(args,out)
    except Exception as exc:
        dump(out/'FAILURE.json',{'status':'failed','error':repr(exc),'traceback':traceback.format_exc()});raise
if __name__=='__main__':main()
