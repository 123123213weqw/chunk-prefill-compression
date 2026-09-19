#!/usr/bin/env python3
"""Query-blind real Qwen chunk representation reconstruction; no generation claims."""
import argparse, collections, gc, hashlib, inspect, json, math, platform, time
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer, DynamicCache
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

LAYERS = [0, 9, 18, 27, 35]
CHUNKS = [7, 31, 55]

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p, obj): Path(p).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n')

def groups(k, v, labels, m):
    counts = torch.bincount(labels, minlength=m)
    assert (counts > 0).all()
    return (torch.stack([k[labels == j].mean(0) for j in range(m)]),
            torch.stack([v[labels == j].mean(0) for j in range(m)]), counts.float().log())

def cluster(k, m):
    chosen = [0]
    distances = (k-k[0]).square().sum(-1)
    for _ in range(m-1):
        distances[chosen] = -1
        chosen.append(int(distances.argmax()))
        distances = torch.minimum(distances, (k-k[chosen[-1]]).square().sum(-1))
    centers = k[chosen].clone()
    for _ in range(20):
        d = (k[:, None]-centers[None]).square().sum(-1)
        labels = d.argmin(-1)
        counts = torch.bincount(labels, minlength=m)
        for empty in torch.where(counts == 0)[0].tolist():
            candidates = torch.where(counts[labels] > 1)[0]
            point = candidates[d[candidates, labels[candidates]].argmax()]
            counts[labels[point]] -= 1
            labels[point] = empty
            counts[empty] += 1
        centers = torch.stack([k[labels == j].mean(0) for j in range(m)])
    return labels

def rel(pred, ref):
    return ((pred-ref).square().sum(-1)/ref.square().sum(-1).clamp_min(1e-20)).sqrt()

@torch.inference_mode()
def capture(args, samples, out):
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    # AutoModel omits the unused LM head; original checkpoint weights remain FP16.
    device_map = {'': 0}
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16, device_map=device_map,
                                    attn_implementation='sdpa', local_files_only=True).eval()
    original = ALL_ATTENTION_FUNCTIONS['sdpa']
    (out/'sdpa_source.txt').write_text(inspect.getsource(original))
    state = {}
    def observed(module, q, k, v, mask, **kwargs):
        result = original(module, q, k, v, mask, **kwargs)
        if module.layer_idx in LAYERS:
            offset = k.shape[-2]-q.shape[-2]
            pairs = [(p, p-offset) for p in state['positions'] if offset <= p < offset+q.shape[-2]]
            if pairs:
                indexes = [i for _, i in pairs]
                state['capture'][module.layer_idx].append({
                    'positions': [p for p, _ in pairs],
                    'q': q[0, :, indexes].detach().cpu(),
                    'output': result[0][0, indexes].transpose(0, 1).detach().cpu()})
        return result
    ALL_ATTENTION_FUNCTIONS.register('sdpa', observed)
    try:
        for s in samples:
            assert s['split'] == 'development' and s['length'] == 4096
            def encode(seg):
                ids = tok.encode(seg['text'], add_special_tokens=False)
                assert len(ids) == seg['token_count']
                assert hashlib.sha256(','.join(map(str,ids)).encode()).hexdigest() == seg['token_id_sha256']
                return ids
            parts = [encode(s['segments'][x]) for x in ['S','T','R']]
            probe = next(p for p in s['probes'] if p['probe_id'] == 'readout')
            qids = encode(probe['Q']); ids = sum(parts, []) + qids
            ns = len(parts[0]); prefix = sum(map(len, parts))
            positions = set(range(len(ids)-8, len(ids)))
            for c in CHUNKS:
                end = ns+(c+1)*64
                positions.update(end+d-1 for d in [1,16,64,256,1024] if end+d-1 < prefix)
            state.update(positions=sorted(positions), capture=collections.defaultdict(list))
            cache = DynamicCache(config=model.config)
            started = time.perf_counter()
            for offset in range(0,len(ids),256):
                block = ids[offset:offset+256]
                result = model(input_ids=torch.tensor([block], device='cuda:0'),
                               position_ids=torch.arange(offset,offset+len(block),device='cuda:0')[None],
                               past_key_values=cache, use_cache=True)
                del result
            saved = {'sample_id':s['sample_id'], 'pair_id':s['pair_id'], 'template':s['template'],
                     'member':s['member'], 'ns':ns, 'prefix':prefix, 'total':len(ids), 'layers':{}}
            for layer in LAYERS:
                blocks=state['capture'][layer]
                pos=sum([b['positions'] for b in blocks],[])
                assert pos == state['positions']
                saved['layers'][layer] = {'positions':pos, 'q':torch.cat([b['q'] for b in blocks],1),
                    'output':torch.cat([b['output'] for b in blocks],1),
                    'k':cache.layers[layer].keys[0].detach().cpu(),
                    'v':cache.layers[layer].values[0].detach().cpu()}
            torch.save(saved,out/(s['sample_id']+'.pt'))
            print(json.dumps({'event':'captured','sample':s['sample_id'],'seconds':time.perf_counter()-started}),flush=True)
            del cache,saved; state.clear(); gc.collect(); torch.cuda.empty_cache()
    finally:
        ALL_ATTENTION_FUNCTIONS.register('sdpa', original)
    del model; gc.collect(); torch.cuda.empty_cache()

@torch.inference_mode()
def analyze(args, samples, out):
    rows=[]; checks=[]; timings=[]
    for s in samples:
        saved=torch.load(out/(s['sample_id']+'.pt'),map_location='cpu',weights_only=True)
        for layer,data in saved['layers'].items():
            q,k,v=data['q'].float(),data['k'].float(),data['v'].float()
            positions=torch.tensor(data['positions']); scale=1/math.sqrt(k.shape[-1])
            # Every GQA query head is evaluated against its shared KV head.
            for kh in range(k.shape[0]):
                qq=q[kh*4:(kh+1)*4]; kk=k[kh]; vv=v[kh]
                logits=(qq@kk.T)*scale
                mask=torch.arange(len(kk))[None,:] > positions[:,None]
                logits=logits.masked_fill(mask[None],float('-inf'))
                truth=logits.softmax(-1)@vv
                reference=data['output'][kh*4:(kh+1)*4].float()
                discrepancy=float(((truth-reference).square().sum()/reference.square().sum().clamp_min(1e-20)).sqrt())
                assert discrepancy < .005, ('capture/reference mismatch',layer,kh,discrepancy)
                checks.append({'sample':s['sample_id'],'layer':layer,'kv_head':kh,'relative_rms':discrepancy})
                for c in CHUNKS:
                    start=saved['ns']+c*64; end=start+64
                    use=positions>=end
                    qp=qq[:,use]; full=truth[:,use]; pp=positions[use]
                    ll=logits[:,use]; cl=ll[:,:,start:end]
                    z=cl.logsumexp(-1); zfull=ll.logsumexp(-1)
                    local=cl.softmax(-1)@vv[start:end]
                    rest=torch.cat([ll[:,:,:start],ll[:,:,end:]],-1)
                    rv=torch.cat([vv[:start],vv[end:]])
                    rz=rest.logsumexp(-1); rout=rest.softmax(-1)@rv
                    for m in [32,16]:
                        begun=time.perf_counter()
                        clustered=groups(kk[start:end],vv[start:end],cluster(kk[start:end],m),m)
                        timings.append({'sample':s['sample_id'],'layer':layer,'head':kh,'chunk':c,'m':m,
                                        'cpu_seconds':time.perf_counter()-begun})
                        pooled=groups(kk[start:end],vv[start:end],torch.arange(64)//(64//m),m)
                        for name, params in [('mean_no_mass',(pooled[0],pooled[1],torch.zeros(m))),
                                             ('mean_mass',pooled),('cluster_mass',clustered),
                                             ('cluster_mass_fp16',(clustered[0].half().float(),clustered[1].half().float(),clustered[2]))]:
                            pk,pv,b=params
                            pl=qp@pk.T*scale+b
                            pz=pl.logsumexp(-1); py=pl.softmax(-1)@pv
                            mass=torch.sigmoid(pz-rz)
                            prediction=mass[:,:,None]*py+(1-mass[:,:,None])*rout
                            mixed=rel(prediction,full); localerr=rel(py,local)
                            for h in range(4):
                                for j,pos in enumerate(pp.tolist()):
                                    distance=pos-end+1
                                    bucket=('question' if pos>=saved['prefix'] else 'near_1_64' if distance<=64 else 'medium_65_512' if distance<=512 else 'far_513_plus')
                                    rows.append({'sample_id':s['sample_id'],'pair_id':s['pair_id'],'template':s['template'],
                                        'layer':layer,'kv_head':kh,'query_head':kh*4+h,'chunk':c,'m':m,'method':name,
                                        'query_position':pos,'distance':distance,'bucket':bucket,
                                        'mixed_relative_l2':float(mixed[h,j]),'chunk_relative_l2':float(localerr[h,j]),
                                        'chunk_logZ_error':float(pz[h,j]-z[h,j]),
                                        'original_chunk_mass':float((z[h,j]-zfull[h,j]).exp()),
                                        'mass_abs_error':float((mass[h,j]-(z[h,j]-zfull[h,j]).exp()).abs())})
        print(json.dumps({'event':'analyzed','sample':s['sample_id'],'records':len(rows)}),flush=True)
    with (out/'records.jsonl').open('w') as f:
        for r in rows: f.write(json.dumps(r)+'\n')
    dump(out/'reference_checks.json',checks);dump(out/'compression_timings.json',timings)
    grouped=collections.defaultdict(list)
    for r in rows:
        for scope in ['all','layer_'+str(r['layer']),r['bucket']]: grouped[(scope,r['method'],r['m'])].append(r)
    summary=[]
    storage=[]
    for m in [32,16]:
        raw=64*128*2*2
        for dtype,width in [('float32',4),('float16',2)]:
            prototype=m*(128*2*width+4)
            storage.append({'n':64,'m':m,'kv_dtype':dtype,'bias_dtype':'float32','raw_chunk_bytes_per_kv_head':raw,'prototype_bytes_per_kv_head':prototype,'chunk_byte_ratio':raw/prototype,'metadata_excluded':True})
    dump(out/'storage.json',storage)
    for (scope,method,m), rr in grouped.items():
        record={'scope':scope,'method':method,'m':m,'n':len(rr)}
        for metric in ['mixed_relative_l2','chunk_relative_l2','chunk_logZ_error','original_chunk_mass','mass_abs_error']:
            t=torch.tensor([r[metric] for r in rr],dtype=torch.float64)
            record[metric]={'mean':float(t.mean()),'rmse':float(t.square().mean().sqrt()),
                            'p95_abs':float(t.abs().quantile(.95)),'p99_abs':float(t.abs().quantile(.99))}
        summary.append(record)
    dump(out/'summary.json',summary)
    lines=['# Real Qwen KV representation reconstruction — development pilot','',
        'RTX 4080, FP16 model; CPU FP32 attention reconstruction. 4 development samples / 2 paired documents; not held-out evidence.',
        'Layers 0/9/18/27/35; all 8 KV and 32 query heads; T chunks 7/31/55, each 64 tokens.',
        'Each condition compresses ONE chunk, leaving all other visible KV raw. Future Q used only for scoring, never clustering.',
        'All sampled queries are after the completed chunk; full causal context includes each query token itself.',
        'Percentages below are attention vector errors, NOT answer error rates. Query/head observations are correlated; no independence CI claimed.','',
        '| Method | 64→m | Mixed mean relative L2 | Mixed p95 | Chunk-only mean relative L2 | Chunk logZ RMSE |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['scope']=='all': lines.append(f"| {r['method']} | {r['m']} | {r['mixed_relative_l2']['mean']:.2%} | {r['mixed_relative_l2']['p95_abs']:.2%} | {r['chunk_relative_l2']['mean']:.2%} | {r['chunk_logZ_error']['rmse']:.4f} |")
    lines+=['',f"Reference capture check: max FP32-reconstruction / actual-FP16-output relative RMS = {max(c['relative_rms'] for c in checks):.6f}; threshold 0.005.",
            '', '## Scope limits', '- No model weights, future queries, answers, or target evidence used to choose clusters.',
            '- cluster_mass_fp16 rounds prototype K/V to FP16 before FP32 scoring; bias stays FP32. Other methods retain FP32 prototypes.',
            '- Physical n→m compression applies only to the tested chunk, not the full 4K memory.',
            '- A small mixed error may only mean this chunk received little original attention; inspect chunk-local errors and masses.',
            '- CPU cluster timings are reference implementation cost, not optimized GPU compression overhead.',
            '- No full-model compressed decoding, readout accuracy, multilayer error propagation, or speedup has been tested.',
            '- No training, learned encoder, all-query robustness, or lossless-general-compression claim.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return {'status':'completed','observations':len(rows),'max_reference_relative_rms':max(c['relative_rms'] for c in checks)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--dataset',required=True);p.add_argument('--out',required=True);p.add_argument('--analyze-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20260907)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    samples=[json.loads(l) for l in Path(args.dataset).read_text().splitlines()]
    assert len(samples)==4 and len({s['pair_id'] for s in samples})==2
    meta={'status':'started','scope':'real-Qwen per-chunk attention reconstruction development pilot',
          'host':platform.node(),'torch':torch.__version__,'model':args.model,'dtype':'float16',
          'dataset_sha256':digest(args.dataset),'script_sha256':digest(__file__),
          'config_sha256':digest(Path(args.model)/'config.json'),'samples':[s['sample_id'] for s in samples],
          'model_device_map':'all layers GPU0', 'gpu':torch.cuda.get_device_name(0), 'prefill_block':256,
          'layers':LAYERS,'chunks':CHUNKS,'m':[32,16],'test_set_used':False}
    import transformers
    meta['transformers']=transformers.__version__;dump(out/'manifest.json',meta)
    try:
        if not args.analyze_only: capture(args,samples,out)
        meta.update(analyze(args,samples,out))
    except Exception as exc:
        meta.update(status='failed',error=repr(exc));dump(out/'manifest.json',meta)
        raise
    dump(out/'manifest.json',meta)
    print(json.dumps(meta),flush=True)
if __name__=='__main__': main()
