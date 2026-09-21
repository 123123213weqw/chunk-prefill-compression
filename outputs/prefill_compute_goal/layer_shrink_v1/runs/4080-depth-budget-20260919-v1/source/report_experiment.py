#!/usr/bin/env python3
"""Independent, fail-closed audit. No model, GPU, or mutable engine imports."""
import argparse,collections,hashlib,importlib.util,json,math,statistics
from pathlib import Path

EXPECTED={
    'native_full':('identity',12,64,True),
    'identity_d12':('identity',12,64,False),
    'identity_d24':('identity',24,64,False),
    'mean_d12_k32':('pair_mean',12,32,False),
    'mean_d24_k32':('pair_mean',24,32,False),
    'mean_d12_k48':('pair_mean_48',12,48,False),
}
IDENTITIES=('identity_d12','identity_d24')
CANDIDATES=('mean_d12_k32','mean_d24_k32','mean_d12_k48')
TIMED=('native_b1024','native_b4096',*IDENTITIES,*CANDIDATES)
OPS=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj')
METRICS=('prefix_seconds','query_ttft_seconds','total_ttft_seconds','kv_bytes_after_Q','peak_allocated_bytes','peak_allocated_over_model_bytes')
DATASET_SHA='b5a29f56aab0e0637af8fd74ecfc0442de112a68f200bec8fa6cf69a40fd6a45'

def read(p):return json.loads(p.read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12),(a,b)
def coverage(rows,keys,expected):
    observed=[tuple(r[k] for k in keys) for r in rows]
    assert len(observed)==len(set(observed)),'duplicate records'
    assert set(observed)==set(expected),'missing or unexpected records'

def audit(p):
    p=Path(p);manifest=read(p/'manifest.json');assert manifest['status']=='completed'
    assert manifest['test_used'] is False and manifest['shape_hooks_excluded_from_timing'] is True
    args=manifest['args'];assert args['block']==1024 and args['max_tokens']==96 and args['warmups']==1 and args['repeats']==5
    assert manifest['method_specs']=={m:dict(zip(('method','split_depth','keep','native'),v)) for m,v in EXPECTED.items()}
    assert manifest['timing_methods']==list(TIMED)
    required={'engine.py','run_experiment.py','test_engine.py','specs.py','report_experiment.py','test_report.py','PROTOCOL.md','methods.py','smoke4.jsonl'}
    assert set(manifest['source_sha256'])==required
    for name,sha in manifest['source_sha256'].items():assert digest(p/'source'/name)==sha,name
    assert digest(p/'source/smoke4.jsonl')==manifest['dataset_sha256']==DATASET_SHA
    data=[json.loads(l) for l in (p/'source/smoke4.jsonl').read_text().splitlines()]
    assert len(data)==4 and len({s['pair_id'] for s in data})==2
    assert all(s['split']=='development' and s['length']==4096 for s in data)
    ds={s['sample_id']:s for s in data};assert len(ds)==4
    for pair in {s['pair_id'] for s in data}:assert sorted(s['member'] for s in data if s['pair_id']==pair)==[0,1]
    spec=importlib.util.spec_from_file_location('frozen_scoring',p/'source/methods.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    rows=read(p/'correctness.json');gates=read(p/'identity_gates.json');shapes=read(p/'shape_audits.json');trials=read(p/'timings.json')
    coverage(rows,('sample_id','method','probe'),((sid,m,q) for sid in ds for m in EXPECTED for q in ('readout','control')))
    coverage(gates,('sample_id','identity_method','probe'),((sid,m,q) for sid in ds for m in IDENTITIES for q in ('readout','control')))
    coverage(shapes,('sample_id','method'),((sid,m) for sid in ds for m in EXPECTED))
    timed_sids={s['sample_id'] for s in data if s['member']==0}
    coverage(trials,('sample_id','method','repeat'),((sid,m,r) for sid in timed_sids for m in TIMED for r in range(-1,5)))
    lookup={(r['sample_id'],r['method'],r['probe']):r for r in rows}
    for r in rows:
        sample=ds[r['sample_id']];gold=next(q['gold'] for q in sample['probes'] if q['probe_id']==r['probe'])
        assert r['gold']==gold and r['pair_id']==sample['pair_id'] and r['template']==sample['template']
        for k,v in scorer.strict_score(r['answer'],gold).items():assert r[k]==v,(k,r['sample_id'],r['method'])
        assert r['generated_tokens']==len(r['generated_ids']) and 0<r['generated_tokens']<=96
        if r['hit_token_cap']:assert r['generated_tokens']==96
        if r['method'] in IDENTITIES:assert r['generated_ids']==lookup[(r['sample_id'],'native_full',r['probe'])]['generated_ids']
    for g in gates:
        assert g['generated_tokens_equal'] is True
        for key,limit in (('max_cache_relative_rms',.005),('logit_relative_rms',.005),('first_token_kl',.001)):
            assert math.isfinite(g[key]) and g[key]<limit
            assert g[key]>=(-1e-6 if key=='first_token_kl' else 0)
    scores={};expected_summary={}
    for m in EXPECTED:
        subset=[r for r in rows if r['method']==m];rr=[r for r in subset if r['probe']=='readout'];cc=[r for r in subset if r['probe']=='control']
        pairs=collections.defaultdict(list)
        for r in rr:pairs[r['pair_id']].append(r['correct'])
        scores[m]={'readout_correct':sum(r['correct'] for r in rr),'control_correct':sum(r['correct'] for r in cc),'n':4,
                   'field_correct':sum(r['field_correct'] for r in rr),'fields':sum(r['fields'] for r in rr),
                   'pairs_both_correct':sum(all(v) for v in pairs.values()),'pairs':2,
                   'format_failures':sum(not r['format_ok'] for r in subset),'token_caps':sum(r['hit_token_cap'] for r in subset),
                   'templates':{t:{'correct':sum(r['correct'] for r in rr if r['template']==t),'n':sum(r['template']==t for r in rr)} for t in sorted({r['template'] for r in rr})}}
        expected_summary[m]={q:{'correct':sum(r['correct'] for r in subset if r['probe']==q),'n':4,
                                'field_correct':sum(r['field_correct'] for r in subset if r['probe']==q),
                                'fields':sum(r['fields'] for r in subset if r['probe']==q)} for q in ('readout','control')}
    assert read(p/'correctness_summary.json')==expected_summary
    shape_lookup={(r['sample_id'],r['method']):r for r in shapes};ratios={m:[] for m in EXPECTED}
    for r in shapes:
        sid,m=r['sample_id'],r['method'];_,depth,keep,_=EXPECTED[m]
        lengths={k:ds[sid]['segments'][k]['token_count'] for k in ('S','T','R')}
        assert r['segment_lengths']==lengths and lengths['T']%64==0
        full=sum(lengths.values());upper=full-lengths['T']+lengths['T']//64*keep
        assert (r['split_depth'],r['keep'],r['lower_layer_tokens'],r['upper_layer_tokens'])==(depth,keep,full,upper)
        expected_counts={(i,op):(full if i<depth else upper) for i in range(36) for op in OPS}
        counts=collections.Counter();flops=0;dims={}
        for h in r['actual_module_inputs']:
            key=(h['layer'],h['op']);assert key in expected_counts and h['tokens']>0
            counts[key]+=h['tokens'];dim=(h['in_features'],h['out_features']);assert min(dim)>0
            if key in dims:assert dims[key]==dim
            dims[key]=dim;flops+=2*h['tokens']*dim[0]*dim[1]
        assert dict(counts)==expected_counts and flops==r['linear_flops']
        base=shape_lookup[(sid,'native_full')]
        base_dims={(h['layer'],h['op']):(h['in_features'],h['out_features']) for h in base['actual_module_inputs']}
        assert dims==base_dims
        ratio=(depth*full+(36-depth)*upper)/(36*full)
        close(r['linear_token_layer_ratio'],ratio);close(flops/base['linear_flops'],ratio);ratios[m].append(ratio)
        layer_lengths=[full]*depth+[upper]*(36-depth)
        cache_bytes=sum(n*dims[(i,'k_proj')][1]*2*2 for i,n in enumerate(layer_lengths))
        for q in ('readout','control'):
            row=lookup[(sid,m,q)];assert row['layer_lengths']==layer_lengths
            assert row['cache_bytes']==cache_bytes and row['position_metadata_bytes']==8*sum(layer_lengths)
    assert ratios['mean_d24_k32']==ratios['mean_d12_k48']
    for r in trials:
        assert r['warmup']==(r['repeat']<0) and r['status'] in ('OK','OOM')
        isolated=all(t['status']=='OK' and not t['foreign_compute_processes'] for t in (r['gpu_before'],r['gpu_after']))
        assert r['isolation_observed']==isolated
        if r['status']=='OK':
            for metric in METRICS:assert math.isfinite(r[metric]) and r[metric]>0
            close(r['prefix_seconds']+r['query_ttft_seconds'],r['total_ttft_seconds'])
    observed_isolation=all(r['isolation_observed'] for r in trials)
    summary=[];expected_perf=[]
    for sid in sorted(timed_sids):
        for m in TIMED:
            subset=[r for r in trials if r['sample_id']==sid and r['method']==m and not r['warmup']]
            good=[r for r in subset if r['status']=='OK']
            item={'sample_id':sid,'method':m,'successful_trials':len(good),'oom_trials':5-len(good)}
            ep={'sample_id':sid,'method':m,'measured':len(good),'oom':5-len(good)}
            for metric in METRICS:
                if good:
                    values=[r[metric] for r in good];item[metric]=statistics.median(values)
                    ep[metric]={'median':statistics.median(values),'min':min(values),'max':max(values)}
            summary.append(item);expected_perf.append(ep)
    assert read(p/'performance_summary.json')==expected_perf
    perf={(r['sample_id'],r['method']):r for r in summary};comparisons=[]
    for sid in sorted(timed_sids):
        bases=[perf[(sid,m)] for m in ('native_b1024','native_b4096')]
        if any(b['successful_trials']!=5 for b in bases):continue
        base=min(bases,key=lambda r:r['prefix_seconds'])
        for m in CANDIDATES:
            c=perf[(sid,m)]
            if c['successful_trials']!=5:continue
            comparisons.append({'sample_id':sid,'method':m,'baseline':base['method'],
                                'prefix_time_reduction':1-c['prefix_seconds']/base['prefix_seconds'],
                                'prefix_speed_ratio':base['prefix_seconds']/c['prefix_seconds'],
                                'total_ttft_time_reduction':1-c['total_ttft_seconds']/base['total_ttft_seconds'],
                                'isolated_run_observed':observed_isolation})
    return {'status':'PASS','correctness_records':len(rows),'identity_gates':len(gates),'shape_audits':len(shapes),'timing_trials':len(trials),
            'source_hashes_verified':True,'dataset_gold_verified':True,'strict_scores_recomputed':True,'shape_flops_recomputed':True,
            'timing_summaries_recomputed':True,'all_trial_boundaries_isolated':observed_isolation,
            'performance_scope':'boundary-observed isolation; residual interference possible' if observed_isolation else 'non-isolated or unknown; diagnostic timing only',
            'scores':scores,'linear_flop_ratios':ratios,'timing_summary':summary,'comparisons':comparisons}

def render(a):
    s=a['scores'];lines=['# 深度 / 压缩率对照：4K 开发预实验','',
        'Qwen3-4B-Instruct-2507 / FP16 / RTX 4080。固定 4 条旧开发样本、2 对，不是盲测或泛化结论。','',
        '## 读取正确性','', '| 方法 | readout | control | 字段命中 | 两成员均正确 | 格式失败 / 封顶 |', '|---|---:|---:|---:|---:|---:|']
    for m,v in s.items():lines.append(f"| {m} | {v['readout_correct']}/4 | {v['control_correct']}/4 | {v['field_correct']}/{v['fields']} | {v['pairs_both_correct']}/2 | {v['format_failures']} / {v['token_caps']} |")
    lines+=['','## 结论边界','',f"同轮原生 Full readout {s['native_full']['readout_correct']}/4，旧方法 d12/k32 {s['mean_d12_k32']['readout_correct']}/4。",
        f"晚压缩 d24/k32 {s['mean_d24_k32']['readout_correct']}/4；轻压缩 d12/k48 {s['mean_d12_k48']['readout_correct']}/4。"]
    if s['native_full']['readout_correct']!=4:lines.append('**Full 未全对，当前数据不可直接用于宣称压缩保真改善。**')
    for m in ('mean_d24_k32','mean_d12_k48'):
        v=s[m];eligible=v['readout_correct']==4 and v['control_correct']==4 and s['native_full']['readout_correct']==4
        lines.append(f"- {m}："+('仅达到扩大开发评测的门槛，不证明鲁棒或无损。' if eligible else '未达到 readout/control 全对的扩大评测门槛；不能称保真问题已解决。'))
    lines+=['','## 计算量审计','',f"通过 {a['identity_gates']} 个 identity probe 门槛、{a['shape_audits']} 组实际形状审计。七种线性算子均按实际输入行数计账。"]
    for m in CANDIDATES:
        rr=a['linear_flop_ratios'][m];lines.append(f"- {m}：线性 FLOPs 为 Full 的 {min(rr):.4%}–{max(rr):.4%}。")
    lines+=['两个新条件线性计算预算相同，但 attention、合并开销及信息损失不同。不是总 FLOPs 的完整统计。',
            '', '## 计时（1 warmup + 5 repeats / 文档 / 方法）','']
    lines.append('**所有 trial 边界未观察到其他 compute PID；仍不能排除采样间干扰。**' if a['all_trial_boundaries_isolated'] else '**观察到其他 GPU compute 任务或遥测不可用：以下计时仅为诊断值，不用于可靠的提速结论。**')
    lines+=['每对仅 member 0，共 2 个计时文档；重复不是独立样本。含在线合并等开销，不计模型加载、不计完整答案生成。',
            '', '| 样本 | 方法 | Prefill ms | Q 首 token ms | 总 TTFT ms | Q 后 KV MiB | 峰值 allocated MiB |','|---|---|---:|---:|---:|---:|---:|']
    for r in a['timing_summary']:
        if r['successful_trials']==5:lines.append(f"| {r['sample_id'][:8]} | {r['method']} | {r['prefix_seconds']*1000:.2f} | {r['query_ttft_seconds']*1000:.2f} | {r['total_ttft_seconds']*1000:.2f} | {r['kv_bytes_after_Q']/2**20:.2f} | {r['peak_allocated_bytes']/2**20:.2f} |")
        else:lines.append(f"| {r['sample_id'][:8]} | {r['method']} | {r['successful_trials']}/5 成功，{r['oom_trials']} OOM | — | — | — | — |")
    lines+=['','## 样本与模板诊断','']
    for m in CANDIDATES:lines.append(f"- {m}："+'；'.join(f"{t} {v['correct']}/{v['n']}" for t,v in s[m]['templates'].items()))
    lines+=['','源代码、协议、数据、逐条答案和运行环境见本目录快照及 JSON。独立审计 PASS 表示记录一致，不代表压缩方法有效。']
    return '\n'.join(lines)+'\n'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();p=Path(args.directory)
    a=audit(p);report=render(a)
    (p/'AUDIT.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
    (p/'REPORT_ZH.md').write_text(report);print(report)
if __name__=='__main__':main()
