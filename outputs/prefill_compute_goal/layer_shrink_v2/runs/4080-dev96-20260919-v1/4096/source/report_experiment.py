#!/usr/bin/env python3
"""Independent, fail-closed audit. No model, GPU, or mutable engine imports."""
import argparse,collections,hashlib,importlib.util,json,math,statistics
from pathlib import Path
from dataset_checks import validate_development
from paired_stats import comparison,strata_scores

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
DATASET_SHA='c775d3fcaf2cc3a6f32b3c06d8b2ccec02e800d9146f4b8ad3c329c2f8e8c3c5'

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
    required={'engine.py','run_experiment.py','test_engine.py','specs.py','report_experiment.py','test_report.py','PROTOCOL.md','methods.py','development.jsonl',
              'dataset_checks.py','data_reference.py','paired_stats.py','test_matrix.py','pipeline.py','aggregate_report.py'}
    assert set(manifest['source_sha256'])==required
    for name,sha in manifest['source_sha256'].items():assert digest(p/'source'/name)==sha,name
    assert digest(p/'source/development.jsonl')==manifest['dataset_sha256']==DATASET_SHA
    all_data=[json.loads(l) for l in (p/'source/development.jsonl').read_text().splitlines()]
    validate_development(all_data)
    validation=read(p/'DATA_VALIDATION.json')
    assert validation=={'status':'PASS','samples':96,'pairs':48,'tokenizer_revalidated':True,
                        'gold_reparsed':True,'single_field_interventions_verified':True}
    assert args['length'] in (4096,8192,16384)
    data=sorted([s for s in all_data if s['length']==args['length']],key=lambda s:(s['template'],s['evidence_layout'],s['instance'],s['member']))
    n=len(data);pairs=n//2;assert n==32 and len({s['pair_id'] for s in data})==16
    ds={s['sample_id']:s for s in data};assert len(ds)==n
    assert manifest['selected_sample_ids']==list(ds) and manifest['sample_count']==n and manifest['pair_count']==pairs
    assert manifest['identity_gate_expected']==4*n
    assert manifest['gpu_at_start']['status']=='OK' and not manifest['gpu_at_start']['foreign_compute_processes']
    spec=importlib.util.spec_from_file_location('frozen_scoring',p/'source/methods.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    rows=read(p/'correctness.json');gates=read(p/'identity_gates.json');shapes=read(p/'shape_audits.json');trials=read(p/'timings.json')
    coverage(rows,('sample_id','method','probe'),((sid,m,q) for sid in ds for m in EXPECTED for q in ('readout','control')))
    coverage(gates,('sample_id','identity_method','probe'),((sid,m,q) for sid in ds for m in IDENTITIES for q in ('readout','control')))
    coverage(shapes,('sample_id','method'),((sid,m) for sid in ds for m in EXPECTED))
    timed_sids={s['sample_id'] for s in data if s['member']==0 and s['instance']==0}
    assert len(timed_sids)==4 and set(manifest['timing_sample_ids'])==timed_sids
    coverage(trials,('sample_id','method','repeat'),((sid,m,r) for sid in timed_sids for m in TIMED for r in range(-1,5)))
    lookup={(r['sample_id'],r['method'],r['probe']):r for r in rows}
    for r in rows:
        sample=ds[r['sample_id']];gold=next(q['gold'] for q in sample['probes'] if q['probe_id']==r['probe'])
        assert r['gold']==gold and r['pair_id']==sample['pair_id'] and r['template']==sample['template']
        for key in ('length','evidence_layout','instance','member'):assert r[key]==sample[key]
        assert r['status'] in ('OK','OOM')
        if r['status']=='OOM':
            assert r['correct'] is False and r['field_correct']==0 and r['fields']==len(gold) and r['error']
            assert r['answer'] is None and r['generated_ids']==[]
            continue
        for k,v in scorer.strict_score(r['answer'],gold).items():assert r[k]==v,(k,r['sample_id'],r['method'])
        assert r['generated_tokens']==len(r['generated_ids']) and 0<r['generated_tokens']<=96
        if r['hit_token_cap']:assert r['generated_tokens']==96
        if r['method'] in IDENTITIES and lookup[(r['sample_id'],'native_full',r['probe'])]['status']=='OK':assert r['generated_ids']==lookup[(r['sample_id'],'native_full',r['probe'])]['generated_ids']
    for g in gates:
        assert g['generated_tokens_equal'] is True
        for key,limit in (('max_cache_relative_rms',.005),('logit_relative_rms',.005),('first_token_kl',.001)):
            assert math.isfinite(g[key]) and g[key]<limit
            assert g[key]>=(-1e-6 if key=='first_token_kl' else 0)
    scores={};expected_summary={}
    for m in EXPECTED:
        subset=[r for r in rows if r['method']==m];rr=[r for r in subset if r['probe']=='readout'];cc=[r for r in subset if r['probe']=='control']
        pair_flags=collections.defaultdict(list)
        for r in rr:pair_flags[r['pair_id']].append(r['correct'])
        scores[m]={'readout_correct':sum(r['correct'] for r in rr),'control_correct':sum(r['correct'] for r in cc),'n':n,
                   'field_correct':sum(r['field_correct'] for r in rr),'fields':sum(r['fields'] for r in rr),
                   'pairs_both_correct':sum(all(v) for v in pair_flags.values()),'pairs':pairs,
                   'unavailable':sum(r['status']!='OK' for r in subset),
                   'format_failures':sum(r['status']=='OK' and not r['format_ok'] for r in subset),'token_caps':sum(r['hit_token_cap'] for r in subset),
                   'templates':{t:{'correct':sum(r['correct'] for r in rr if r['template']==t),'n':sum(r['template']==t for r in rr)} for t in sorted({r['template'] for r in rr})}}
        expected_summary[m]={q:{'correct':sum(r['correct'] for r in subset if r['probe']==q),'n':n,
                                'unavailable':sum(r['status']!='OK' for r in subset if r['probe']==q),
                                'field_correct':sum(r['field_correct'] for r in subset if r['probe']==q),
                                'fields':sum(r['fields'] for r in subset if r['probe']==q)} for q in ('readout','control')}
    assert read(p/'correctness_summary.json')==expected_summary
    shape_lookup={(r['sample_id'],r['method']):r for r in shapes};ratios={m:[] for m in EXPECTED}
    for r in shapes:
        sid,m=r['sample_id'],r['method'];_,depth,keep,_=EXPECTED[m]
        assert r['status'] in ('OK','OOM')
        if r['status']=='OOM':
            assert r['error'] and all(lookup[(sid,m,q)]['status']=='OOM' for q in ('readout','control'))
            continue
        lengths={k:ds[sid]['segments'][k]['token_count'] for k in ('S','T','R')}
        assert r['segment_lengths']==lengths and lengths['T']%64==0
        full=sum(lengths.values());upper=full-lengths['T']+lengths['T']//64*keep
        assert (r['split_depth'],r['keep'],r['lower_layer_tokens'],r['upper_layer_tokens'])==(depth,keep,full,upper)
        expected_counts={(i,op):(full if i<depth else upper) for i in range(36) for op in OPS}
        counts=collections.Counter();flops=0;dims={}
        for h in r['actual_module_inputs']:
            assert isinstance(h['calls'],int) and h['calls']>0
            key=(h['layer'],h['op']);assert key in expected_counts and h['tokens']>0
            counts[key]+=h['tokens'];dim=(h['in_features'],h['out_features']);assert min(dim)>0
            if key in dims:assert dims[key]==dim
            dims[key]=dim;flops+=2*h['tokens']*dim[0]*dim[1]
        assert dict(counts)==expected_counts and flops==r['linear_flops']
        base=shape_lookup[(sid,'native_full')]
        if base['status']=='OK':
            base_dims={(h['layer'],h['op']):(h['in_features'],h['out_features']) for h in base['actual_module_inputs']}
            assert dims==base_dims
        expected_full_flops=sum(2*full*dim[0]*dim[1] for dim in dims.values())
        ratio=(depth*full+(36-depth)*upper)/(36*full)
        close(r['linear_token_layer_ratio'],ratio);close(flops/expected_full_flops,ratio);ratios[m].append(ratio)
        layer_lengths=[full]*depth+[upper]*(36-depth)
        cache_bytes=sum(n*dims[(i,'k_proj')][1]*2*2 for i,n in enumerate(layer_lengths))
        for q in ('readout','control'):
            row=lookup[(sid,m,q)]
            if row['status']!='OK':continue
            assert row['layer_lengths']==layer_lengths
            assert row['cache_bytes']==cache_bytes and row['position_metadata_bytes']==8*sum(layer_lengths)
    for sid in ds:
        a=shape_lookup[(sid,'mean_d24_k32')];b=shape_lookup[(sid,'mean_d12_k48')]
        if a['status']==b['status']=='OK':assert a['linear_flops']==b['linear_flops']
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
    return {'status':'PASS','sample_count':n,'pair_count':pairs,'length':args['length'],
            'paired_comparisons':{m:comparison(rows,m) for m in CANDIDATES},'stratified_scores':strata_scores(rows),
            'correctness_records':len(rows),'identity_gates':len(gates),'shape_audits':len(shapes),'timing_trials':len(trials),
            'source_hashes_verified':True,'dataset_gold_verified':True,'strict_scores_recomputed':True,'shape_flops_recomputed':True,
            'timing_summaries_recomputed':True,'all_trial_boundaries_isolated':observed_isolation,
            'performance_scope':'boundary-observed isolation; residual interference possible' if observed_isolation else 'non-isolated or unknown; diagnostic timing only',
            'scores':scores,'linear_flop_ratios':ratios,'timing_summary':summary,'comparisons':comparisons}

def render(a):
    s=a['scores'];n=a['sample_count'];pairs=a['pair_count']
    lines=[f"# 深度 / 压缩率：{a['length']//1024}K 完整开发阶段",'',
           f"固定 {n} 条 / {pairs} 对，覆盖两模板、两布局、每格四对。已有开发数据，不是独立盲测。",'',
           '## 正确性（OOM 包含在总分母，另外报告）','',
           '| 方法 | readout | control | 字段命中 | 两成员均对 | 格式错 / 封顶 / OOM |',
           '|---|---:|---:|---:|---:|---:|']
    for m,v in s.items():
        lines.append(f"| {m} | {v['readout_correct']}/{n} | {v['control_correct']}/{n} | {v['field_correct']}/{v['fields']} | {v['pairs_both_correct']}/{pairs} | {v['format_failures']} / {v['token_caps']} / {v['unavailable']} |")
    lines+=['','## 相对 Full 的损伤','',
            '| 方法 | Full 对→方法错 | Full 错→方法对 | 准确率差 (百分点) | 95% 配对 CI (百分点) |',
            '|---|---:|---:|---:|---|']
    for m,c in a['paired_comparisons'].items():
        lo,hi=c['ci95']
        lines.append(f"| {m} | {c['full_correct_to_method_wrong']} | {c['full_wrong_to_method_correct']} | {100*c['method_minus_full']:.2f} | [{100*lo:.2f}, {100*hi:.2f}] |")
    lines+=['2000 次分层 pair-cluster bootstrap，反事实成员不拆开；开发集描述性 CI，不是正式多重检验或无损保证。',
            '若存在 OOM，它计入未成功，同时单列；不得由只保留成功运行的分母夸大准确率。',
            '', '## 读取分层','', '| 方法 | 模板 | 布局 | 正确 | OOM |','|---|---|---|---:|---:|']
    for r in a['stratified_scores']:
        if r['probe']=='readout':lines.append(f"| {r['method']} | {r['template']} | {r['layout']} | {r['correct']}/{r['n']} | {r['unavailable']} |")
    lines+=['','## 实际计算量','',f"{a['identity_gates']} 项 identity 检查；{a['shape_audits']} 组逐层算子形状记录。"]
    for m in CANDIDATES:
        rr=a['linear_flop_ratios'][m]
        lines.append(f"- {m}：成功形状记录 {len(rr)}/{n}；线性 FLOPs / Full " + (f"{min(rr):.4%}–{max(rr):.4%}。" if rr else '不可评估。'))
    lines+=['只统计七种线性算子，不等同于完整 attention FLOPs 或时延。','', '## 独立计时','']
    lines.append('所有 trial 边界未观察到其他 compute PID（不排除采样间干扰）。' if a['all_trial_boundaries_isolated'] else '**存在并发或未知遥测，以下仅为诊断值，不作可靠提速结论。**')
    lines+=['4 个预先固定代表文档（每模板/布局的 instance 0、member 0），1 warmup + 5 repeats；未按正确性筛选。',
            '每列分别取中位数；总 TTFT 中位数不一定等于两个分段中位数相加。',
            '', '| 样本 | 方法 | Prefill ms | Q 首 token ms | 总 TTFT ms | Q 后 KV MiB |', '|---|---|---:|---:|---:|---:|']
    for r in a['timing_summary']:
        if r['successful_trials']==5:
            lines.append(f"| {r['sample_id'][:8]} | {r['method']} | {r['prefix_seconds']*1000:.2f} | {r['query_ttft_seconds']*1000:.2f} | {r['total_ttft_seconds']*1000:.2f} | {r['kv_bytes_after_Q']/2**20:.2f} |")
        else:lines.append(f"| {r['sample_id'][:8]} | {r['method']} | {r['successful_trials']}/5 成功，{r['oom_trials']} OOM | — | — | — |")
    lines+=['','### 与同轮较快原生基线比较','','| 样本 | 方法 | 原生基线 | Prefill 时间减少 | 速度比 |','|---|---|---|---:|---:|']
    for r in a['comparisons']:
        lines.append(f"| {r['sample_id'][:8]} | {r['method']} | {r['baseline']} | {r['prefix_time_reduction']:.2%} | {r['prefix_speed_ratio']:.3f}× |")
    lines+=['','独立审计 PASS 只表示记录一致。每格仅四对，未达到普遍无损/鲁棒的证据要求；失败样本和 Full 原生能力限制必须保留。']
    return '\n'.join(lines)+'\n'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();p=Path(args.directory)
    a=audit(p);report=render(a)
    (p/'AUDIT.json').write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n')
    (p/'REPORT_ZH.md').write_text(report);print(report)
if __name__=='__main__':main()
