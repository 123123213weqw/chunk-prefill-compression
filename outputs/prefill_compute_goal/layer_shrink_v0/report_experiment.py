#!/usr/bin/env python3
"""Independent audit + report for the identity-gated 4K development pilot."""
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory');args=parser.parse_args();p=Path(args.directory)
    manifest=read(p/'manifest.json');assert manifest['status']=='completed'
    for name,sha in manifest['source_sha256'].items():assert digest(p/'source'/name)==sha,name
    assert manifest['test_used'] is False
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_scoring',p/'source/methods.py')
    scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)
    correctness=read(p/'correctness.json');gates=read(p/'identity_gates.json');shapes=read(p/'shape_audits.json');trials=read(p/'timings.json')
    assert len(correctness)==32 and len(gates)==8 and len(shapes)==16 and len(trials)==60
    for r in correctness:
        result=scorer.strict_score(r['answer'],r['gold'])
        for k,v in result.items():assert r[k]==v,(k,r)
    for g in gates:assert g['generated_tokens_equal'] and g['max_cache_relative_rms']<.005 and g['logit_relative_rms']<.005 and g['first_token_kl']<.001
    byshape={(r['sample_id'],r['method']):r for r in shapes}
    ratios=[]
    for r in shapes:
        sums=collections.Counter();flops=0
        for h in r['actual_module_inputs']:
            sums[(h['layer'],h['op'])]+=h['tokens'];flops+=2*h['tokens']*h['in_features']*h['out_features']
        assert flops==r['linear_flops']
        for (layer,op),tokens in sums.items():assert tokens==(r['lower_layer_tokens'] if layer<12 else r['upper_layer_tokens'])
        assert len(sums)==36*7
        if r['method']=='pair_mean_d12':ratios.append(r['linear_flops']/byshape[(r['sample_id'],'native_full')]['linear_flops'])
    groups=collections.defaultdict(list)
    for r in trials:
        if not r['warmup']:groups[(r['sample_id'],r['method'])].append(r)
        if r['status']=='OK':
            for k in ('prefix_seconds','query_ttft_seconds','total_ttft_seconds','peak_allocated_bytes'):
                assert math.isfinite(r[k]) and r[k]>0
            assert abs(r['prefix_seconds']+r['query_ttft_seconds']-r['total_ttft_seconds'])<1e-10
    assert len(groups)==10 and all(len(v)==5 for v in groups.values())
    summary=[]
    for (sid,method),rr in sorted(groups.items()):
        good=[r for r in rr if r['status']=='OK'];item={'sample_id':sid,'method':method,'successful_trials':len(good),'oom_trials':len(rr)-len(good)}
        for metric in ['prefix_seconds','query_ttft_seconds','total_ttft_seconds','kv_bytes_after_Q','peak_allocated_bytes','peak_allocated_over_model_bytes']:
            if good:item[metric]=statistics.median(r[metric] for r in good)
        summary.append(item)
    # Compare against both native block sizes; select faster native per sample,
    # never use the slower layerwise identity to headline speed improvements.
    lookup={(r['sample_id'],r['method']):r for r in summary};comparisons=[]
    for sid in sorted({r['sample_id'] for r in summary}):
        baselines=[lookup[(sid,m)] for m in ('native_b1024','native_b4096')]
        if any(b['successful_trials']!=5 for b in baselines):continue
        base=min(baselines,key=lambda b:b['prefix_seconds'])
        for method in ('layerwise_identity','pair_mean_d12','endpoint_d12'):
            c=lookup[(sid,method)]
            if c['successful_trials']!=5:continue
            comparisons.append({'sample_id':sid,'method':method,'baseline':base['method'],
                                'prefix_time_reduction':1-c['prefix_seconds']/base['prefix_seconds'],
                                'prefix_speed_ratio':base['prefix_seconds']/c['prefix_seconds'],
                                'total_ttft_time_reduction_same_baseline':1-c['total_ttft_seconds']/base['total_ttft_seconds'],
                                'kv_byte_reduction':1-c['kv_bytes_after_Q']/base['kv_bytes_after_Q']})
    lines=['# 层间缩短序列：真实 Qwen 4K 开发实验','',
           'RTX 4080 / Qwen3-4B-Instruct-2507 / FP16。4 条开发样本（2 个反事实样本对），不是正式泛化测试。',
           '前 12 层完整计算；T 每个 64-token chunk 在层间变成 32 个表示，后 24 层真正少算 QKV/MLP。S/R/Q 保留。',
           'pair_mean 是相邻 hidden states 均值，不是上一轮 post-RoPE KV 聚类；endpoint 是同预算直接保留组末位置。',
           '', '## 结论', '',
           '本轮已验证实际减少 prefill 运算并缩短耗时，但固定 d12、64→32 的无训练均值合并未保住全部字段，因此是“有计算收益、保真不足”的实验结果，不是可部署成功。',
           '均值合并的 prefill 时间相对两种原生基线中较快者减少约 32.8%–32.9%（约 1.49×）；readout 从 Full 的 4/4 降为 2/4，control 保持 4/4。',
           '均值合并通过两个延迟模板成员、失败于两个库存模板成员；端点保留恰好相反。这两对样本不足以判断普遍优劣。失败输出为合法 JSON，但数字有误，不是仅输出格式不符。',
           '', '## 1. 原生对齐检查', '',
           f"8 个 probe 的原生 / layerwise identity 生成 token 序列全部一致；最大缓存相对 RMS {max(g['max_cache_relative_rms'] for g in gates):.6g}、最大首次 logits 相对 RMS {max(g['logit_relative_rms'] for g in gates):.6g}。",
           '', '## 2. 读取正确性', '', '| 方法 | readout 正确 | control 正确 | 读取字段命中 |','|---|---:|---:|---:|']
    scores={}
    for method in ('native_full','layerwise_identity','pair_mean_d12','endpoint_d12'):
        rr=[r for r in correctness if r['method']==method and r['probe']=='readout'];cc=[r for r in correctness if r['method']==method and r['probe']=='control']
        scores[method]={'readout_correct':sum(r['correct'] for r in rr),'control_correct':sum(r['correct'] for r in cc),'n':4}
        lines.append(f"| {method} | {sum(r['correct'] for r in rr)}/4 | {sum(r['correct'] for r in cc)}/4 | {sum(r['field_correct'] for r in rr)}/{sum(r['fields'] for r in rr)} |")
    lines+=['','## 3. 真实模块输入与计算量','',
            f"模块输入钩子验证 36 层的 Q/K/V/O、gate/up/down 共七个线性算子的输入行数。pair_mean 的线性 FLOPs 为 Full 的 {min(ratios):.2%}–{max(ratios):.2%}。这是运行时形状推算的线性 FLOPs，不包含 attention QK/AV、非线性和压缩开销。",
            '', '## 4. Prefill 与首次 token 延迟', '',
            '正确性与性能分开运行；无 shape hooks 的计时、1 warmup + 5 repeats、方法顺序随机。每对仅取 member 0，两个计时样本；下表每项均为 5 次测量中位数。',
            '原生性能基线使用未修改 HF model.model，不附加自定义位置追踪；提供 b1024、b4096 两种原生基线。计入合并、mask、位置、输入搬移等开销；不计模型加载。不代表完整答案延迟。',
            '', '| 样本 | 方法 | Prefill ms | Q→首 token ms | 总 TTFT ms | Q 后 KV MiB | 峰值 allocated MiB |', '|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        if r['successful_trials']==5:
            lines.append(f"| {r['sample_id'][:8]} | {r['method']} | {r['prefix_seconds']*1000:.2f} | {r['query_ttft_seconds']*1000:.2f} | {r['total_ttft_seconds']*1000:.2f} | {r['kv_bytes_after_Q']/2**20:.2f} | {r['peak_allocated_bytes']/2**20:.2f} |")
        else:lines.append(f"| {r['sample_id'][:8]} | {r['method']} | 成功 {r['successful_trials']}/5；OOM {r['oom_trials']} | — | — | — | — |")
    lines+=['','### 相对较快原生 prefill 基线','','| 样本 | 方法 | 基线 | Prefill 时间减少 | 速度比 | 同基线总 TTFT 时间减少 |','|---|---|---|---:|---:|---:|']
    for r in comparisons:
        lines.append(f"| {r['sample_id'][:8]} | {r['method']} | {r['baseline']} | {r['prefix_time_reduction']:.2%} | {r['prefix_speed_ratio']:.3f}× | {r['total_ttft_time_reduction_same_baseline']:.2%} |")
    lines+=['','## 5. 边界','',
            '- 只有 2 个独立样本对。不得由 4/4 宣称普遍无损，也不能把多次计时当作更多独立文档。',
            '- 少算线性 FLOPs 与实际时间必须分开判断；若字段正确性下降，不能将速度收益包装成可用方法。',
            '- 本轮只测 4K、固定 d12、固定 64→32、无训练。没有长文本扩展、其他比例/深度或跨任务保证。',
            '- 不使用或恢复 Goal；此轮按用户普通请求执行。']
    (p/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    audit={'status':'PASS','correctness_records':len(correctness),'identity_gates':len(gates),'shape_audits':len(shapes),'timing_trials':len(trials),
           'source_hashes_verified':True,'strict_scores_recomputed':True,'shape_flops_recomputed':True,'timing_medians_recomputed':True,
           'scores':scores,'linear_flop_ratios':ratios,'comparisons':comparisons}
    (p/'AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
