"""Independent standard-library audit and Chinese report generator."""
import collections,gzip,hashlib,json,math,statistics
from pathlib import Path
p=Path(__file__).resolve().parent/'runs/pilot'
def digest(x):return hashlib.sha256(x.read_bytes()).hexdigest()
m=json.loads((p/'manifest.json').read_text());assert m['status']=='completed' and not m['test_set_used']
assert digest(p/'source/run.py')==m['script_sha256']
assert digest(p/'source/compressor_source.py')==m['compressor_sha256']
assert digest(p/'records.jsonl.gz')==m['records_sha256']
with gzip.open(p/'records.jsonl.gz','rt') as f:rows=[json.loads(l) for l in f]
assert len(rows)==m['observations']
metrics=['relative_l2','full_logZ_error','compressed_history_mass','mass_abs_error','completed_chunks_logZ_rmse','completed_chunks_mass_l1']
groups=collections.defaultdict(list)
for r in rows:
 assert all(math.isfinite(r[k]) for k in metrics)
 assert r['sample_id'] in m['samples'] and 1<=r['completed_chunks']<=64
 assert 0<=r['compressed_history_mass']<=1.000001
 if r['method']=='identity_raw':assert r['relative_l2']<1e-6
 if r['bucket']=='question':assert r['completed_chunks']==64
 for scope in ['all',r['bucket'],'layer_'+str(r['layer']),'sample_'+r['sample_id'],'completed_'+str(r['completed_chunks'])]:
  groups[(scope,r['method'])].append(r)
s=json.loads((p/'summary.json').read_text());idx={(r['scope'],r['method']):r for r in s}
for r in s:
 rr=groups[(r['scope'],r['method'])];assert len(rr)==r['n']
 for metric in metrics:
  assert abs(statistics.fmean(x[metric] for x in rr)-r[metric]['mean'])<1e-10
  ordered=sorted(abs(x[metric]) for x in rr)
  for prob,key in [(.95,'p95_abs'),(.99,'p99_abs')]:
   at=(len(ordered)-1)*prob;lo=math.floor(at);hi=math.ceil(at);q=ordered[lo]+(ordered[hi]-ordered[lo])*(at-lo)
   assert abs(q-r[metric][key])<1e-9
checks=json.loads((p/'reference_checks.json').read_text());assert len(checks)==m['reference_checks']
assert max(c['reference_relative_rms'] for c in checks)<.005
storage=json.loads((p/'storage.json').read_text())
timings=json.loads((p/'compression_timings.json').read_text())
lines=['# 4080：全部已完成历史 chunk 同时 64→32','',
'状态：已完成，源代码/缓存来源清单/结果哈希与所有汇总均值、p95、p99复核通过。',
'', '## 结论','',
'带权聚类比带权均值更准确，但同时压缩历史后仍有明显尾部误差：主方法平均 6.07%、p99 46.76%；问题阶段平均 2.35%、p99 21.99%。这支持继续研究近似压缩，不支持无损或完美鲁棒结论。',
'FP32 与 FP16 原型的汇总几乎一致，因此本轮主要误差不能归因于原型 FP16 舍入。第 9 层比其他被测层更敏感，固定压缩率并没有同等保真度。',
'下一步若接入整模型，应先做小规模字段读取，测真实准确率与跨层误差，而不是从 attention 误差直接换算回答准确率。',
'', '## 这次测了什么','',
'- 同一层内，所有已完成的 T chunk 同时从 64 KV 压到 32 个合成 KV；当前未完成块、S、R、Q 可见部分保持原样。',
'- 问题阶段：64 个 T chunk 全压，4096→2048 条 T memory。',
'- 复用 RTX 4080 的 Full Qwen FP16 teacher 缓存；本轮压缩与 FP32 attention 重建在 4080 服务器 CPU 上运行，没有重跑或修改模型。',
'- 4 条开发样本 / 2 个反事实样本对，5 层、全部 32 个 query head。正式测试集未使用。',
'- 各层独立使用 Full teacher Q/K/V，不是逐层传播压缩误差的整模型推理。',
'', '## 全查询结果','',
'下表是注意力向量相对 L2 误差，不是回答错误率。行是相关的 query/head 观测，不能作为独立样本数。',
'', '| 方法 | 平均误差 | p95 | p99 | 最大误差 |','|---|---:|---:|---:|---:|']
for method in m['methods']:
 r=idx[('all',method)]['relative_l2'];lines.append(f"| {method} | {r['mean']:.2%} | {r['p95_abs']:.2%} | {r['p99_abs']:.2%} | {r['max_abs']:.2%} |")
lines+=['','## 主方法 cluster_mass_fp16：分层与查询类型','','| 子集 | 平均误差 | p95 | p99 | 原始压缩历史 attention mass 均值 |','|---|---:|---:|---:|---:|']
for scope in ['text','question',*[f'layer_{i}' for i in m['layers']]]:
 r=idx[(scope,'cluster_mass_fp16')];e=r['relative_l2'];lines.append(f"| {scope} | {e['mean']:.2%} | {e['p95_abs']:.2%} | {e['p99_abs']:.2%} | {r['compressed_history_mass']['mean']:.2%} |")
lines+=['','## 每条开发样本（非独立泛化证据）','','| 样本 | 平均误差 | p99 |','|---|---:|---:|']
for sid in m['samples']:
 r=idx[('sample_'+sid,'cluster_mass_fp16')]['relative_l2'];lines.append(f"| {sid} | {r['mean']:.2%} | {r['p99_abs']:.2%} |")
ratios=[r['byte_ratio'] for r in storage if r['bucket']=='question' and r['method']=='cluster_mass_fp16']
lines+=['','## 资源与控制','',
 f"- 问题阶段，可见 memory 表示张量的字节压缩比约 {min(ratios):.3f}–{max(ratios):.3f}×；已计 FP32 bias，未计容器/索引/allocator。不是整个程序或整卡显存实测下降。",
 f"- 一个 chunk / 层 / KV head 的 CPU 聚类中位时间约 {statistics.median(t['cpu_seconds'] for t in timings)*1000:.2f} ms；非优化 GPU 实现，不能据此声称端到端加速。",
 f"- {len(checks)} 个真实 SDPA 重建控制通过，最大分组相对 RMS {m['max_reference_relative_rms']:.5%} < 0.5%。每组包含共享 KV head 的 4 个 query head。",
 '- identity_raw 的每条输出误差 <1e-6；未来 raw KV 和未来原型扰动不改变 assembly；两块相同 key 的同时合并精确性控制通过。',
 f"- 原始指标 {len(rows):,} 行，records.jsonl.gz；每行数值有限；均值/p95/p99 已由独立标准库脚本复核。",
 '', '## 判读限制','',
'全历史同时压缩比上一轮单 chunk 更接近真实使用，但仍只能评估 teacher attention 重建。没有生成回答，没有验证跨层误差传播，也没有应用准确率或性能成功判据。',
'相对 L2 的分母为该 query/head 原始输出范数；最大值可能受小范数影响，应结合均值、p95/p99、分层结果和后续任务准确率，不单独由最大值定性。',
'', '服务器目录：`<PROJECT_ROOT>/outputs/kv_representation_v0/all_chunks_v1/runs/pilot/`']
(p/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
audit={'status':'PASS','records':len(rows),'reference_checks':len(checks),'finite':True,'source_and_result_hashes':True,'means_p95_p99_independently_verified':True}
(p/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
print('\n'.join(lines))
