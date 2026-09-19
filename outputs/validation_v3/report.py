#!/usr/bin/env python3
"""Pair-clustered results and predeclared eligibility gates; never rescore gold."""
import argparse
import json
import re
import statistics
from pathlib import Path
from methods import paired_summary, compare_seed_family
from data import read


def summarize(out, bootstrap=2000):
    out = Path(out)
    rows = read(out / 'results.jsonl')
    status = json.loads((out / 'status.json').read_text())
    conditions = sorted({c for r in rows for c in r['conditions']})
    results = {}
    for c in conditions:
        results[c] = {p: paired_summary(rows, c, p, bootstrap=bootstrap) for p in ('readout', 'calculation', 'control')}
        metrics = [r['conditions'][c]['metrics'] for r in rows if c in r['conditions']]
        results[c]['resources'] = {k: statistics.mean(m[k] for m in metrics if m[k] is not None) for k in
                                   ('T_kept', 'cache_bytes', 'complete_evidence_kept', 'evidence_token_recall')
                                   if any(m[k] is not None for m in metrics)}
        results[c]['answer_diagnostics'] = {}
        for probe_id in ('readout', 'calculation', 'control'):
            probes = [p for r in rows for p in r['conditions'].get(c, {}).get('probes', []) if p['probe_id'] == probe_id]
            results[c]['answer_diagnostics'][probe_id] = {
                'answers': len(probes), 'format_errors': sum(not p['format_ok'] for p in probes),
                'hit_generation_cap': sum(p.get('hit_token_cap', False) for p in probes),
                'field_correct': sum(p['field_correct'] for p in probes), 'field_total': sum(p['fields'] for p in probes)}
        results[c]['strata'] = {f'{t}/{n}/{layout}': paired_summary([r for r in rows if (r['template'], r['length'], r['evidence_layout']) == (t, n, layout)], c, 'readout', bootstrap=bootstrap)
                                for t, n, layout in sorted({(r['template'], r['length'], r['evidence_layout']) for r in rows})}
    seed_groups = {}
    for c in conditions:
        match = re.match(r'(.+)_s\d+$', c)
        if match:
            seed_groups.setdefault(match[1], []).append(c)
    # Seed results share the same data: report seed means, never inflate N.
    seed_averages = {}
    for name, names in seed_groups.items():
        seed_averages[name] = {p: {'mean_accuracy': statistics.mean(results[c][p]['accuracy'] for c in names if results[c][p]['accuracy'] is not None),
                                 'replications': len(names), 'independent_pairs': results[names[0]][p]['pairs']}
                               for p in ('readout', 'calculation', 'control') if any(results[c][p]['accuracy'] is not None for c in names)}
    full = results.get('full', {})
    phase = status['phase']
    def gate(probe, threshold):
        m = full.get(probe, {})
        if status['status'] != 'COMPLETE' or m.get('n', 0) == 0:
            return 'NOT_EVALUABLE'
        return 'PASS' if m['accuracy'] >= threshold else 'FAIL'
    gates = {'short_readout_ge95': gate('readout', .95) if phase == 'calibration' else 'NOT_TESTED',
             'short_calculation_ge90': gate('calculation', .90) if phase == 'calibration' else 'NOT_TESTED',
             'long_readout_per_length_ge90': 'NOT_TESTED',
             'short_control_ge95': gate('control', .95) if phase == 'calibration' else 'NOT_TESTED'}
    if phase != 'calibration' and rows:
        grouped = [paired_summary([r for r in rows if r['length'] == n], 'full', 'readout', bootstrap=0) for n in (4096, 8192, 16384)]
        gates['long_readout_per_length_ge90'] = ('NOT_EVALUABLE' if any(not g['n'] for g in grouped) else
                                                'PASS' if all(g['accuracy'] >= .90 for g in grouped) else 'FAIL')
    primary_controls = [c for c in conditions if re.fullmatch(r'offline_random_token_semantic_matched_x4_s\d+', c)]
    primary = compare_seed_family(rows, 'offline_semantic_x4', primary_controls, bootstrap=bootstrap)
    summary = {'status': status, 'gates': gates, 'conditions': results, 'random_seed_means': seed_averages,
               'primary_x4_equal_K_semantic_minus_random': primary,
               'complete_pair_count': full.get('readout', {}).get('pairs', 0),
               'inference_unit': 'counterfactual pair; seeds and probes are NOT independent samples',
               'scope': 'synthetic structured records, query-blind compression; no natural-text generalization claim'}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    def cell(m):
        return 'N/A' if not m.get('n') else f"{m['correct']}/{m['n']} ({m['accuracy']:.1%})"
    lines = ['# v3 评测报告', '', f"状态：{status['status']}，完成 {status['completed']}/{status['target']} 条。", '',
             '只有两成员均完成的反事实对进入统计。随机种子不增加独立样本数。', '',
             '| 条件 | 字段读取 | 模型计算 | R 正控制 |', '|---|---:|---:|---:|']
    for c, result in results.items():
        lines.append('| ' + c + ' | ' + ' | '.join(cell(result[p]) for p in ('readout', 'calculation', 'control')) + ' |')
    lines += ['', '## 预注册校准门槛', ''] + [f'- {k}: {v}' for k, v in gates.items()]
    lines += ['', '## 主要方法比较：离线 x4，实际 K 严格相等', '',
              'Semantic 相对五个随机 token 种子的平均读取准确率差（先种子内平均，再按反事实对 bootstrap）：',
              '```json', json.dumps(primary, ensure_ascii=False, indent=2), '```']
    lines += ['', '## 解释约束', '',
              '- Full 是 b1024 的未压缩 prefill；chunked_full 是 b64，均保留全部 KV，不能将 Full 称为一次性 prefill。',
              '- 保持率分母为零时为 N/A；OOM、缺失条件和不完整配对不冒充 0 分。',
              '- 相同预算优先比较 semantic_matched；随机 span 不能精确凑预算时另有 span_matched token 对照。',
              '- 在线方法的预算是上限，比较时须同时看实际 KV 字节数；不能直接把相同 x4 标签当作相同资源。',
              '- full_history_same_positions 为独立机制诊断，不属于可部署在线方法。',
              '- 普通运行的时间含诊断/CPU snapshot 开销，速度结论只使用 performance.py 的专用结果。']
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': status, 'gates': gates, 'full': full}, ensure_ascii=False, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    a = p.parse_args()
    summarize(a.run_dir, a.bootstrap)
