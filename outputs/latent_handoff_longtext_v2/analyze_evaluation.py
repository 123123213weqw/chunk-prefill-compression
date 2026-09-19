#!/usr/bin/env python3
"""Read-only analysis of longtext results; no rescoring or selector tuning."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def fraction(a, b):
    return {'correct': a, 'total': b, 'rate': a / b if b else None}


def accuracy(rows, condition, primary_only=False):
    correct = total = retained = denominator = 0
    for row in rows:
        full = {p['probe_id']: p for p in row['conditions']['one_shot_full']}
        for p in row['conditions'].get(condition, []):
            if primary_only and p['probe_id'] != 'primary':
                continue
            total += 1
            correct += bool(p['match'])
            if full[p['probe_id']]['match']:
                denominator += 1
                retained += bool(p['match'])
    return {'raw': fraction(correct, total), 'full_correct_retention': fraction(retained, denominator)}


def mean(values):
    return statistics.mean(values) if values else None


def display(metric):
    if not metric['total']:
        return 'N/A'
    return f"{metric['correct']}/{metric['total']} ({metric['rate']:.1%})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    out = Path(args.run_dir)
    data = {r['sample_id']: r for r in read(args.dataset)}
    rows = read(out / 'results.jsonl')
    conditions = sorted({c for r in rows for c in r['conditions']})
    stats, failures = {}, []
    for c in conditions:
        metrics = [r['condition_metrics'][c] for r in rows if c in r['condition_metrics']]
        coverage = []
        for r in rows:
            m = r['condition_metrics'].get(c, {})
            if 'selected_t_positions' in m:
                kept = set(m['selected_t_positions'])
                source = data[r['sample_id']]
                facts = set(source['probes'][0]['evidence'])
                spans = [e['token_span'] for e in source['evidence_spans'] if e['text'] in facts]
                needed = {i for a, b in spans for i in range(a, b)}
                coverage.append({'sample_id': r['sample_id'], 'token_recall': len(kept & needed) / len(needed),
                                 'all_primary_evidence_kept': needed <= kept})
            for p in r['conditions'].get(c, []):
                if not p['match']:
                    failures.append({'sample_id': r['sample_id'], 'condition': c, 'probe_id': p['probe_id'],
                                     'class': p['information_class'], 'answer': p['answer'], 'gold': p['gold']})
        stats[c] = {
            'all_probes': accuracy(rows, c), 'primary': accuracy(rows, c, True),
            'primary_by_length': {str(t): accuracy([r for r in rows if r['token_lengths']['T'] == t], c, True)
                                  for t in sorted({r['token_lengths']['T'] for r in rows})},
            'mean_build_seconds': mean([m['build_seconds'] for m in metrics]),
            'mean_cache_bytes': mean([m['cache_bytes'] for m in metrics]),
            'mean_T_kept_tokens': mean([m['T_kept_tokens'] for m in metrics if 'T_kept_tokens' in m]),
            'actual_T_compression_ratios': [m['T_compression_ratio'] for m in metrics if 'T_compression_ratio' in m],
            'evidence_coverage': coverage,
        }
    log_text = (out / 'run.log').read_text() if (out / 'run.log').exists() else ''
    status = 'COMPLETE' if len(rows) == len(data) else ('PARTIAL_OOM' if 'OutOfMemoryError' in log_text else 'PARTIAL')
    report = {'status': status, 'completed_samples': len(rows), 'dataset_samples': len(data),
              'dataset_sha256': hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
              'conditions': stats, 'failures': failures,
              'limitations': [
                  'Smoke16 仅含 middle-position 的 development 样本，不是留出的 test split。',
                  '事实使用 key=value 格式，且带 evidence_bundle 标记；不能外推到自然长文档。',
                  '数学操作数跨越 64-token chunk，但集中在较短证据区域内，并非分散在全文的多跳任务。',
                  'candidate_status 虽然正负平衡，但由模板/长度/位置/seed 的奇偶决定，相关标签也出现在可见 stream ID 中；不应称为无泄漏配对反事实。',
                  '原 runner 的 Gate B 将未跑的 x8 记为 false；本次 x8 应解释为未测试。Full-correct 分母为 0 的数学保持率应为 N/A，而非 0%。',
                  'Semantic x4 只约束最大 KV 预算，并不保证精确保留 1/4，须看实际 token 数。',
                  'build_seconds 只是单次诊断计时，不是严格速度基准，也没有测量 decode 吞吐。',
              ]}
    diagnostic_path = out / 'diagnostics' / 'answerability_summary.json'
    if diagnostic_path.exists():
        report['answerability_diagnostics'] = json.loads(diagnostic_path.read_text())
    (out / 'EVALUATION.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# Longtext v2 smoke 评估', '', f'状态：{status}。完成 {len(rows)}/{len(data)} 条。未调整选择器或 matcher。', '',
             '## 实际答题准确率', '', '| 条件 | 全部 probes | 主问题（原始准确率） | 主问题 Full-correct 保持率 |',
             '|---|---:|---:|---:|']
    for c, s in stats.items():
        lines.append(f"| {c} | {display(s['all_probes']['raw'])} | {display(s['primary']['raw'])} | {display(s['primary']['full_correct_retention'])} |")
    lines += ['', '## 主问题按 T 长度分层（原始准确率）', '', '| 条件 | 4K | 8K | 16K | 32K |', '|---|---:|---:|---:|---:|']
    for c, s in stats.items():
        cells = [display(s['primary_by_length'].get(str(t), {'raw': fraction(0, 0)})['raw']) for t in (4096, 8192, 16384, 32768)]
        lines.append('| ' + c + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## KV 与完整主证据保留', '', '| 条件 | 平均保留 T tokens | 实际 T 压缩比范围 | 完整主证据保留样本 |', '|---|---:|---:|---:|']
    for c, s in stats.items():
        ratios, coverage = s['actual_T_compression_ratios'], s['evidence_coverage']
        if not ratios:
            continue
        lines.append(f"| {c} | {s['mean_T_kept_tokens']:.1f} | {min(ratios):.2f}–{max(ratios):.2f}× | {sum(x['all_primary_evidence_kept'] for x in coverage)}/{len(coverage)} |")
    lines += ['', '## 解释边界', ''] + ['- ' + x for x in report['limitations']]
    if status == 'PARTIAL_OOM':
        lines += ['', '32K 的一次性 Full prefill 在 4080 上 OOM；启用 expandable_segments 重试仍失败。没有停止其他用户进程，也未替换 Full 基线。32K 四条未得到完整对照结果，不计入分母。']
    if 'answerability_diagnostics' in report:
        lines += ['', '## 独立可回答性诊断（不替换原始评分）', '',
                  '使用原始问题，但将 T 换成必要证据短文本；另外给未压缩长文本新增操作数 JSON 提取问题。诊断生成上限为 96 tokens，主实验为 48 tokens。', '']
        for mode, classes in report['answerability_diagnostics'].items():
            for cls, m in classes.items():
                lines.append(f"- {mode} / {cls}: {m['correct']}/{m['total']}")
        lines += ['', '诊断解释：短证据数学题仍全部答错，而长文本操作数提取全部答对；当前模型与原提问方式的计算表现未通过基础校准，不能将数学失败全部归咎于压缩。',
                  '短证据 IP 题仍输出 IP:实际端口，提示 IP 占位符的提问方式需要单独改版验证，不应放宽 matcher 把它判对。',
                  '自动选择器在已完成的 12 条中仅有 9 条完整保留主证据；4K 下可见 deployment_region、items_per_shard、retry_count 等整行被删，存在独立的选择器问题。']
    (out / 'EVALUATION.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
