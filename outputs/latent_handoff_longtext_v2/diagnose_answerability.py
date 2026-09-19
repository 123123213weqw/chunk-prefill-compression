#!/usr/bin/env python3
"""Separate diagnostics: evidence-only task baseline and long-context operand readout.

These probes never replace or rescore the original compression evaluation.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'chunkpack_validation_v1'))
from run_phase0_chunkpack import build_reference_cache, evaluate_cache, release_cache
from generate_longtext_dataset import segment_metadata, serialize_probe


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--max-long-t', type=int, default=32768)
    args = p.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'answerability.jsonl'
    if path.exists() and not args.resume:
        raise FileExistsError(path)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    enc = lambda text: tok.encode(text, add_special_tokens=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16,
                attn_implementation='sdpa', local_files_only=True, trust_remote_code=False).to('cuda:0')
    model.eval()
    rows = [json.loads(x) for x in Path(args.dataset).read_text().splitlines()]
    records = [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
    done = {(r['sample_id'], r['mode']) for r in records}
    with path.open('a') as f:
        for source in rows:
            for mode in ('evidence_only_original_questions', 'long_full_operand_readout'):
                if mode == 'long_full_operand_readout' and not source['template_type'].startswith('math_'):
                    continue
                if mode == 'long_full_operand_readout' and source['token_targets']['T'] > args.max_long_t:
                    continue
                if (source['sample_id'], mode) in done:
                    continue
                S = enc(source['segments']['S']['text'])
                R = enc(source['segments']['R']['text'])
                if mode == 'evidence_only_original_questions':
                    evidence = [s for probe in source['probes'] if probe['information_class'].startswith('T_')
                                for s in probe['evidence']]
                    T = enc('<|im_start|>tool\n' + '\n'.join(evidence) + '\n<|im_end|>\n')
                    probes = source['probes']
                else:
                    T = enc(source['segments']['T']['text'])
                    fields = [s.split('=', 1)[0] for s in source['probes'][0]['evidence']]
                    gold = {key: int(next(s.split('=', 1)[1] for s in source['probes'][0]['evidence']
                                         if s.startswith(key + '='))) for key in fields}
                    question = '从日志逐项读取以下字段的实际数值，不进行计算：' + ', '.join(fields) + '。只输出 JSON，值使用整数。'
                    probes = [{'probe_id': 'operand-readout', 'information_class': 'T_only_exact',
                               'Q': segment_metadata(enc, serialize_probe(question)),
                               'gold': {'type': 'json_fields', 'answer': gold}}]
                logical = len(S) + len(T) + len(R)
                cache, _ = build_reference_cache(model, S, T, R, logical)
                results, _ = evaluate_cache(model, tok, cache, probes, mode, logical, logical, 96, None)
                record = {'sample_id': source['sample_id'], 'mode': mode, 'T_tokens': len(T), 'probes': results}
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
                f.flush()
                records.append(record)
                release_cache(cache)
                print(source['sample_id'], mode, [(x['probe_id'], x['match']) for x in results], flush=True)
    summary = {}
    for mode in sorted({r['mode'] for r in records}):
        selected = [r for r in records if r['mode'] == mode]
        summary[mode] = {cls: {'correct': sum(p['match'] for r in selected for p in r['probes'] if p['information_class'] == cls),
                              'total': sum(1 for r in selected for p in r['probes'] if p['information_class'] == cls)}
                        for cls in sorted({p['information_class'] for r in selected for p in r['probes']})}
    (out / 'answerability_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
