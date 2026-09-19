#!/usr/bin/env python3
"""Paired v3 runner. Calibration, uncompressed baseline, offline, online, suite.

Full is uncompressed b1024 prefill, not mislabeled one-shot. Each probe restores
the prefix. Offline conditions project the SAME CPU snapshot of full KV.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'chunkpack_validation_v1'))
import run_phase0_chunkpack as old
from data import read, ids_hash
from methods import strict_score, span_partition, semantic_positions, random_tokens, random_spans, oracle_positions, seed_for


def code_hashes():
    paths = [ROOT / name for name in ('run.py', 'methods.py', 'data.py')]
    paths += [ROOT.parent / 'chunkpack_validation_v1/run_phase0_chunkpack.py', ROOT.parent / 'latent_handoff_prefill/run_prefill_experiment.py']
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def sync():
    torch.cuda.synchronize()


def drop(cache):
    for layer in cache.layers:
        layer.keys = None
        layer.values = None
    gc.collect()
    torch.cuda.empty_cache()


def snapshot(cache):
    sync()
    return [(l.keys.detach().cpu(), l.values.detach().cpu()) for l in cache.layers]


def project(model, saved, ns, nt, nr, kept):
    assert len(set(kept)) == len(kept) and all(0 <= i < nt for i in kept)
    positions = list(range(ns)) + [ns+i for i in sorted(kept)] + list(range(ns+nt, ns+nt+nr))
    idx = torch.tensor(positions, dtype=torch.long)
    cache = DynamicCache(config=model.config)
    for layer, (k, v) in zip(cache.layers, saved):
        keys = k.index_select(-2, idx).to('cuda:0')
        vals = v.index_select(-2, idx).to('cuda:0')
        layer.lazy_initialization(keys, vals)
        layer.keys, layer.values = keys, vals
    old.assert_cache_length(cache, len(positions), 'v3 project')
    return cache


def encode(tok, segment):
    ids = tok.encode(segment['text'], add_special_tokens=False)
    if len(ids) != segment['token_count'] or ids_hash(ids) != segment['token_id_sha256']:
        raise AssertionError('tokenizer does not match dataset')
    return ids


def evaluate(model, tok, cache, sample, logical, max_tokens, reference=None):
    physical = cache.get_seq_length()
    results, refs = [], {}
    for probe in sample['probes']:
        q = encode(tok, probe['Q'])
        sync()
        start = time.perf_counter()
        decoded = old.greedy_answer(model, tok, cache, q, logical, max_tokens, physical)
        sync()
        elapsed = time.perf_counter() - start
        # Generation and teacher-forcing are independent reads of the prefix.
        score, distribution = old.teacher_distribution(model, tok, cache, q, {'answer': probe['gold']}, logical, physical)
        record = dict(strict_score(decoded.text, probe['gold']), probe_id=probe['probe_id'], answer=decoded.text,
                      gold=probe['gold'], generated_tokens=len(decoded.token_ids), hit_token_cap=len(decoded.token_ids) == max_tokens,
                      answer_wall_seconds=elapsed, gold_mean_logprob=score['mean_logprob'])
        if reference:
            ref = reference[probe['probe_id']]
            record['sequence_kl_from_full'] = old.mean_sequence_kl(ref['distribution'], distribution)
            record['literal_equal_full'] = ref['answer'] == decoded.text
        refs[probe['probe_id']] = {'distribution': distribution, 'answer': decoded.text}
        results.append(record)
        old.assert_cache_length(cache, physical, 'probe independence')
    return results, refs


def online_random_span(model, tok, S, T, R, cap, seed, block):
    """Causal whole-line priority reservoir; never resurrects evicted entries."""
    cache = DynamicCache(config=model.config)
    old.prefill_blocks(model, cache, S, 0, block)
    retained, pending = [], []
    peak = len(S)
    for off in range(0, len(T), block):
        prior = sorted([e for s in retained for e in s['entries']] + pending, key=lambda e: e['logical'])
        before = cache.get_seq_length()
        chunk = T[off:off+block]
        old.forward_cache(model, cache, chunk, len(S)+off)
        peak = max(peak, cache.get_seq_length())
        mapping = {e['logical']: len(S)+i for i, e in enumerate(prior)}
        entries = [{'logical': off+i, 'token_id': token} for i, token in enumerate(chunk)]
        mapping.update({off+i: before+i for i in range(len(chunk))})
        complete, pending = old.split_complete_line_spans(tok, pending+entries, off+len(chunk) == len(T))
        for entries in complete:
            priority = seed_for(str(seed), entries[0]['logical'])
            retained.append({'entries': entries, 'priority': priority})
        used, kept = 0, []
        for span in sorted(retained, key=lambda x: -x['priority']):
            n = len(span['entries'])
            if used+n <= max(0, cap-len(pending)):
                kept.append(span)
                used += n
        retained = sorted(kept, key=lambda x: x['entries'][0]['logical'])
        old.compact_all_t_entries(cache, len(S), mapping, [e for s in retained for e in s['entries']] + pending)
    assert not pending
    selected = sorted(e['logical'] for s in retained for e in s['entries'])
    old.prefill_blocks(model, cache, R, len(S)+len(T), block)
    return cache, {'selected_t_positions': selected, 'peak_physical_tokens': peak,
                   'causal': True, 'policy': 'seeded whole-line priority reservoir; budget is cap'}


def online_static(model, S, T, R, kept, block):
    cache = DynamicCache(config=model.config)
    old.prefill_blocks(model, cache, S, 0, block)
    positions = set(kept)
    peak = len(S)
    for off in range(0, len(T), block):
        chunk = T[off:off+block]
        before = cache.get_seq_length()
        old.forward_cache(model, cache, chunk, len(S)+off)
        peak = max(peak, cache.get_seq_length())
        old.compact_appended_tail(cache, before, [i for i in range(len(chunk)) if off+i in positions])
    old.prefill_blocks(model, cache, R, len(S)+len(T), block)
    return cache, {'selected_t_positions': kept, 'peak_physical_tokens': peak, 'causal': True,
                   'policy': 'content-independent positions drawn before reading T'}


def condition_metrics(cache, kept, T, required, elapsed, meta=None):
    positions = set(kept)
    return {'T_kept': len(kept), 'T_raw': len(T), 'T_ratio': len(T)/len(kept) if kept else None,
            'cache_bytes': old.cache_bytes(cache), 'physical_tokens': cache.get_seq_length(),
            'evidence_token_recall': len(positions & required)/len(required) if required else None,
            'complete_evidence_kept': required <= positions, 'build_wall_seconds': elapsed,
            'selected_t_positions': kept, **(meta or {})}


@torch.inference_mode()
def run_sample(model, tok, sample, args, event):
    S, T, R = [encode(tok, sample['segments'][key]) for key in ('S', 'T', 'R')]
    if args.phase == 'calibration':
        T = encode(tok, sample['short_T'])
    logical = len(S)+len(T)+len(R)
    row = {k: sample[k] for k in ('sample_id', 'pair_id', 'member', 'template', 'length', 'evidence_layout', 'split')}
    row.update(phase=args.phase, actual_T=len(T), conditions={})
    required = {i for s in sample['evidence_spans'] for i in range(*s['token_span'])} if args.phase != 'calibration' else set()
    sync()
    start = time.perf_counter()
    full, _ = old.build_reference_cache(model, S, T, R, args.full_block)
    sync()
    build = time.perf_counter()-start
    metrics = condition_metrics(full, list(range(len(T))), T, required, build, {'prefill_block': args.full_block, 'compression': False})
    probes, reference = evaluate(model, tok, full, sample, logical, args.max_new_tokens)
    row['conditions']['full'] = {'probes': probes, 'metrics': metrics}
    event(row['sample_id'], 'full', row['conditions']['full'])
    if args.phase == 'calibration':
        drop(full)
        return row
    saved = snapshot(full)
    drop(full)

    def finish(name, cache, kept, elapsed, meta=None):
        try:
            m = condition_metrics(cache, kept, T, required, elapsed, meta)
            probes, _ = evaluate(model, tok, cache, sample, logical, args.max_new_tokens, reference)
            row['conditions'][name] = {'probes': probes, 'metrics': m}
            event(row['sample_id'], name, row['conditions'][name])
        finally:
            drop(cache)

    def offline(name, kept, select_seconds=0.0, meta=None):
        sync()
        start = time.perf_counter()
        cache = project(model, saved, len(S), len(T), len(R), kept)
        sync()
        finish(name, cache, kept, build+select_seconds+time.perf_counter()-start,
               {'history': 'same_full_snapshot', 'selection_seconds': select_seconds,
                'timing_note': 'includes Full prefill and CPU snapshot restore; not deployable speed benchmark', **(meta or {})})

    sync()
    start = time.perf_counter()
    chunked, _ = old.build_chunked_full(model, S, T, R, args.chunk_size)
    sync()
    finish('chunked_full', chunked, list(range(len(T))), time.perf_counter()-start)
    offline('drop_T_cached_R', [])
    # Distinguishes residual historical influence in R from reconstruction.
    sync()
    start = time.perf_counter()
    replay, _ = old.build_drop_cache(model, S, T, R, args.chunk_size)
    sync()
    finish('drop_T_replay_R', replay, [], time.perf_counter()-start, {'history': 'R recomputed without T, original logical positions'})
    if args.phase == 'baseline':
        return row
    spans = span_partition(tok, T)
    for ratio in args.ratios:
        cap = len(T)//ratio
        if args.phase in ('offline', 'suite'):
            t0 = time.perf_counter()
            semantic = semantic_positions(spans, cap, old.semantic_span_score)
            offline(f'offline_semantic_x{ratio}', semantic, time.perf_counter()-t0, {'budget_cap': cap})
            try:
                t0 = time.perf_counter()
                oracle = oracle_positions(len(T), required, cap, seed_for(sample['pair_id'], 0))
                offline(f'offline_oracle_x{ratio}', oracle, time.perf_counter()-t0, {'uses_gold_evidence': True})
            except ValueError as exc:
                event(sample['sample_id'], f'offline_oracle_x{ratio}', {'status': 'BUDGET_INFEASIBLE', 'error': str(exc)})
            for replication in args.random_seeds:
                seed = seed_for(sample['pair_id'], replication)
                for label, budget in [('cap', cap), ('semantic_matched', len(semantic))]:
                    t0 = time.perf_counter()
                    token_keep = random_tokens(len(T), budget, seed)
                    offline(f'offline_random_token_{label}_x{ratio}_s{replication}', token_keep, time.perf_counter()-t0,
                            {'comparison_K': budget, 'random_replication': replication})
                    t0 = time.perf_counter()
                    span_keep = random_spans(spans, budget, seed)
                    offline(f'offline_random_span_{label}_x{ratio}_s{replication}', span_keep, time.perf_counter()-t0,
                            {'requested_K': budget, 'budget_gap': budget-len(span_keep), 'random_replication': replication})
                    if len(span_keep) != budget:
                        offline(f'offline_random_token_span_matched_{label}_x{ratio}_s{replication}',
                                random_tokens(len(T), len(span_keep), seed), meta={'comparison_K': len(span_keep), 'random_replication': replication})
        if args.phase in ('online', 'suite'):
            sync()
            start = time.perf_counter()
            cache, meta = old.build_semantic_span_cache(model, tok, S, T, R, ratio, args.chunk_size)
            sync()
            selected = meta['selected_t_positions']
            finish(f'online_semantic_x{ratio}', cache, selected, time.perf_counter()-start,
                   {'budget_cap': cap, 'causal': True, 'peak_physical_tokens': meta['peak_physical_tokens']})
            offline(f'full_history_same_positions_semantic_x{ratio}', selected,
                    meta={'diagnostic_only': True, 'purpose': 'same final positions, different KV histories'})
            for replication in args.random_seeds:
                seed = seed_for(sample['pair_id'], replication)
                for kind in ('token', 'span'):
                    sync()
                    start = time.perf_counter()
                    if kind == 'token':
                        cache, meta = online_static(model, S, T, R, random_tokens(len(T), cap, seed), args.chunk_size)
                    else:
                        cache, meta = online_random_span(model, tok, S, T, R, cap, seed, args.chunk_size)
                    sync()
                    finish(f'online_random_{kind}_x{ratio}_s{replication}', cache, meta['selected_t_positions'],
                           time.perf_counter()-start, {'budget_cap': cap, 'causal': True, 'random_replication': replication,
                                                     'peak_physical_tokens': meta['peak_physical_tokens']})
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--phase', choices=['calibration', 'baseline', 'offline', 'online', 'suite'], required=True)
    p.add_argument('--ratios', type=int, nargs='+', default=[2, 4, 8])
    p.add_argument('--random-seeds', type=int, nargs='+', default=list(range(5)))
    p.add_argument('--chunk-size', type=int, default=64)
    p.add_argument('--full-block', type=int, default=1024)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--lock')
    args = p.parse_args()
    if any(r < 1 for r in args.ratios) or len(set(args.random_seeds)) != len(args.random_seeds):
        raise ValueError('invalid ratio/seeds')
    rows = read(args.dataset)
    for pair in {r['pair_id'] for r in rows}:
        if sorted(r['member'] for r in rows if r['pair_id'] == pair) != [0, 1]:
            raise ValueError('input must contain complete pairs')
    hashes = code_hashes()
    dataset_hash = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    if any(r['split'] == 'test' for r in rows):
        if not args.lock:
            raise ValueError('held-out test requires frozen development lock; no bypass')
        lock = json.loads(Path(args.lock).read_text())
        if lock['code_hashes'] != hashes or lock['test_sha256'] != dataset_hash:
            raise ValueError('frozen lock does not match code or test dataset')
        if str(Path(args.model).resolve()) != str(Path(lock['model']).resolve()):
            raise ValueError('model differs from frozen development model')
        if args.phase != 'suite' or args.ratios != lock['ratios'] or args.random_seeds != lock['random_seeds']:
            raise ValueError('test settings differ from frozen protocol')
        if (args.chunk_size, args.full_block, args.max_new_tokens) != (64, 1024, 128):
            raise ValueError('test decoding/prefill differs from frozen protocol')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ('resume', 'out', 'lock')}
    config.update(code_hashes=hashes, dataset_sha256=dataset_hash)
    manifest = out / 'manifest.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != config:
            raise ValueError('existing run differs or --resume omitted; use a fresh output directory')
    manifest.write_text(json.dumps(config, indent=2) + '\n')
    result_path = out / 'results.jsonl'
    done = {r['sample_id'] for r in read(result_path)} if result_path.exists() else set()
    status = {'status': 'RUNNING', 'completed': len(done), 'target': len(rows), 'phase': args.phase}
    def checkpoint():
        (out / 'status.json').write_text(json.dumps(status, indent=2)+'\n')
    checkpoint()
    torch.manual_seed(20260906)
    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16, attn_implementation='sdpa',
                    local_files_only=True, trust_remote_code=False).to('cuda:0')
        model.eval()
        runtime = {'torch': torch.__version__, 'transformers': __import__('transformers').__version__,
                   'gpu': torch.cuda.get_device_name(0), 'allocator': os.environ.get('PYTORCH_ALLOC_CONF'),
                   'gpu_state': subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout,
                   'model_config': model.config.to_dict(),
                   'tokenizer_hashes': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(args.model).glob('*')
                                        if f.name in ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')}}
        (out / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
        with (out / 'events.jsonl').open('a') as events, result_path.open('a') as results:
            def event(sid, condition, result):
                events.write(json.dumps({'sample_id': sid, 'condition': condition, 'result': result}, ensure_ascii=False)+'\n')
                events.flush()
            for sample in rows:
                if sample['sample_id'] in done:
                    continue
                print('START', status['completed']+1, '/', len(rows), sample['sample_id'], sample['length'], flush=True)
                r = run_sample(model, tok, sample, args, event)
                results.write(json.dumps(r, ensure_ascii=False)+'\n')
                results.flush()
                os.fsync(results.fileno())
                status['completed'] += 1
                checkpoint()
                print('DONE', status['completed'], '/', len(rows), flush=True)
        status['status'] = 'COMPLETE'
    except Exception as exc:
        status.update(status='OOM' if isinstance(exc, torch.OutOfMemoryError) else 'FAILED', error=repr(exc))
        raise
    finally:
        checkpoint()


if __name__ == '__main__':
    main()
