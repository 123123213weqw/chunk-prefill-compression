#!/usr/bin/env python3
"""Dedicated end-to-end GPU benchmark; fixed decode work, warmups, randomized order.

No CPU full-cache snapshot/restore is included. Offline cache deletion is not
advertised as prefill acceleration. Correctness comes from run.py, not this test.
"""
import argparse
import json
import random
import statistics
import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import read
from methods import random_tokens, seed_for
from run import old, encode, sync, drop, online_static, code_hashes


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--length', type=int, default=16384)
    p.add_argument('--max-samples', type=int, default=4)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--decode-tokens', type=int, default=64)
    p.add_argument('--ratio', type=int, default=4)
    a = p.parse_args()
    if a.warmups < 1 or a.repeats < 3 or a.decode_tokens < 2:
        raise ValueError('need >=1 warmup, >=3 repeats and >=2 decode tokens')
    rows = [r for r in read(a.dataset) if r['length'] == a.length and r['member'] == 0]
    if any(r['split'] != 'development' for r in rows):
        raise ValueError('performance tuning only uses development data')
    rows = rows[:a.max_samples]
    if not rows:
        raise ValueError('no benchmark samples')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'trials.jsonl').exists():
        raise FileExistsError('benchmark is immutable; use new output directory')
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation='sdpa',
                                                local_files_only=True, trust_remote_code=False).to('cuda:0')
    model.eval()
    manifest = dict(vars(a), code_hashes=code_hashes(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                    task='fixed-token throughput, ignoring EOS; not answer correctness')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    trials = []
    with (out / 'trials.jsonl').open('w') as f:
        for sample in rows:
            S, T, R = [encode(tok, sample['segments'][k]) for k in ('S', 'T', 'R')]
            Q = encode(tok, sample['probes'][0]['Q'])
            logical = len(S)+len(T)+len(R)
            for repeat in range(-a.warmups, a.repeats):
                methods = ['full_b1024', 'full_b64', 'online_semantic', 'online_random_token']
                random.Random(seed_for(sample['pair_id'], repeat)).shuffle(methods)
                for method in methods:
                    cache = None
                    sync()
                    torch.cuda.reset_peak_memory_stats()
                    start = time.perf_counter()
                    try:
                        if method.startswith('full_'):
                            block = int(method.split('b')[-1])
                            cache, _ = old.build_reference_cache(model, S, T, R, block)
                        elif method == 'online_semantic':
                            cache, _ = old.build_semantic_span_cache(model, tok, S, T, R, a.ratio, 64)
                        else:
                            kept = random_tokens(len(T), len(T)//a.ratio, seed_for(sample['pair_id'], 0))
                            cache, _ = online_static(model, S, T, R, kept, 64)
                        sync()
                        build_seconds = time.perf_counter()-start
                        kv_bytes = old.cache_bytes(cache)
                        physical = cache.get_seq_length()
                        qstart = time.perf_counter()
                        result = old.forward_cache(model, cache, Q, logical)
                        token = int(result.logits[0, -1].argmax().item())
                        del result
                        sync()
                        ttft = time.perf_counter()-qstart
                        dstart = time.perf_counter()
                        for step in range(a.decode_tokens-1):
                            result = old.forward_cache(model, cache, [token], logical+len(Q)+step)
                            token = int(result.logits[0, -1].argmax().item())
                            del result
                        sync()
                        decode = time.perf_counter()-dstart
                        row = {'sample_id': sample['sample_id'], 'repeat': repeat, 'warmup': repeat < 0,
                               'method': method, 'status': 'OK', 'prefix_seconds': build_seconds,
                               'query_ttft_seconds': ttft, 'decode_seconds': decode,
                               'decode_tokens_after_first': a.decode_tokens-1,
                               'decode_tokens_per_second': (a.decode_tokens-1)/decode,
                               'end_to_end_seconds': build_seconds+ttft+decode,
                               'cache_bytes_before_Q': kv_bytes, 'physical_prefix_tokens': physical,
                               'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                               'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
                    except torch.OutOfMemoryError as exc:
                        row = {'sample_id': sample['sample_id'], 'repeat': repeat, 'warmup': repeat < 0,
                               'method': method, 'status': 'OOM', 'error': str(exc)}
                    finally:
                        if cache is not None:
                            drop(cache)
                    f.write(json.dumps(row)+'\n')
                    f.flush()
                    trials.append(row)
                    print(sample['sample_id'], repeat, method, row['status'], flush=True)
    stats = {}
    for method in sorted({r['method'] for r in trials}):
        valid = [r for r in trials if r['method'] == method and not r['warmup'] and r['status'] == 'OK']
        stats[method] = {'valid_trials': len(valid), 'oom_trials': sum(r['method'] == method and r['status'] == 'OOM' for r in trials)}
        for key in ('prefix_seconds', 'query_ttft_seconds', 'decode_tokens_per_second', 'end_to_end_seconds', 'cache_bytes_before_Q', 'peak_allocated_bytes'):
            values = [r[key] for r in valid]
            if values:
                stats[method][key] = {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
    (out / 'summary.json').write_text(json.dumps(stats, indent=2)+'\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
