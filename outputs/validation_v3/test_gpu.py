#!/usr/bin/env python3
"""GPU integration invariants; not a model accuracy benchmark."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from data import read
from run import old, snapshot, project, drop, evaluate, encode, online_random_span


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16, attn_implementation='sdpa',
                                                local_files_only=True, trust_remote_code=False).to('cuda:0').eval()
    sample = read(a.dataset)[0]
    S, R = [encode(tok, sample['segments'][k]) for k in ('S', 'R')]
    T = encode(tok, sample['short_T'])
    ns, nt, nr = len(S), len(T), len(R)
    logical = ns+nt+nr
    full, _ = old.build_reference_cache(model, S, T, R, 1024)
    saved = snapshot(full)
    drop(full)
    checks = {}
    for selected in (list(range(nt)), list(range(0, nt, 2)), []):
        cache = project(model, saved, ns, nt, nr, selected)
        idx = torch.tensor(list(range(ns)) + [ns+i for i in selected] + list(range(ns+nt, logical)))
        for layer, (k, v) in zip(cache.layers, saved):
            assert torch.equal(layer.keys.cpu(), k.index_select(-2, idx))
            assert torch.equal(layer.values.cpu(), v.index_select(-2, idx))
        checks[f'project_{len(selected)}_tokens_bitwise_exact'] = True
        if selected == []:
            captured = []
            def hook(module, inputs, kwargs):
                captured.append(int(kwargs['position_ids'][0, 0]))
            handle = model.register_forward_pre_hook(hook, with_kwargs=True)
            q = encode(tok, sample['probes'][0]['Q'])
            old.greedy_answer(model, tok, cache, q, logical, 4, ns+nr)
            handle.remove()
            assert captured[0] == logical
            checks['logical_positions_do_not_follow_physical_length'] = True
        drop(cache)
    cache = project(model, saved, ns, nt, nr, list(range(nt)))
    original, _ = evaluate(model, tok, cache, sample, logical, 128)
    reversed_sample = dict(sample, probes=list(reversed(sample['probes'])))
    reversed_results, _ = evaluate(model, tok, cache, reversed_sample, logical, 128)
    assert {p['probe_id']: p['answer'] for p in original} == {p['probe_id']: p['answer'] for p in reversed_results}
    for layer, (k, v) in zip(cache.layers, saved):
        assert torch.equal(layer.keys.cpu(), k)
        assert torch.equal(layer.values.cpu(), v)
    checks['probe_order_independence_and_no_KV_mutation'] = True
    drop(cache)

    long_T = encode(tok, sample['segments']['T'])[:256]
    changed = list(long_T)
    changed[-2] = tok.encode('z', add_special_tokens=False)[0]
    real_forward = old.forward_cache
    for policy in ('semantic', 'random_span'):
        histories = []
        for source in (long_T, changed):
            observations = []
            def observe(model, cache, ids, logical_start, logits_to_keep=1):
                result = real_forward(model, cache, ids, logical_start, logits_to_keep)
                if logical_start == ns:
                    observations.append(cache.layers[0].keys.detach().cpu().clone())
                return result
            old.forward_cache = observe
            try:
                if policy == 'semantic':
                    cache, _ = old.build_semantic_span_cache(model, tok, S, source, R, 4, 64)
                else:
                    cache, _ = online_random_span(model, tok, S, source, R, 64, 123, 64)
                histories.append(observations)
                drop(cache)
            finally:
                old.forward_cache = real_forward
        assert len(histories[0]) == len(histories[1]) == 1
        assert torch.equal(histories[0][0], histories[1][0])
        checks[f'{policy}_future_perturbation_prefix_invariant'] = True
    report = {'status': 'PASS', 'checks': checks, 'scope': 'cache projection/crop/position invariants and first-prefix perturbation; not exhaustive causal proof'}
    Path(a.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
