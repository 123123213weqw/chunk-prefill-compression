"""Pure scoring, span selection, paired statistics. No model or gold in selectors."""
import hashlib
import json
import math
import random
import re
import statistics


def strict_score(answer, gold):
    text = answer.strip()
    if text.startswith('```') and text.endswith('```'):
        text = re.sub(r'^```(?:json)?\s*', '', text)[:-3].strip()
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError('duplicate JSON key')
            obj[k] = v
        return obj
    try:
        obj = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError):
        return {'correct': False, 'format_ok': False, 'parsed': None, 'field_correct': 0, 'fields': len(gold)}
    valid = isinstance(obj, dict) and set(obj) == set(gold)
    hits = sum(k in obj and type(obj[k]) is type(v) and obj[k] == v for k, v in gold.items()) if isinstance(obj, dict) else 0
    types_ok = valid and all(type(obj[k]) is type(v) for k, v in gold.items())
    return {'correct': types_ok and hits == len(gold), 'format_ok': bool(types_ok), 'parsed': obj,
            'field_correct': hits, 'fields': len(gold)}


def span_partition(tokenizer, ids):
    spans, start = [], 0
    for i, token in enumerate(ids):
        if '\n' in tokenizer.decode([token], clean_up_tokenization_spaces=False):
            spans.append({'a': start, 'b': i+1, 'text': tokenizer.decode(ids[start:i+1], clean_up_tokenization_spaces=False)})
            start = i+1
    if start < len(ids):
        spans.append({'a': start, 'b': len(ids), 'text': tokenizer.decode(ids[start:], clean_up_tokenization_spaces=False)})
    return spans


def semantic_positions(spans, budget, score_fn):
    # Freeze and reuse v2 scoring/fingerprint policy; this is a baseline, not a
    # claim that the heuristic is a learned semantic compressor.
    from collections import Counter
    seen, best = Counter(), {}
    for span in spans:
        score, fingerprint = score_fn(span['text'], seen)
        seen[fingerprint] += 1
        candidate = dict(span, score=score)
        if fingerprint not in best or score > best[fingerprint]['score']:
            best[fingerprint] = candidate
    chosen = []
    for span in sorted(best.values(), key=lambda x: (-x['score'], x['a'])):
        if span['score'] > 0 and len(chosen) + span['b'] - span['a'] <= budget:
            chosen.extend(range(span['a'], span['b']))
    return sorted(chosen)


def random_tokens(n, k, seed):
    if not 0 <= k <= n:
        raise ValueError('invalid token budget')
    return sorted(random.Random(seed).sample(range(n), k))


def random_spans(spans, cap, seed):
    """Randomized subset-sum: whole spans, maximum reachable budget <= cap.

    Not a uniform sampler over all subsets. Order is seeded; exact reachable
    budget is preferred. Caller reports actual K and uses matched-token control.
    """
    order = list(range(len(spans)))
    random.Random(seed).shuffle(order)
    reachable, parent = 1, {}
    mask = (1 << (cap+1)) - 1
    for index in order:
        s = spans[index]
        length = s['b'] - s['a']
        novel = ((reachable << length) & mask) & ~reachable
        new = novel
        while new:
            low = new & -new
            target = low.bit_length() - 1
            parent[target] = (target-length, index)
            new -= low
        reachable |= novel
        if reachable & (1 << cap):
            break
    target = reachable.bit_length() - 1
    chosen = []
    while target:
        target, index = parent[target]
        chosen.extend(range(spans[index]['a'], spans[index]['b']))
    return sorted(chosen)


def oracle_positions(n, required, budget, seed):
    required = set(required)
    if len(required) > budget:
        raise ValueError('oracle budget insufficient for full required evidence')
    extras = [i for i in range(n) if i not in required]
    return sorted(required | set(random.Random(seed).sample(extras, budget-len(required))))


def seed_for(pair_id, replication):
    return int(hashlib.sha256(f'v3:{pair_id}:{replication}'.encode()).hexdigest()[:16], 16)


def paired_summary(records, condition, probe_id, reference='full', bootstrap=2000):
    pairs = {}
    for r in records:
        if condition not in r['conditions'] or reference not in r['conditions']:
            continue
        p = next((p for p in r['conditions'][condition]['probes'] if p['probe_id'] == probe_id), None)
        q = next((p for p in r['conditions'][reference]['probes'] if p['probe_id'] == probe_id), None)
        if p is not None and q is not None:
            pairs.setdefault(r['pair_id'], []).append((r['member'], bool(p['correct']), bool(q['correct'])))
    complete = [p for p in pairs.values() if len(p) == 2 and {x[0] for x in p} == {0, 1}]
    trials = [x for p in complete for x in p]
    n = len(trials)
    if not n:
        return {'n': 0, 'pairs': 0, 'accuracy': None, 'paired_delta': None, 'retention': None}
    num = sum(a for _, a, b in trials)
    full_n = sum(b for _, a, b in trials)
    delta = sum(a-b for _, a, b in trials)/n
    rng = random.Random(20260906)
    draws = sorted(statistics.mean(sum(a-b for _, a, b in rng.choice(complete))/2 for _ in complete) for _ in range(bootstrap))
    ci = [draws[int(.025*(len(draws)-1))], draws[int(.975*(len(draws)-1))]] if draws else None
    return {'n': n, 'pairs': len(complete), 'correct': num, 'accuracy': num/n, 'paired_delta': delta,
            'delta_pair_bootstrap_95ci': ci, 'both_members_correct': sum(all(a for _, a, b in p) for p in complete)/len(complete),
            'full_correct_denominator': full_n, 'retention': sum(a and b for _, a, b in trials)/full_n if full_n else None}


def compare_seed_family(records, automatic, controls, probe_id='readout', bootstrap=2000):
    """Average random seeds within each member BEFORE pair-cluster resampling."""
    groups = {}
    for row in records:
        names = [automatic, *controls]
        if not controls or any(name not in row['conditions'] for name in names):
            continue
        if any(row['conditions'][c]['metrics']['T_kept'] != row['conditions'][automatic]['metrics']['T_kept'] for c in controls):
            raise ValueError('primary comparison requires identical actual K')
        scores = {c: next(p['correct'] for p in row['conditions'][c]['probes'] if p['probe_id'] == probe_id) for c in names}
        delta = float(scores[automatic]) - statistics.mean(float(scores[c]) for c in controls)
        groups.setdefault(row['pair_id'], []).append((row['member'], delta))
    pairs = [statistics.mean(x[1] for x in values) for values in groups.values() if len(values) == 2 and {x[0] for x in values} == {0, 1}]
    if not pairs:
        return {'pairs': 0, 'mean_delta': None, '95ci': None}
    rng = random.Random(20260906)
    draws = sorted(statistics.mean(rng.choice(pairs) for _ in pairs) for _ in range(bootstrap))
    return {'pairs': len(pairs), 'members': len(pairs)*2, 'random_seeds': len(controls),
            'mean_delta': statistics.mean(pairs),
            '95ci': [draws[int(.025*(len(draws)-1))], draws[int(.975*(len(draws)-1))]] if draws else None}
