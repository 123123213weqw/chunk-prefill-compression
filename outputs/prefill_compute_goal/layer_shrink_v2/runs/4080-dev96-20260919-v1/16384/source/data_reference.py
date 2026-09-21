#!/usr/bin/env python3
"""Deterministic paired benchmark; validation reparses text, never trusts state."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

LENGTHS = (4096, 8192, 16384)
LAYOUTS = ('clustered', 'distributed')
FIELDS = {'inventory': ('shard_count', 'items_per_shard', 'rejected_items'),
          'latency': ('base_latency_ms', 'retry_count', 'retry_penalty_ms', 'parallel_discount_ms')}
BASE_SEED = 20260906


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def seed(value):
    return int(digest(f'{BASE_SEED}:{value}')[:16], 16)


def ids_hash(ids):
    return digest(','.join(map(str, ids)))


def read(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def write(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in rows))


def seg(enc, text):
    ids = enc(text)
    return {'text': text, 'token_count': len(ids), 'token_id_sha256': ids_hash(ids)}


def question(enc, text):
    return seg(enc, f'<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n')


def compute(template, f):
    if template == 'inventory':
        return f['shard_count'] * f['items_per_shard'] - f['rejected_items']
    return f['base_latency_ms'] + f['retry_count'] * f['retry_penalty_ms'] - f['parallel_discount_ms']


def fact(job, key, value):
    return f'job={job} field={key} value={value}\n'


def reparse(row):
    parsed = {}
    for job, key, value in re.findall(r'^job=([A-F0-9]+) field=([a-z_]+) value=(\d+)$', row['segments']['T']['text'], re.M):
        if job == row['target_job']:
            if key in parsed:
                raise ValueError('duplicate target field')
            parsed[key] = int(value)
    return parsed


def fit_T(enc, rng, target, lines, layout, fields):
    prefix, suffix = '<|im_start|>tool\n', '<|im_end|>\n'
    fillers = []
    # Several records per non-target job; identical vocabulary and value ranges.
    for i in range(target // 10 + 100):
        job = digest(f'{rng}:{i // len(fields)}')[:12].upper()
        fillers.append(fact(job, fields[i % len(fields)], 10 + seed(f'{rng}:val:{i}') % 80))
    average = len(enc(''.join(fillers[:30]))) / 30
    n = max(1, int((target - len(enc(prefix + ''.join(lines) + suffix))) / average) - 2)

    def build(n, pad=0):
        positions = ([int(n * .48) + i for i in range(len(lines))] if layout == 'clustered'
                     else [round(n * (.12 + i * .24)) for i in range(len(lines))])
        inserts = dict(zip(positions, lines))
        text = prefix + ''.join(inserts.get(i, '') + fillers[i] for i in range(n))
        return text + 'archive_end' + (' .' * pad) + '\n' + suffix

    while len(enc(build(n))) > target:
        n -= 1
    while len(enc(build(n + 1))) <= target:
        n += 1
    for backtrack in range(4):
        base = build(n - backtrack)
        gap = target - len(enc(base))
        for pad in range(max(0, gap - 5), gap + 20):
            text = build(n - backtrack, pad)
            if len(enc(text)) == target:
                return text
    raise ValueError('exact T fit failed')


def evidence_spans(enc, text, evidence):
    ids = enc(text)
    result = []
    for line in evidence:
        assert text.count(line) == 1
        a = len(enc(text[:text.index(line)]))
        needle = enc(line)
        assert ids[a:a + len(needle)] == needle, 'non-compositional evidence boundary'
        b = a + len(needle)
        result.append({'text': line, 'token_span': [a, b], 'chunk_ids': list(range(a // 64, (b-1) // 64+1))})
    return result


def make_pair(enc, template, length, layout, instance):
    group = f'{template}:{length}:{layout}:{instance}'
    rng = random.Random(seed(group))
    pair_id = digest('pair:' + group)[:20]
    job = digest('target:' + group)[:12].upper()
    fields = FIELDS[template]
    values = ([rng.randint(10, 18), rng.randint(10, 29), rng.randint(10, 19)] if template == 'inventory'
              else [rng.randint(30, 58), rng.randint(10, 15), rng.randint(10, 19), rng.randint(10, 19)])
    state = dict(zip(fields, values))
    old_line = fact(job, fields[0], state[fields[0]])
    evidence = [fact(job, key, state[key]) for key in fields]
    base_T = fit_T(enc, group, length, evidence, layout, fields)
    S = seg(enc, '<|im_start|>system\n你是记录核验助手。不同 job 的记录不可混用。根据记录回答后续问题，缺少字段时不要猜测。<|im_end|>\n'
                '<|im_start|>user\n请读取工具返回的作业档案，等待后续核验。<|im_end|>\n')
    R = seg(enc, '<|im_start|>assistant\n档案已读取。下一步动作标识为 verify-checkpoint。<|im_end|>\n')
    formula = ('shard_count × items_per_shard − rejected_items' if template == 'inventory'
               else 'base_latency_ms + retry_count × retry_penalty_ms − parallel_discount_ms')
    result = []
    for member in (0, 1):
        f = dict(state)
        f[fields[0]] += member
        T = base_T.replace(old_line, fact(job, fields[0], f[fields[0]]))
        ev = [fact(job, key, f[key]) for key in fields]
        probes = [
            {'probe_id': 'readout', 'class': 'readout', 'Q': question(enc, f'读取 job={job} 的以下字段：{", ".join(fields)}。只输出 JSON 对象，字段值必须为整数；不要输出示例或占位符。'), 'gold': f},
            {'probe_id': 'calculation', 'class': 'calculation', 'Q': question(enc, f'根据 job={job} 的字段计算 {formula}。遵循先乘后加减。只输出 JSON 对象，唯一字段 result 的值为计算得到的整数。'), 'gold': {'result': compute(template, f)}},
            {'probe_id': 'control', 'class': 'control', 'Q': question(enc, '登记的下一步动作标识是什么？只输出 JSON 对象，唯一字段 action 的值为该标识字符串。'), 'gold': {'action': 'verify-checkpoint'}},
        ]
        result.append({'schema_version': 'chunkpack-validation/v3', 'sample_id': digest(f'sample:{group}:{member}')[:24],
                       'pair_id': pair_id, 'member': member, 'instance': instance,
                       'split': 'development' if instance < 4 else 'test', 'template': template,
                       'length': length, 'evidence_layout': layout, 'target_job': job,
                       'segments': {'S': S, 'T': seg(enc, T), 'R': R}, 'probes': probes,
                       'evidence': ev, 'evidence_spans': evidence_spans(enc, T, ev),
                       'short_T': seg(enc, '<|im_start|>tool\n' + ''.join(ev) + '<|im_end|>\n'),
                       'changed_field': fields[0], 'formula': formula, 'logical_next_position': S['token_count'] + len(enc(T)) + R['token_count']})
    return result


def validate(rows, enc):
    errors, groups = [], {}
    def check(ok, msg):
        if not ok:
            errors.append(msg)
    for r in rows:
        sid = r['sample_id']
        groups.setdefault(r['pair_id'], []).append(r)
        parsed = reparse(r)
        check(set(parsed) == set(FIELDS[r['template']]), sid + ':fields')
        check(parsed == r['probes'][0]['gold'], sid + ':readout gold')
        check({'result': compute(r['template'], parsed)} == r['probes'][1]['gold'], sid + ':math gold')
        for segment in [*r['segments'].values(), r['short_T'], *[p['Q'] for p in r['probes']]]:
            ids = enc(segment['text'])
            check(len(ids) == segment['token_count'] and ids_hash(ids) == segment['token_id_sha256'], sid + ':token metadata')
        check(r['segments']['T']['token_count'] == r['length'], sid + ':budget')
        computed_spans = evidence_spans(enc, r['segments']['T']['text'], r['evidence'])
        check(computed_spans == r['evidence_spans'], sid + ':spans')
        positions = sorted(s['token_span'][0] for s in computed_spans)
        if r['evidence_layout'] == 'distributed':
            check(min(b-a for a, b in zip(positions, positions[1:])) >= .15 * r['length'], sid + ':separation')
        for visible in [*r['segments'].values(), *[p['Q'] for p in r['probes']]]:
            check(sid not in visible['text'] and r['pair_id'] not in visible['text'], sid + ':visible label')
        check(r['target_job'] not in r['segments']['S']['text'] + r['segments']['R']['text'], sid + ':query blind')
    for pid, pair in groups.items():
        check(len(pair) == 2, pid + ':pair missing')
        if len(pair) != 2:
            continue
        a, b = sorted(pair, key=lambda r: r['member'])
        check(a['split'] == b['split'], pid + ':split leak')
        check(a['segments']['S'] == b['segments']['S'] and a['segments']['R'] == b['segments']['R'], pid + ':SR differs')
        check([p['Q'] for p in a['probes']] == [p['Q'] for p in b['probes']], pid + ':Q differs')
        aa, bb = a['segments']['T']['text'].splitlines(), b['segments']['T']['text'].splitlines()
        check(len(aa) == len(bb) and sum(x != y for x, y in zip(aa, bb)) == 1, pid + ':not one-line intervention')
        check(a['probes'][0]['gold'] != b['probes'][0]['gold'] and a['probes'][1]['gold'] != b['probes'][1]['gold'], pid + ':unchanged gold')
        check([x['token_span'] for x in a['evidence_spans']] == [x['token_span'] for x in b['evidence_spans']], pid + ':span shifts')
    grid = Counter((r['template'], r['length'], r['evidence_layout'], r['instance'], r['member']) for r in rows)
    check(len(rows) == len(set(r['sample_id'] for r in rows)), 'duplicate IDs')
    expected = {(t, n, layout, i, m) for t in FIELDS for n in LENGTHS for layout in LAYOUTS for i in range(10) for m in (0, 1)}
    check(set(grid) == expected and all(n == 1 for n in grid.values()), 'incomplete grid')
    return {'status': 'PASS' if not errors else 'FAIL', 'samples': len(rows), 'pairs': len(groups),
            'split_counts': dict(Counter(r['split'] for r in rows)), 'errors': errors,
            'checks': ['gold independently reparsed from T', 'token counts and hashes', 'single-field paired intervention',
                       'identical S/R/Q within pairs', 'no split-crossing pairs', 'no visible sample/pair labels',
                       'query-blind selectors do not see target_job in S', 'distributed evidence spacing >= 15% T', 'exact T budgets']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True, trust_remote_code=False)
    enc = lambda x: tok.encode(x, add_special_tokens=False)
    out = Path(a.output)
    if a.validate_only:
        report = validate(read(out / 'all.jsonl'), enc)
    else:
        if (out / 'manifest.json').exists():
            raise FileExistsError('dataset is immutable; use another output directory')
        out.mkdir(parents=True, exist_ok=True)
        rows = []
        for t in FIELDS:
            for n in LENGTHS:
                for layout in LAYOUTS:
                    for i in range(10):
                        rows.extend(make_pair(enc, t, n, layout, i))
        report = validate(rows, enc)
        if report['status'] != 'PASS':
            raise AssertionError(report)
        write(out / 'all.jsonl', rows)
        for split in ('development', 'test'):
            write(out / f'{split}.jsonl', [r for r in rows if r['split'] == split])
        write(out / 'smoke4.jsonl', [r for r in rows if r['length'] == 4096 and r['evidence_layout'] == 'distributed' and r['instance'] == 0])
        write(out / 'calibration24.jsonl', [r for r in rows if r['instance'] == 0])
        manifest = {'schema': 'v3', 'seed': BASE_SEED, 'files': {f.name: {'bytes': f.stat().st_size,
                    'sha256': hashlib.sha256(f.read_bytes()).hexdigest(), 'samples': len(read(f))} for f in sorted(out.glob('*.jsonl'))},
                    'tokenizer_files': {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(Path(a.tokenizer).glob('*')) if f.is_file()},
                    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (out / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
