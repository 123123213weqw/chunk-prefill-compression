#!/usr/bin/env python3
"""Synthetic single-head KV representation study, not an LLM benchmark.

All methods synthesize 16 KV entries from 64. They are evaluated alongside
uncompressed external KV, so missing attention mass cannot cancel unnoticed.
The fitted prototype is per-cache capacity fitting, NOT a deployable encoder.
"""
import json
import math
from pathlib import Path
import torch

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
OUT = Path(__file__).resolve().parent


def chunk(q, k, v, bias):
    logits = q @ k.T / math.sqrt(k.shape[1]) + bias
    return logits.logsumexp(-1), logits.softmax(-1) @ v


def combined(q, k, v, bias, other_k, other_v):
    kk = torch.cat([k, other_k])
    vv = torch.cat([v, other_v])
    bb = torch.cat([bias, torch.zeros(len(other_k))])
    return (q @ kk.T / math.sqrt(k.shape[1]) + bb).softmax(-1) @ vv


def summarize_groups(k, v, labels, m):
    counts = torch.bincount(labels, minlength=m)
    assert bool((counts > 0).all())
    kk = torch.stack([k[labels == j].mean(0) for j in range(m)])
    vv = torch.stack([v[labels == j].mean(0) for j in range(m)])
    return kk, vv, counts.to(k.dtype).log()


def clusters(k, m):
    # Deterministic farthest-point initialization + Lloyd, using keys only.
    chosen = [0]
    for _ in range(m-1):
        distance = torch.cdist(k, k[chosen]).square().min(-1).values
        distance[chosen] = -1
        chosen.append(int(distance.argmax()))
    centers = k[chosen].clone()
    for _ in range(20):
        distance = torch.cdist(k, centers).square()
        labels = distance.argmin(-1)
        counts = torch.bincount(labels, minlength=m)
        for empty in torch.where(counts == 0)[0].tolist():
            candidates = torch.where(counts[labels] > 1)[0]
            point = candidates[distance[candidates, labels[candidates]].argmax()]
            counts[labels[point]] -= 1
            labels[point] = empty
            counts[empty] += 1
        centers = torch.stack([k[labels == j].mean(0) for j in range(m)])
    return labels


def fit_prototypes(initial, k, v, q_train, q_val, other_k, other_v):
    params = [p.detach().clone().requires_grad_() for p in initial]
    optimizer = torch.optim.Adam(params, lr=.02)
    target = {}
    for name, q in [('train', q_train), ('val', q_val)]:
        z, y = chunk(q, k, v, torch.zeros(len(k)))
        target[name] = (z.detach(), y.detach(), combined(q, k, v, torch.zeros(len(k)), other_k, other_v).detach())

    def loss(q, truth):
        z, y = chunk(q, *params)
        whole = combined(q, *params, other_k, other_v)
        tz, ty, tw = truth
        # log partition + normalized chunk numerator + mixed-context output.
        return ((z-tz).square().mean()
                + (y-ty).square().mean()/ty.square().mean().clamp_min(1e-12)
                + (whole-tw).square().mean()/tw.square().mean().clamp_min(1e-12))

    best = [p.detach().clone() for p in params]
    with torch.no_grad():
        best_val = float(loss(q_val, target['val']))
    best_step = 0
    for step in range(1, 301):
        optimizer.zero_grad()
        value = loss(q_train, target['train'])
        value.backward()
        optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                validation = float(loss(q_val, target['val']))
            if validation < best_val:
                best_val, best_step = validation, step
                best = [p.detach().clone() for p in params]
    return best, {'best_step': best_step, 'validation_loss': best_val,
                  'fit_type': 'per-cache prototype fitting on synthetic training queries; no shared encoder trained'}


def relative_rms(pred, actual):
    return float(((pred-actual).square().sum()/actual.square().sum().clamp_min(1e-20)).sqrt())


def experiment(case, noise, trial):
    torch.manual_seed(20260907+trial)
    n, m, d = 64, 16, 32
    group = torch.arange(n)//4
    base_k, base_v = torch.randn(m, d), torch.randn(m, d)
    k = base_k[group] + noise * torch.randn(n, d)
    v = base_v[group] + .4 * torch.randn(n, d)
    if case == 'independent_keys_values':
        k, v = torch.randn(n, d), torch.randn(n, d)
    other_k, other_v = torch.randn(32, d), torch.randn(32, d)
    train, val = torch.randn(256, d), torch.randn(128, d)
    tests = {'iid': torch.randn(512, d), 'large_norm_shift': 4*torch.randn(512, d)}
    pool = summarize_groups(k, v, group, m)
    cluster_mean = summarize_groups(k, v, clusters(k, m), m)
    fitted, fit_meta = fit_prototypes(cluster_mean, k, v, train, val, other_k, other_v)
    methods = {'mean_pool_no_mass': (pool[0], pool[1], torch.zeros(m)),
               'mean_pool_with_mass': pool,
               'key_cluster_with_mass': cluster_mean,
               'fitted_prototypes_with_mass': fitted}
    rows = []
    for distribution, q in tests.items():
        truth = combined(q, k, v, torch.zeros(n), other_k, other_v)
        truth_logz, _ = chunk(q, k, v, torch.zeros(n))
        for name, params in methods.items():
            result = combined(q, *params, other_k, other_v)
            z, _ = chunk(q, *params)
            error = relative_rms(result, truth)
            if case == 'identical_within_groups' and name in ('mean_pool_with_mass', 'key_cluster_with_mass'):
                assert error < 1e-12
            if case == 'identical_within_groups' and name == 'mean_pool_no_mass':
                assert error > .01
            rows.append({'case': case, 'seed_index': trial, 'queries': distribution, 'method': name,
                         'output_relative_rms': error, 'chunk_logZ_rmse': float((z-truth_logz).square().mean().sqrt())})
    return rows, fit_meta


def main():
    records, fits = [], []
    for case, noise in [('identical_within_groups', 0), ('nearby_keys', .15), ('independent_keys_values', 0)]:
        for trial in range(3):
            rows, fit = experiment(case, noise, trial)
            records.extend(rows)
            fits.append(dict(case=case, seed_index=trial, **fit))
    # A single compressed KV has constant output when it is the only memory.
    # The original two distinct keys/values yield opposite tanh outputs.
    q = torch.tensor([[-2.], [2.]])
    k = torch.tensor([[-1.], [1.]])
    v = torch.tensor([[-1.], [1.]])
    _, y = chunk(q, k, v, torch.zeros(2))
    assert y[0, 0] < -.9 and y[1, 0] > .9
    report = {'status': 'PASS', 'scope': 'synthetic FP64 single-head attention, no Qwen integration or measured speedup',
              'n': 64, 'm': 16, 'head_dim': 32, 'external_raw_KV': 32, 'seeds': 3,
              'queries': {'train': 256, 'validation': 128, 'test_iid': 512, 'test_shift': 512},
              'counterexample_two_KV_to_one': {'original_outputs': y[:, 0].tolist(), 'one_KV_output': 'constant for all q; cannot match both'},
              'records': records, 'fits': fits}
    (OUT/'toy_results.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# KV 表示压缩：数值原型', '', '仅为合成单头 FP64 attention；不是 Qwen 准确率或性能结果。', '',
             '64 个 KV 压成 16 个，旁边保留 32 个未压缩 KV，所有方法都有相同上下文。三组独立随机种子。', '',
             '| 数据 | 查询分布 | 方法 | Attention 输出相对 RMS 误差（三种子均值） |', '|---|---|---|---:|']
    for case in ('identical_within_groups', 'nearby_keys', 'independent_keys_values'):
        for distribution in ('iid', 'large_norm_shift'):
            for method in ('mean_pool_no_mass', 'mean_pool_with_mass', 'key_cluster_with_mass', 'fitted_prototypes_with_mass'):
                selected = [r['output_relative_rms'] for r in records if (r['case'], r['queries'], r['method']) == (case, distribution, method)]
                lines.append(f'| {case} | {distribution} | {method} | {sum(selected)/len(selected):.6f} |')
    lines += ['', '## 检查与边界', '',
              '- 相同 key 组，加 log(group_size) 后在混合上下文中的误差 <1e-12。',
              '- 相同 key 组，不加质量修正仍产生误差，因为压缩组与外部原始 KV 的相对 softmax 权重变化。',
              '- 两个相反 key/value 的反例：一个固定 KV 的输出不随 q 变化，不能同时匹配两种查询。',
              '- fitted 是每个缓存单独用训练 query 优化 K/V/b 的容量试验；测试 query 未参与拟合，但这不是一次前向可部署的共享编码器。',
              '- 大范数 query 是分布外压力测试，验证平均误差下降是否具有稳定性；没有全 query 无损保证。',
              '- 未模拟多层误差累积、GQA、RoPE、实际自然文本 query 或自定义 attention 内核的性能。']
    (OUT/'TOY_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
