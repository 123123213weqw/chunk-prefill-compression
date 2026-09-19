#!/usr/bin/env python3
"""Analytical useful-matmul FLOPs only; not measured hardware work or latency."""
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent
# Fixed Qwen3-4B-Instruct-2507 dimensions inspected from server config 2026-09-07.
H=2560;Q=32*128;KV=8*128;F=9728;L=36

def layer(n):
    qkvo=4*n*H*(Q+KV)  # FLOPs: Q/K/V projections plus attention-output projection.
    mlp=6*n*H*F        # SwiGLU gate/up/down linear projections.
    attn=2*Q*n*(n+1)   # QK^T and AV, useful causal pairs (not masked square kernel).
    return {'qkvo':qkvo,'mlp':mlp,'attention':attn,'total':qkvo+mlp+attn}

def estimate(n,full_depth,r):
    assert 0<=full_depth<=L and 0<r<=1
    m=int(n*r);a=layer(n);b=layer(m)
    raw=L*a['total'];new=full_depth*a['total']+(L-full_depth)*b['total']
    return {'n':n,'full_depth':full_depth,'remaining_depth':L-full_depth,'keep_fraction':r,'upper_tokens':m,
            'baseline_useful_flops':raw,'candidate_useful_flops':new,'compute_fraction':new/raw,
            'ideal_flops_ratio':raw/new,'linear_compute_fraction':(full_depth*n+(L-full_depth)*m)/(L*n),
            'attention_fraction_of_baseline':a['attention']/a['total']}

def main():
    assert layer(1)['total']==layer(1)['qkvo']+layer(1)['mlp']+layer(1)['attention']
    assert estimate(4096,12,1)['compute_fraction']==1
    assert estimate(4096,36,.5)['compute_fraction']==1
    assert estimate(4096,4,.5)['compute_fraction']<estimate(4096,12,.5)['compute_fraction']<estimate(4096,24,.5)['compute_fraction']
    data={'scope':'analytic useful matmul FLOPs, not measured speedup','dimensions':{'hidden':H,'query_width':Q,'kv_width':KV,'intermediate':F,'layers':L},
          'assumptions':['all positions reduced; protected S/R/Q positions ignored in this idealization',
                         'two FLOPs per multiply-add; attention counts only valid causal pairs',
                         'no pooling/routing/normalization/RoPE/softmax/LM-head/launch/memory/decoder overhead',
                         'hardware throughput need not remain constant after shrinking matrices',
                         'no semantics or accuracy guarantee'],
          'records':[estimate(n,d,r) for n in (4096,8192,16384) for d in (4,12,24) for r in (.75,.5,.25)]}
    (OUT/'cost_model.json').write_text(json.dumps(data,indent=2)+'\n')
    lines=['# Prefill 计算量账本（非性能结果）','',
           'Qwen3-4B 固定配置：hidden=2560、Q width=4096、KV width=1024、MLP width=9728、36 层。',
           '乘加计 2 FLOPs；只计有效因果 attention 对，不计 masked-square 内核可能执行的额外工作。',
           '', '每层：QKVO = 4 n h (dQ+dKV)，MLP = 6 n h f，attention = 2 dQ n(n+1)。',
           '', '| 长度 | Full 中线性层 FLOPs 占比 | 第 12 层后减半：总 FLOPs 保留 | 理想 FLOPs 比（非实测速比） |',
           '|---:|---:|---:|---:|']
    for n in (4096,8192,16384):
        r=estimate(n,12,.5)
        lines.append(f"| {n} | {1-r['attention_fraction_of_baseline']:.2%} | {r['compute_fraction']:.2%} | {r['ideal_flops_ratio']:.3f}× |")
    lines+=['', '本表理想化地缩短全部位置；实际 S/R/Q 不压缩，还要计入压缩算子、数据搬移、mask 和后端效率，所以不能当作耗时预测或保证的加速上界。',
            '仅对已算好的 KV 做压缩，不会消除本层已经发生的 QKVO/MLP FLOPs；必须在某些运算开始前减少输入行数或稀疏计算连接。']
    (OUT/'COST_MODEL.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
