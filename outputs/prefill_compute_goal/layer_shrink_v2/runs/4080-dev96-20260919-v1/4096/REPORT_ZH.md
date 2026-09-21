# 深度 / 压缩率：4K 完整开发阶段

固定 32 条 / 16 对，覆盖两模板、两布局、每格四对。已有开发数据，不是独立盲测。

## 正确性（OOM 包含在总分母，另外报告）

| 方法 | readout | control | 字段命中 | 两成员均对 | 格式错 / 封顶 / OOM |
|---|---:|---:|---:|---:|---:|
| native_full | 32/32 | 32/32 | 112/112 | 16/16 | 0 / 0 / 0 |
| identity_d12 | 32/32 | 32/32 | 112/112 | 16/16 | 0 / 0 / 0 |
| identity_d24 | 32/32 | 32/32 | 112/112 | 16/16 | 0 / 0 / 0 |
| mean_d12_k32 | 10/32 | 32/32 | 78/112 | 4/16 | 0 / 0 / 0 |
| mean_d24_k32 | 32/32 | 32/32 | 112/112 | 16/16 | 0 / 0 / 0 |
| mean_d12_k48 | 20/32 | 32/32 | 99/112 | 9/16 | 0 / 0 / 0 |

## 相对 Full 的损伤

| 方法 | Full 对→方法错 | Full 错→方法对 | 准确率差 (百分点) | 95% 配对 CI (百分点) |
|---|---:|---:|---:|---|
| mean_d12_k32 | 22 | 0 | -68.75 | [-84.38, -56.25] |
| mean_d24_k32 | 0 | 0 | 0.00 | [0.00, 0.00] |
| mean_d12_k48 | 12 | 0 | -37.50 | [-59.38, -15.62] |
2000 次分层 pair-cluster bootstrap，反事实成员不拆开；开发集描述性 CI，不是正式多重检验或无损保证。
若存在 OOM，它计入未成功，同时单列；不得由只保留成功运行的分母夸大准确率。

## 读取分层

| 方法 | 模板 | 布局 | 正确 | OOM |
|---|---|---|---:|---:|
| identity_d12 | inventory | clustered | 8/8 | 0 |
| identity_d12 | inventory | distributed | 8/8 | 0 |
| identity_d12 | latency | clustered | 8/8 | 0 |
| identity_d12 | latency | distributed | 8/8 | 0 |
| identity_d24 | inventory | clustered | 8/8 | 0 |
| identity_d24 | inventory | distributed | 8/8 | 0 |
| identity_d24 | latency | clustered | 8/8 | 0 |
| identity_d24 | latency | distributed | 8/8 | 0 |
| mean_d12_k32 | inventory | clustered | 0/8 | 0 |
| mean_d12_k32 | inventory | distributed | 5/8 | 0 |
| mean_d12_k32 | latency | clustered | 0/8 | 0 |
| mean_d12_k32 | latency | distributed | 5/8 | 0 |
| mean_d12_k48 | inventory | clustered | 3/8 | 0 |
| mean_d12_k48 | inventory | distributed | 5/8 | 0 |
| mean_d12_k48 | latency | clustered | 6/8 | 0 |
| mean_d12_k48 | latency | distributed | 6/8 | 0 |
| mean_d24_k32 | inventory | clustered | 8/8 | 0 |
| mean_d24_k32 | inventory | distributed | 8/8 | 0 |
| mean_d24_k32 | latency | clustered | 8/8 | 0 |
| mean_d24_k32 | latency | distributed | 8/8 | 0 |
| native_full | inventory | clustered | 8/8 | 0 |
| native_full | inventory | distributed | 8/8 | 0 |
| native_full | latency | clustered | 8/8 | 0 |
| native_full | latency | distributed | 8/8 | 0 |

## 实际计算量

128 项 identity 检查；192 组逐层算子形状记录。
- mean_d12_k32：成功形状记录 32/32；线性 FLOPs / Full 67.2110%–67.2110%。
- mean_d24_k32：成功形状记录 32/32；线性 FLOPs / Full 83.6055%–83.6055%。
- mean_d12_k48：成功形状记录 32/32；线性 FLOPs / Full 83.6055%–83.6055%。
只统计七种线性算子，不等同于完整 attention FLOPs 或时延。

## 独立计时

所有 trial 边界未观察到其他 compute PID（不排除采样间干扰）。
4 个预先固定代表文档（每模板/布局的 instance 0、member 0），1 warmup + 5 repeats；未按正确性筛选。
每列分别取中位数；总 TTFT 中位数不一定等于两个分段中位数相加。

| 样本 | 方法 | Prefill ms | Q 首 token ms | 总 TTFT ms | Q 后 KV MiB |
|---|---|---:|---:|---:|---:|
| 408aec24 | native_b1024 | 592.18 | 34.42 | 626.83 | 593.58 |
| 408aec24 | native_b4096 | 678.04 | 34.41 | 712.48 | 593.58 |
| 408aec24 | identity_d12 | 592.88 | 34.36 | 627.26 | 593.58 |
| 408aec24 | identity_d24 | 592.28 | 34.33 | 626.59 | 593.58 |
| 408aec24 | mean_d12_k32 | 391.43 | 29.84 | 421.27 | 401.58 |
| 408aec24 | mean_d24_k32 | 491.88 | 32.08 | 523.95 | 497.58 |
| 408aec24 | mean_d12_k48 | 490.89 | 32.04 | 522.92 | 497.58 |
| 547c82f7 | native_b1024 | 592.85 | 34.86 | 627.71 | 594.56 |
| 547c82f7 | native_b4096 | 677.75 | 34.78 | 712.51 | 594.56 |
| 547c82f7 | identity_d12 | 592.57 | 34.68 | 627.24 | 594.56 |
| 547c82f7 | identity_d24 | 592.04 | 34.73 | 626.77 | 594.56 |
| 547c82f7 | mean_d12_k32 | 391.45 | 30.15 | 421.60 | 402.56 |
| 547c82f7 | mean_d24_k32 | 491.94 | 32.49 | 524.43 | 498.56 |
| 547c82f7 | mean_d12_k48 | 491.29 | 32.47 | 523.76 | 498.56 |
| 78cc3045 | native_b1024 | 592.35 | 34.74 | 627.11 | 594.14 |
| 78cc3045 | native_b4096 | 677.84 | 34.73 | 712.52 | 594.14 |
| 78cc3045 | identity_d12 | 592.14 | 34.58 | 626.72 | 594.14 |
| 78cc3045 | identity_d24 | 592.59 | 34.68 | 627.31 | 594.14 |
| 78cc3045 | mean_d12_k32 | 391.53 | 30.11 | 421.60 | 402.14 |
| 78cc3045 | mean_d24_k32 | 491.86 | 32.37 | 524.28 | 498.14 |
| 78cc3045 | mean_d12_k48 | 491.14 | 32.42 | 523.56 | 498.14 |
| 909880cf | native_b1024 | 592.25 | 34.78 | 627.00 | 594.56 |
| 909880cf | native_b4096 | 677.84 | 34.78 | 712.62 | 594.56 |
| 909880cf | identity_d12 | 592.12 | 34.67 | 626.76 | 594.56 |
| 909880cf | identity_d24 | 592.21 | 34.71 | 626.91 | 594.56 |
| 909880cf | mean_d12_k32 | 391.47 | 30.20 | 421.64 | 402.56 |
| 909880cf | mean_d24_k32 | 492.00 | 32.42 | 524.41 | 498.56 |
| 909880cf | mean_d12_k48 | 491.02 | 32.43 | 523.43 | 498.56 |

### 与同轮较快原生基线比较

| 样本 | 方法 | 原生基线 | Prefill 时间减少 | 速度比 |
|---|---|---|---:|---:|
| 408aec24 | mean_d12_k32 | native_b1024 | 33.90% | 1.513× |
| 408aec24 | mean_d24_k32 | native_b1024 | 16.94% | 1.204× |
| 408aec24 | mean_d12_k48 | native_b1024 | 17.10% | 1.206× |
| 547c82f7 | mean_d12_k32 | native_b1024 | 33.97% | 1.515× |
| 547c82f7 | mean_d24_k32 | native_b1024 | 17.02% | 1.205× |
| 547c82f7 | mean_d12_k48 | native_b1024 | 17.13% | 1.207× |
| 78cc3045 | mean_d12_k32 | native_b1024 | 33.90% | 1.513× |
| 78cc3045 | mean_d24_k32 | native_b1024 | 16.96% | 1.204× |
| 78cc3045 | mean_d12_k48 | native_b1024 | 17.09% | 1.206× |
| 909880cf | mean_d12_k32 | native_b1024 | 33.90% | 1.513× |
| 909880cf | mean_d24_k32 | native_b1024 | 16.93% | 1.204× |
| 909880cf | mean_d12_k48 | native_b1024 | 17.09% | 1.206× |

独立审计 PASS 只表示记录一致。每格仅四对，未达到普遍无损/鲁棒的证据要求；失败样本和 Full 原生能力限制必须保留。
