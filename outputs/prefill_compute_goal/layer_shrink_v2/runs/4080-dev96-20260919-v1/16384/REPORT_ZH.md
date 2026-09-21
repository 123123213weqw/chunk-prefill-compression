# 深度 / 压缩率：16K 完整开发阶段

固定 32 条 / 16 对，覆盖两模板、两布局、每格四对。已有开发数据，不是独立盲测。

## 正确性（OOM 包含在总分母，另外报告）

| 方法 | readout | control | 字段命中 | 两成员均对 | 格式错 / 封顶 / OOM |
|---|---:|---:|---:|---:|---:|
| native_full | 30/32 | 32/32 | 110/112 | 15/16 | 0 / 0 / 0 |
| identity_d12 | 30/32 | 32/32 | 110/112 | 15/16 | 0 / 0 / 0 |
| identity_d24 | 30/32 | 32/32 | 110/112 | 15/16 | 0 / 0 / 0 |
| mean_d12_k32 | 10/32 | 32/32 | 70/112 | 5/16 | 0 / 0 / 0 |
| mean_d24_k32 | 26/32 | 32/32 | 106/112 | 13/16 | 0 / 0 / 0 |
| mean_d12_k48 | 16/32 | 32/32 | 93/112 | 7/16 | 0 / 0 / 0 |

## 相对 Full 的损伤

| 方法 | Full 对→方法错 | Full 错→方法对 | 准确率差 (百分点) | 95% 配对 CI (百分点) |
|---|---:|---:|---:|---|
| mean_d12_k32 | 20 | 0 | -62.50 | [-81.25, -43.75] |
| mean_d24_k32 | 4 | 0 | -12.50 | [-25.00, 0.00] |
| mean_d12_k48 | 14 | 0 | -43.75 | [-65.62, -25.00] |
2000 次分层 pair-cluster bootstrap，反事实成员不拆开；开发集描述性 CI，不是正式多重检验或无损保证。
若存在 OOM，它计入未成功，同时单列；不得由只保留成功运行的分母夸大准确率。

## 读取分层

| 方法 | 模板 | 布局 | 正确 | OOM |
|---|---|---|---:|---:|
| identity_d12 | inventory | clustered | 8/8 | 0 |
| identity_d12 | inventory | distributed | 8/8 | 0 |
| identity_d12 | latency | clustered | 8/8 | 0 |
| identity_d12 | latency | distributed | 6/8 | 0 |
| identity_d24 | inventory | clustered | 8/8 | 0 |
| identity_d24 | inventory | distributed | 8/8 | 0 |
| identity_d24 | latency | clustered | 8/8 | 0 |
| identity_d24 | latency | distributed | 6/8 | 0 |
| mean_d12_k32 | inventory | clustered | 4/8 | 0 |
| mean_d12_k32 | inventory | distributed | 0/8 | 0 |
| mean_d12_k32 | latency | clustered | 6/8 | 0 |
| mean_d12_k32 | latency | distributed | 0/8 | 0 |
| mean_d12_k48 | inventory | clustered | 6/8 | 0 |
| mean_d12_k48 | inventory | distributed | 3/8 | 0 |
| mean_d12_k48 | latency | clustered | 6/8 | 0 |
| mean_d12_k48 | latency | distributed | 1/8 | 0 |
| mean_d24_k32 | inventory | clustered | 8/8 | 0 |
| mean_d24_k32 | inventory | distributed | 8/8 | 0 |
| mean_d24_k32 | latency | clustered | 8/8 | 0 |
| mean_d24_k32 | latency | distributed | 2/8 | 0 |
| native_full | inventory | clustered | 8/8 | 0 |
| native_full | inventory | distributed | 8/8 | 0 |
| native_full | latency | clustered | 8/8 | 0 |
| native_full | latency | distributed | 6/8 | 0 |

## 实际计算量

128 项 identity 检查；192 组逐层算子形状记录。
- mean_d12_k32：成功形状记录 32/32；线性 FLOPs / Full 66.8044%–66.8044%。
- mean_d24_k32：成功形状记录 32/32；线性 FLOPs / Full 83.4022%–83.4022%。
- mean_d12_k48：成功形状记录 32/32；线性 FLOPs / Full 83.4022%–83.4022%。
只统计七种线性算子，不等同于完整 attention FLOPs 或时延。

## 独立计时

所有 trial 边界未观察到其他 compute PID（不排除采样间干扰）。
4 个预先固定代表文档（每模板/布局的 instance 0、member 0），1 warmup + 5 repeats；未按正确性筛选。
每列分别取中位数；总 TTFT 中位数不一定等于两个分段中位数相加。

| 样本 | 方法 | Prefill ms | Q 首 token ms | 总 TTFT ms | Q 后 KV MiB |
|---|---|---:|---:|---:|---:|
| 0b1bb629 | native_b1024 | 3861.57 | 79.45 | 3940.92 | 2322.70 |
| 0b1bb629 | native_b4096 | 3940.45 | 79.40 | 4019.84 | 2322.70 |
| 0b1bb629 | identity_d12 | 3860.55 | 79.40 | 3939.78 | 2322.70 |
| 0b1bb629 | identity_d24 | 3860.59 | 79.33 | 3939.92 | 2322.70 |
| 0b1bb629 | mean_d12_k32 | 2267.43 | 59.56 | 2327.02 | 1554.70 |
| 0b1bb629 | mean_d24_k32 | 3018.04 | 69.44 | 3087.47 | 1938.70 |
| 0b1bb629 | mean_d12_k48 | 3005.50 | 69.36 | 3074.80 | 1938.70 |
| 6cd20ffc | native_b1024 | 3862.10 | 80.17 | 3942.26 | 2322.14 |
| 6cd20ffc | native_b4096 | 3939.07 | 80.21 | 4019.35 | 2322.14 |
| 6cd20ffc | identity_d12 | 3861.29 | 80.16 | 3941.44 | 2322.14 |
| 6cd20ffc | identity_d24 | 3862.99 | 80.29 | 3943.28 | 2322.14 |
| 6cd20ffc | mean_d12_k32 | 2267.50 | 59.86 | 2327.42 | 1554.14 |
| 6cd20ffc | mean_d24_k32 | 3012.40 | 69.82 | 3082.22 | 1938.14 |
| 6cd20ffc | mean_d12_k48 | 3005.45 | 69.94 | 3075.31 | 1938.14 |
| 731e0f2c | native_b1024 | 3861.28 | 80.37 | 3941.59 | 2322.28 |
| 731e0f2c | native_b4096 | 3938.60 | 80.33 | 4018.90 | 2322.28 |
| 731e0f2c | identity_d12 | 3864.28 | 80.38 | 3944.70 | 2322.28 |
| 731e0f2c | identity_d24 | 3860.45 | 80.17 | 3940.83 | 2322.28 |
| 731e0f2c | mean_d12_k32 | 2267.41 | 59.83 | 2327.32 | 1554.28 |
| 731e0f2c | mean_d24_k32 | 3012.48 | 69.82 | 3082.30 | 1938.28 |
| 731e0f2c | mean_d12_k48 | 3010.51 | 70.07 | 3080.59 | 1938.28 |
| d2a6c37c | native_b1024 | 3861.10 | 79.79 | 3941.04 | 2321.86 |
| d2a6c37c | native_b4096 | 3938.86 | 79.84 | 4018.72 | 2321.86 |
| d2a6c37c | identity_d12 | 3857.70 | 79.92 | 3937.63 | 2321.86 |
| d2a6c37c | identity_d24 | 3860.00 | 79.79 | 3939.65 | 2321.86 |
| d2a6c37c | mean_d12_k32 | 2267.77 | 59.47 | 2327.29 | 1553.86 |
| d2a6c37c | mean_d24_k32 | 3018.55 | 69.56 | 3088.12 | 1937.86 |
| d2a6c37c | mean_d12_k48 | 3010.14 | 69.62 | 3079.74 | 1937.86 |

### 与同轮较快原生基线比较

| 样本 | 方法 | 原生基线 | Prefill 时间减少 | 速度比 |
|---|---|---|---:|---:|
| 0b1bb629 | mean_d12_k32 | native_b1024 | 41.28% | 1.703× |
| 0b1bb629 | mean_d24_k32 | native_b1024 | 21.84% | 1.279× |
| 0b1bb629 | mean_d12_k48 | native_b1024 | 22.17% | 1.285× |
| 6cd20ffc | mean_d12_k32 | native_b1024 | 41.29% | 1.703× |
| 6cd20ffc | mean_d24_k32 | native_b1024 | 22.00% | 1.282× |
| 6cd20ffc | mean_d12_k48 | native_b1024 | 22.18% | 1.285× |
| 731e0f2c | mean_d12_k32 | native_b1024 | 41.28% | 1.703× |
| 731e0f2c | mean_d24_k32 | native_b1024 | 21.98% | 1.282× |
| 731e0f2c | mean_d12_k48 | native_b1024 | 22.03% | 1.283× |
| d2a6c37c | mean_d12_k32 | native_b1024 | 41.27% | 1.703× |
| d2a6c37c | mean_d24_k32 | native_b1024 | 21.82% | 1.279× |
| d2a6c37c | mean_d12_k48 | native_b1024 | 22.04% | 1.283× |

独立审计 PASS 只表示记录一致。每格仅四对，未达到普遍无损/鲁棒的证据要求；失败样本和 Full 原生能力限制必须保留。
