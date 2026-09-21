# 深度 / 压缩率：8K 完整开发阶段

固定 32 条 / 16 对，覆盖两模板、两布局、每格四对。已有开发数据，不是独立盲测。

## 正确性（OOM 包含在总分母，另外报告）

| 方法 | readout | control | 字段命中 | 两成员均对 | 格式错 / 封顶 / OOM |
|---|---:|---:|---:|---:|---:|
| native_full | 32/32 | 32/32 | 112/112 | 16/16 | 0 / 0 / 0 |
| identity_d12 | 32/32 | 32/32 | 112/112 | 16/16 | 0 / 0 / 0 |
| identity_d24 | 32/32 | 32/32 | 112/112 | 16/16 | 0 / 0 / 0 |
| mean_d12_k32 | 9/32 | 32/32 | 77/112 | 3/16 | 0 / 0 / 0 |
| mean_d24_k32 | 32/32 | 32/32 | 112/112 | 16/16 | 0 / 0 / 0 |
| mean_d12_k48 | 23/32 | 32/32 | 101/112 | 10/16 | 0 / 0 / 0 |

## 相对 Full 的损伤

| 方法 | Full 对→方法错 | Full 错→方法对 | 准确率差 (百分点) | 95% 配对 CI (百分点) |
|---|---:|---:|---:|---|
| mean_d12_k32 | 23 | 0 | -71.88 | [-84.38, -56.25] |
| mean_d24_k32 | 0 | 0 | 0.00 | [0.00, 0.00] |
| mean_d12_k48 | 9 | 0 | -28.12 | [-46.88, -9.38] |
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
| mean_d12_k32 | inventory | clustered | 2/8 | 0 |
| mean_d12_k32 | inventory | distributed | 1/8 | 0 |
| mean_d12_k32 | latency | clustered | 6/8 | 0 |
| mean_d12_k32 | latency | distributed | 0/8 | 0 |
| mean_d12_k48 | inventory | clustered | 6/8 | 0 |
| mean_d12_k48 | inventory | distributed | 5/8 | 0 |
| mean_d12_k48 | latency | clustered | 7/8 | 0 |
| mean_d12_k48 | latency | distributed | 5/8 | 0 |
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
- mean_d12_k32：成功形状记录 32/32；线性 FLOPs / Full 66.9411%–66.9411%。
- mean_d24_k32：成功形状记录 32/32；线性 FLOPs / Full 83.4705%–83.4705%。
- mean_d12_k48：成功形状记录 32/32；线性 FLOPs / Full 83.4705%–83.4705%。
只统计七种线性算子，不等同于完整 attention FLOPs 或时延。

## 独立计时

所有 trial 边界未观察到其他 compute PID（不排除采样间干扰）。
4 个预先固定代表文档（每模板/布局的 instance 0、member 0），1 warmup + 5 repeats；未按正确性筛选。
每列分别取中位数；总 TTFT 中位数不一定等于两个分段中位数相加。

| 样本 | 方法 | Prefill ms | Q 首 token ms | 总 TTFT ms | Q 后 KV MiB |
|---|---|---:|---:|---:|---:|
| 827a434d | native_b1024 | 1400.86 | 49.03 | 1449.88 | 1170.00 |
| 827a434d | native_b4096 | 1543.43 | 49.20 | 1592.65 | 1170.00 |
| 827a434d | identity_d12 | 1398.00 | 48.91 | 1446.89 | 1170.00 |
| 827a434d | identity_d24 | 1397.82 | 48.88 | 1446.67 | 1170.00 |
| 827a434d | mean_d12_k32 | 879.24 | 39.20 | 918.48 | 786.00 |
| 827a434d | mean_d24_k32 | 1140.42 | 44.10 | 1184.50 | 978.00 |
| 827a434d | mean_d12_k48 | 1137.97 | 43.87 | 1181.84 | 978.00 |
| 8bec401e | native_b1024 | 1399.30 | 49.55 | 1448.93 | 1170.70 |
| 8bec401e | native_b4096 | 1542.57 | 49.74 | 1592.30 | 1170.70 |
| 8bec401e | identity_d12 | 1400.35 | 49.47 | 1449.91 | 1170.70 |
| 8bec401e | identity_d24 | 1400.60 | 49.56 | 1450.22 | 1170.70 |
| 8bec401e | mean_d12_k32 | 878.84 | 39.74 | 918.56 | 786.70 |
| 8bec401e | mean_d24_k32 | 1138.70 | 44.73 | 1183.43 | 978.70 |
| 8bec401e | mean_d12_k48 | 1138.09 | 44.44 | 1182.65 | 978.70 |
| a458cd77 | native_b1024 | 1400.55 | 49.02 | 1449.57 | 1170.00 |
| a458cd77 | native_b4096 | 1543.83 | 49.20 | 1593.02 | 1170.00 |
| a458cd77 | identity_d12 | 1400.11 | 48.96 | 1449.06 | 1170.00 |
| a458cd77 | identity_d24 | 1400.28 | 48.95 | 1449.20 | 1170.00 |
| a458cd77 | mean_d12_k32 | 878.84 | 39.19 | 918.04 | 786.00 |
| a458cd77 | mean_d24_k32 | 1140.42 | 44.08 | 1184.50 | 978.00 |
| a458cd77 | mean_d12_k48 | 1137.64 | 43.84 | 1181.53 | 978.00 |
| be1c94bb | native_b1024 | 1400.90 | 49.63 | 1450.51 | 1170.42 |
| be1c94bb | native_b4096 | 1543.92 | 49.81 | 1593.73 | 1170.42 |
| be1c94bb | identity_d12 | 1400.02 | 49.60 | 1449.61 | 1170.42 |
| be1c94bb | identity_d24 | 1400.47 | 49.58 | 1450.08 | 1170.42 |
| be1c94bb | mean_d12_k32 | 879.33 | 39.67 | 919.04 | 786.42 |
| be1c94bb | mean_d24_k32 | 1138.85 | 44.64 | 1183.53 | 978.42 |
| be1c94bb | mean_d12_k48 | 1136.62 | 44.34 | 1180.96 | 978.42 |

### 与同轮较快原生基线比较

| 样本 | 方法 | 原生基线 | Prefill 时间减少 | 速度比 |
|---|---|---|---:|---:|
| 827a434d | mean_d12_k32 | native_b1024 | 37.24% | 1.593× |
| 827a434d | mean_d24_k32 | native_b1024 | 18.59% | 1.228× |
| 827a434d | mean_d12_k48 | native_b1024 | 18.77% | 1.231× |
| 8bec401e | mean_d12_k32 | native_b1024 | 37.19% | 1.592× |
| 8bec401e | mean_d24_k32 | native_b1024 | 18.62% | 1.229× |
| 8bec401e | mean_d12_k48 | native_b1024 | 18.67% | 1.230× |
| a458cd77 | mean_d12_k32 | native_b1024 | 37.25% | 1.594× |
| a458cd77 | mean_d24_k32 | native_b1024 | 18.57% | 1.228× |
| a458cd77 | mean_d12_k48 | native_b1024 | 18.77% | 1.231× |
| be1c94bb | mean_d12_k32 | native_b1024 | 37.23% | 1.593× |
| be1c94bb | mean_d24_k32 | native_b1024 | 18.71% | 1.230× |
| be1c94bb | mean_d12_k48 | native_b1024 | 18.86% | 1.233× |

独立审计 PASS 只表示记录一致。每格仅四对，未达到普遍无损/鲁棒的证据要求；失败样本和 Full 原生能力限制必须保留。
