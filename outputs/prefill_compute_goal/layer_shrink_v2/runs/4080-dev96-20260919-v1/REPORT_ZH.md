# 完整开发集 96 条 / 48 对：结果

仅覆盖两种合成模板，不是新数据盲测；反事实成员不拆开统计。

| 方法 | readout | control | 字段命中 | 读取 OOM |
|---|---:|---:|---:|---:|
| native_full | 94/96 | 96/96 | 334/336 | 0 |
| identity_d12 | 94/96 | 96/96 | 334/336 | 0 |
| identity_d24 | 94/96 | 96/96 | 334/336 | 0 |
| mean_d12_k32 | 29/96 | 96/96 | 225/336 | 0 |
| mean_d24_k32 | 90/96 | 96/96 | 330/336 | 0 |
| mean_d12_k48 | 59/96 | 96/96 | 293/336 | 0 |

## 相对 Full 损伤

| 方法 | Full 对→方法错 | Full 错→方法对 | 准确率差 (百分点) | 95% CI (百分点) |
|---|---:|---:|---:|---|
| mean_d12_k32 | 65 | 0 | -67.71 | [-77.08, -59.38] |
| mean_d24_k32 | 4 | 0 | -4.17 | [-8.33, 0.00] |
| mean_d12_k48 | 35 | 0 | -36.46 | [-48.96, -25.00] |
2000 次 template×length×layout 分层 pair-cluster bootstrap；描述性开发统计，不作正式无损/非劣证明。

## 按长度

| 方法 | T 长度 | readout | control |
|---|---:|---:|---:|
| native_full | 4096 | 32/32 | 32/32 |
| identity_d12 | 4096 | 32/32 | 32/32 |
| identity_d24 | 4096 | 32/32 | 32/32 |
| mean_d12_k32 | 4096 | 10/32 | 32/32 |
| mean_d24_k32 | 4096 | 32/32 | 32/32 |
| mean_d12_k48 | 4096 | 20/32 | 32/32 |
| native_full | 8192 | 32/32 | 32/32 |
| identity_d12 | 8192 | 32/32 | 32/32 |
| identity_d24 | 8192 | 32/32 | 32/32 |
| mean_d12_k32 | 8192 | 9/32 | 32/32 |
| mean_d24_k32 | 8192 | 32/32 | 32/32 |
| mean_d12_k48 | 8192 | 23/32 | 32/32 |
| native_full | 16384 | 30/32 | 32/32 |
| identity_d12 | 16384 | 30/32 | 32/32 |
| identity_d24 | 16384 | 30/32 | 32/32 |
| mean_d12_k32 | 16384 | 10/32 | 32/32 |
| mean_d24_k32 | 16384 | 26/32 | 32/32 |
| mean_d12_k48 | 16384 | 16/32 | 32/32 |

## 按长度、模板、布局细分

| 方法 | 长度 | 模板 | 布局 | 正确 | OOM |
|---|---:|---|---|---:|---:|
| mean_d12_k32 | 4096 | inventory | clustered | 0/8 | 0 |
| mean_d12_k32 | 4096 | inventory | distributed | 5/8 | 0 |
| mean_d12_k32 | 4096 | latency | clustered | 0/8 | 0 |
| mean_d12_k32 | 4096 | latency | distributed | 5/8 | 0 |
| mean_d12_k32 | 8192 | inventory | clustered | 2/8 | 0 |
| mean_d12_k32 | 8192 | inventory | distributed | 1/8 | 0 |
| mean_d12_k32 | 8192 | latency | clustered | 6/8 | 0 |
| mean_d12_k32 | 8192 | latency | distributed | 0/8 | 0 |
| mean_d12_k32 | 16384 | inventory | clustered | 4/8 | 0 |
| mean_d12_k32 | 16384 | inventory | distributed | 0/8 | 0 |
| mean_d12_k32 | 16384 | latency | clustered | 6/8 | 0 |
| mean_d12_k32 | 16384 | latency | distributed | 0/8 | 0 |
| mean_d12_k48 | 4096 | inventory | clustered | 3/8 | 0 |
| mean_d12_k48 | 4096 | inventory | distributed | 5/8 | 0 |
| mean_d12_k48 | 4096 | latency | clustered | 6/8 | 0 |
| mean_d12_k48 | 4096 | latency | distributed | 6/8 | 0 |
| mean_d12_k48 | 8192 | inventory | clustered | 6/8 | 0 |
| mean_d12_k48 | 8192 | inventory | distributed | 5/8 | 0 |
| mean_d12_k48 | 8192 | latency | clustered | 7/8 | 0 |
| mean_d12_k48 | 8192 | latency | distributed | 5/8 | 0 |
| mean_d12_k48 | 16384 | inventory | clustered | 6/8 | 0 |
| mean_d12_k48 | 16384 | inventory | distributed | 3/8 | 0 |
| mean_d12_k48 | 16384 | latency | clustered | 6/8 | 0 |
| mean_d12_k48 | 16384 | latency | distributed | 1/8 | 0 |
| mean_d24_k32 | 4096 | inventory | clustered | 8/8 | 0 |
| mean_d24_k32 | 4096 | inventory | distributed | 8/8 | 0 |
| mean_d24_k32 | 4096 | latency | clustered | 8/8 | 0 |
| mean_d24_k32 | 4096 | latency | distributed | 8/8 | 0 |
| mean_d24_k32 | 8192 | inventory | clustered | 8/8 | 0 |
| mean_d24_k32 | 8192 | inventory | distributed | 8/8 | 0 |
| mean_d24_k32 | 8192 | latency | clustered | 8/8 | 0 |
| mean_d24_k32 | 8192 | latency | distributed | 8/8 | 0 |
| mean_d24_k32 | 16384 | inventory | clustered | 8/8 | 0 |
| mean_d24_k32 | 16384 | inventory | distributed | 8/8 | 0 |
| mean_d24_k32 | 16384 | latency | clustered | 8/8 | 0 |
| mean_d24_k32 | 16384 | latency | distributed | 2/8 | 0 |
| native_full | 4096 | inventory | clustered | 8/8 | 0 |
| native_full | 4096 | inventory | distributed | 8/8 | 0 |
| native_full | 4096 | latency | clustered | 8/8 | 0 |
| native_full | 4096 | latency | distributed | 8/8 | 0 |
| native_full | 8192 | inventory | clustered | 8/8 | 0 |
| native_full | 8192 | inventory | distributed | 8/8 | 0 |
| native_full | 8192 | latency | clustered | 8/8 | 0 |
| native_full | 8192 | latency | distributed | 8/8 | 0 |
| native_full | 16384 | inventory | clustered | 8/8 | 0 |
| native_full | 16384 | inventory | distributed | 8/8 | 0 |
| native_full | 16384 | latency | clustered | 8/8 | 0 |
| native_full | 16384 | latency | distributed | 6/8 | 0 |

## 性能与边界

性能仅覆盖预先固定 12 个文档，每方法 1 warmup + 5 repeats；不等于 96 文档的性能分布。
全部 trial 边界未观察到其他 compute PID；仍不能排除边界之间的干扰。
不根据分数筛掉 Full 失败、压缩失败或 OOM。审计 PASS 只说明数据一致。

完整形状、逐条答案、计时和环境见三个阶段目录：
- [4096 阶段报告](4096/REPORT_ZH.md)
- [8192 阶段报告](8192/REPORT_ZH.md)
- [16384 阶段报告](16384/REPORT_ZH.md)
