# KV 表示压缩：数值原型

仅为合成单头 FP64 attention；不是 Qwen 准确率或性能结果。

64 个 KV 压成 16 个，旁边保留 32 个未压缩 KV，所有方法都有相同上下文。三组独立随机种子。

| 数据 | 查询分布 | 方法 | Attention 输出相对 RMS 误差（三种子均值） |
|---|---|---|---:|
| identical_within_groups | iid | mean_pool_no_mass | 0.579268 |
| identical_within_groups | iid | mean_pool_with_mass | 0.000000 |
| identical_within_groups | iid | key_cluster_with_mass | 0.000000 |
| identical_within_groups | iid | fitted_prototypes_with_mass | 0.000000 |
| identical_within_groups | large_norm_shift | mean_pool_no_mass | 0.303871 |
| identical_within_groups | large_norm_shift | mean_pool_with_mass | 0.000000 |
| identical_within_groups | large_norm_shift | key_cluster_with_mass | 0.000000 |
| identical_within_groups | large_norm_shift | fitted_prototypes_with_mass | 0.000000 |
| nearby_keys | iid | mean_pool_no_mass | 0.582443 |
| nearby_keys | iid | mean_pool_with_mass | 0.027641 |
| nearby_keys | iid | key_cluster_with_mass | 0.027641 |
| nearby_keys | iid | fitted_prototypes_with_mass | 0.026081 |
| nearby_keys | large_norm_shift | mean_pool_no_mass | 0.339732 |
| nearby_keys | large_norm_shift | mean_pool_with_mass | 0.085500 |
| nearby_keys | large_norm_shift | key_cluster_with_mass | 0.085500 |
| nearby_keys | large_norm_shift | fitted_prototypes_with_mass | 0.083776 |
| independent_keys_values | iid | mean_pool_no_mass | 1.046733 |
| independent_keys_values | iid | mean_pool_with_mass | 0.638009 |
| independent_keys_values | iid | key_cluster_with_mass | 0.550560 |
| independent_keys_values | iid | fitted_prototypes_with_mass | 0.455734 |
| independent_keys_values | large_norm_shift | mean_pool_no_mass | 1.073933 |
| independent_keys_values | large_norm_shift | mean_pool_with_mass | 1.021876 |
| independent_keys_values | large_norm_shift | key_cluster_with_mass | 0.846774 |
| independent_keys_values | large_norm_shift | fitted_prototypes_with_mass | 0.884430 |

## 检查与边界

- 相同 key 组，加 log(group_size) 后在混合上下文中的误差 <1e-12。
- 相同 key 组，不加质量修正仍产生误差，因为压缩组与外部原始 KV 的相对 softmax 权重变化。
- 两个相反 key/value 的反例：一个固定 KV 的输出不随 q 变化，不能同时匹配两种查询。
- fitted 是每个缓存单独用训练 query 优化 K/V/b 的容量试验；测试 query 未参与拟合，但这不是一次前向可部署的共享编码器。
- 大范数 query 是分布外压力测试，验证平均误差下降是否具有稳定性；没有全 query 无损保证。
- 未模拟多层误差累积、GQA、RoPE、实际自然文本 query 或自定义 attention 内核的性能。
