# All completed chunks 64→32: 开发预实验

## 与上一轮的唯一主变化

上一轮一次只替换一个 chunk；本轮对每个真实 query，在同一个 teacher layer 中同时替换所有已经结束的 T chunk。每块 64→32，post-RoPE key 聚类和 log(count) 偏置仍使用上一轮同一实现。没有学习、未来 Q 调参或正式测试集访问。

复用上一轮 RTX 4080 上已保存的 Full Qwen FP16 Q/K/V 与真实 SDPA 输出，不重复跑模型。压缩与显式 FP32 attention 重建在同一服务器 CPU 上执行；不声称 GPU 部署或端到端速度。

## 因果范围

- S 始终原样保留。
- 对 query 的逻辑位置 p，只压缩结束位置严格小于 p 的完整 T chunk。
- 当前未完成的 T chunk、R、Q 的可见部分原样保留；包括 query 自身。
- 问题阶段全部 64 个 T chunk 均完成：4096 个 T KV→2048 个原型。
- 不同层逐层独立比较，Q/K/V 全来自未压缩 Full teacher。不是把压缩误差从第 0 层传播到第 35 层的整模型实验。
- 离线可预生成后面的 chunk 原型，但 assembly 不索引这些未来 chunk。用未来 raw KV / 原型扰动的因果自检验证。

## 样本与方法

- 冻结 smoke4：4 条 4K 开发样本，两个反事实样本对；不是 4 个独立文档。
- 层 0/9/18/27/35，全 8 KV head / 32 query head。
- 复用上一轮预定的真实查询位置，不根据本轮误差挑选 query。
- identity_raw：64→64，不改变 KV，验证 assembly 与 reference。
- mean_no_mass_fp16：均值不加偏置，错误基线。
- mean_mass_fp16：相邻两 token 均值 + log(2)。
- cluster_mass_fp16：key 聚类均值，原型 K/V 舍入 FP16，偏置 FP32。
- cluster_mass_fp32：相同聚类，保留 FP32 原型，精度诊断；64→32 时不代表节省字节。

## 正确性与统计

- identity assembly 必须精确保留可见原始 K/V。
- 改变未来原型与未来 raw KV 不改变 assembly。
- 两块相同-key分组的同时压缩在混合上下文中 max abs error <1e-6。
- 每个真实 query 的四个共享 KV 的 GQA query head，reference 输出与保存的实际 FP16 SDPA 输出相对 RMS <0.005。
- identity_raw 每个 query/head 相对 L2 <1e-6。
- 每条指标必须有限，源数据/代码/缓存 SHA256 留存。

报告输出向量相对 L2 的均值/p95/p99/max；整个上下文 logZ 误差；历史 attention mass、分 chunk logZ RMSE 和 mass L1。分别按 text/question、层、样本、已完成 chunk 数分层。大量 query/head 行高度相关，不把它们当独立文档，不做独立行 CI。

同时报告表示张量字节估算：原 raw K/V FP16，原型按各方法精度，FP32 bias；不含容器/索引/allocator，不代表程序实测内存。CPU 聚类时间不是部署性能数据。

## 判读边界

没有预先经过基准验证的应用误差门槛，本轮不设武断“通过准确率”。若平均值低但 p99 或某层很大，应报告不鲁棒。即使本轮重建良好，仍须整模型压缩后字段读取评测，才能宣称回答保持。
