# KV 表示压缩 v0：构造方法与数学边界

日期：2026-09-07。状态：数学推导和合成数值原型完成；没有接入 Qwen，没有模型准确率或速度结论。旧 v3 验证集及其 held-out 不修改、不使用来调参。

## 1. 范围：真正合成更少的 KV，不是选择性删除

每个已完成 chunk、每一层、每个 KV head，将 n=64 个已计算出的 (K,V) 映射为 m 个 (K̃,Ṽ,b)，先研究 m=16/32。b 是 log weight，原始 KV 的 b=0。

这些条目是多个输入的组合，不是保留原 token 子集，也不需要找到对应词汇表 token。它们仅作为 attention memory，不拿来当输入 token 跑完整 Transformer。

这会减少后续访问历史 KV 的 attention 长度；仍然需要先计算原 chunk 的 token 表示。它不自动减少原始 token 的 QKV/MLP 计算，也不保证 prefill 端到端变快。若要求 64 个输入 token 只经过 16 个完整模型位置，必须另做前端 chunk encoder / 层间下采样和训练，不能冒称本原型已做到。

## 2. 必须同时保留分子与分母

对一个查询 q，定义 chunk 的贡献：

    s_i(q) = q^T k_i / sqrt(d)
    Z_c(q) = sum_i exp(s_i(q))
    N_c(q) = sum_i exp(s_i(q)) v_i

与外部未压缩上下文共存时：

    y(q) = [N_c(q) + N_rest(q)] / [Z_c(q) + Z_rest(q)]

压缩表示的目标：

    Z̃_c(q) = sum_j exp(q^T k̃_j / sqrt(d) + b_j)
    Ñ_c(q) = sum_j exp(q^T k̃_j / sqrt(d) + b_j) ṽ_j

不能只拟合 chunk 内归一化的 N_c/Z_c。给这个 chunk 的所有 logit 加同一常数，不改变其内部输出，却改变它与外部 token 的竞争质量。孤立 chunk 测试会漏掉这个错误。

## 3. 精确合并的特殊条件

若组内所有有效 key 相同，且对当前查询可见性和其他 logit bias 相同：

    k_i = k_g, i in group g
    n_g = group size
    k̃_g = k_g
    ṽ_g = mean(v_i)
    b_g = log(n_g)

则组对分母的贡献恰好为 n_g exp(q^T k_g/sqrt(d))，对分子的贡献也完全相同。因此即使旁边还有任意其他 KV，attention 输出仍相同（忽略浮点误差）。

“相同”是 attention 实际使用的 key，包括 RoPE 后的坐标；不是原始词一样、文本语义相近，或 RoPE 前的 K 相同。一个组若混合可见与不可见 token，上述证明不成立。

如果原条目已经有正权重 w_i，则合并总权重 W=sum w_i，Ṽ=sum(w_i v_i)/W，b=log W；仅相同有效 key 的情况下可精确递归合并。

反例：两个标量 key/value 分别为 (-1,-1)、(+1,+1)，原输出为 tanh(q)。只有一个固定 KV 时，单独 memory 的输出始终是同一个 Ṽ，不可能同时匹配 q=-2 和 q=2。因此任意 KV 压成一个固定 KV 并对所有未来 q 无损，不能普遍成立。

另一个解释：log Z(q) 的 Hessian 为 attention 加权 key 协方差/d；一般不是零，而一个 prototype 的 log Z̃(q)=q^T k̃/sqrt(d)+b 是仿射函数。要在任意外部上下文中精确保留质量，不能拿一个固定线性 logit 普遍代替不同 key 的 log-sum-exp。

## 4. 第一版可实现算法：带质量修正的 key 聚类

每个已完成的 64-token chunk：

1. 在该层/该 KV head 的有效 key 空间分为 m 组（初版用确定性的 farthest-point 初始化和 Lloyd）。
2. 每组计算 K 均值、V 均值、成员数 n_g。
3. 保存 (K̃,Ṽ,log n_g)，不再保存全部原始 KV。
4. 后续 attention 在缩放点积之后加 b，再与原始 KV 一起做 softmax。

伪代码：

    clusters = cluster(K_chunk, m)
    for group in clusters:
        Kp[group] = mean(K_chunk[group])
        Vp[group] = mean(V_chunk[group])
        bias[group] = log(group_size)
    logits = Q @ concat(Kp, K_raw).T / sqrt(d)
    logits += concat(bias, zeros(raw_count))
    output = softmax(logits + valid_causal_mask) @ concat(Vp, V_raw)

有相近 key 才可能近似好；不能以“语义相近”替代 attention 误差测量。固定分组均值、不加 bias 的分组均值、带 bias 的聚类、同预算删除都应作为分开的基线。

为什么需要压力测试：令 k_i=k̄+δ_i，组质量包含 mean exp(q^Tδ_i/sqrt(d))，一般不等于 1。q 范数增大或 key 分散，会放大这个误差。V 的组内差异也影响加权平均；仅看 key 距离不是完整保证。

## 5. 第二版：拟合／学习少量 prototypes

先将 K̃/Ṽ/b 直接设为优化变量，使用与测试查询独立的训练 query 拟合：

    L = MSE(log Z̃, log Z)
        + normalized_MSE(Ñ/Z̃, N/Z)
        + normalized_MSE(ỹ_mixed, y_mixed)

最后两项分别约束 chunk 内输出和与外部记忆混合后的输出。为避免只对训练 query 好，需要独立验证/测试 query，另外测试 query 范数、方向和未来距离的偏移。

toy_compression.py 的 fitted 方法是“每个缓存单独拟合”的容量实验，不是一个共享 compressor。它虽未使用测试 query，但也不能当作无需未来查询、一次前向可部署的方法。

若容量实验值得继续，下一步训练共享 encoder：

    C_theta(K_chunk, V_chunk, layer/head context) -> (K̃,Ṽ,b)

teacher 是冻结模型在开发/训练文档产生的 attention 贡献；未来训练 query 只能作为训练监督，不能成为部署 encoder 的输入。验证按文档/样本对分开，不能只将同一个缓存的 query 随机拆分后宣称跨文档泛化。正式测试时 encoder 只读已生成的 chunk KV，不读问题、gold、未来 token 或未来 query。

## 6. 工程接入约束

- 原始 KV 与压缩 KV 必须都维护 log_weight；未压缩位置权重为 1，即 bias=0。
- 不能只把均值写进普通 DynamicCache 就认为实现正确：attention logit 路径需要加 bias，并保留原 causal/padding mask。
- 先用显式 PyTorch reference attention 检查数值；之后再验证 SDPA/自定义 kernel。不要为了支持 bias 导致后端退化却仍声称加速。
- 如果压缩的是 RoPE 后的 K，就直接在那个坐标系使用 prototype，不能再给它应用一次虚构平均位置的 RoPE。未来 Q 的逻辑位置仍按原序列前进。
- 仅压缩已经完成、对后续 query 全部可见的 chunk；不让包含未来 token 的 prototype 被 chunk 内较早 query 看到。
- GQA 的多个 query heads 共用 KV head，必须共同评估这些 query heads 的重建误差；不能偷偷给每个 query head 独立 KV 后仍按原 GQA 字节数声称节省。
- 单层误差不会自动等于整模型答案误差。必须逐层接入并测累计影响。
- 64→16 只是 chunk 条目数 4×；额外 bias/索引会略减字节压缩比。toy 中另有 32 个原始 KV，所以全 memory 是 96→48，不是全局 4×。

## 7. 本次完成的数值实验

- 三种结构：组内相同 key、相近 key、独立随机 K/V。
- 三个种子，每种固定 64→16。
- 每例加入 32 个未压缩外部 KV，避免分母误差被内部 softmax 掩盖。
- 独立 train=256、validation=128、test IID=512、test query 范数放大 4×=512。
- 精确组的质量修正必须达到 <1e-12 相对 RMS 误差；不加质量的错误版本必须被检出。
- 同预算比较普通均值、质量修正均值、key 聚类、拟合 prototypes。
- 结果见 TOY_REPORT.md / toy_results.json。这里只能证明公式与数值原型行为，不证明真实 LLM 可压缩或可加速。

本次观察：相近 key 的带质量均值 IID 输出误差约 2.76%，大范数 query 约 8.55%；独立随机 K/V 的聚类 IID 约 55.06%，拟合后约 45.57%，大范数压力测试仍约 84.68%/88.44%。不能把“可拟合”直接说成“鲁棒”。

## 8. 下一步只有一个主实验

从 Qwen 的开发样本抓取真实、已完成 chunk 的 K/V，以及仅用于评估的后续真实 Q。先在少量 layer/head 上比较 64→32 和 64→16 的 attention 重建：

    原始 KV vs 均值无质量修正 vs 聚类+质量修正

方法选择不读评估 Q；查询只用来测误差。报告 mixed output 相对误差、logZ 误差、p95/p99、query 距离分层、实际 bytes 和压缩开销。先不做滑动窗口、语义淘汰或 CPU offload。

只有真实 attention 重建可接受，才改整模型 attention 接口并复用 v3 的字段读取评测。数学计算由于基础校准失败继续作为诊断，不拿它做主要成功指标。

## 9. 已查阅的相关工作

- WeightedKV（ICASSP 2025）：保留部分 key 作为锚点，按平均 attention 权重将被移除 token 的 value 合并到邻居。它说明 merging 路线已有工作；这里的聚类 prototype + mass baseline 不是该论文的复现，也不声称首创。https://arxiv.org/abs/2503.01330 ，全文 https://arxiv.org/html/2503.01330v1
- Compressive Transformer：训练式历史记忆压缩路线，并非直接给现有冻结 Qwen 做一个无损替换。https://arxiv.org/abs/1911.05507

以上是针对当前工程目标的推导与候选设计，不是全面文献综述或新颖性结论。
