# ChunkPack-Selective-v1 验证实验方案

## 1. 核心问题

需要分别验证四个命题：

1. `One-shot Full` 与“不删除 KV 的 Chunked Full”是否行为等价；
2. Chunk 结束后立即打包并删除原始 KV，后续 Prefill 是否仍可继续；
3. 不读取未来 Q/gold 的自动选择器能否保留任务信息；
4. 准确率收益能否转化为真实的 KV 显存、Prefill 和 Decode 性能收益。

当前单例 `Oracle Entity Protect` 的 3/3 只证明命题 1、2，以及命题 3 的可行性上界，不能证明泛化。

## 2. 严格因果约束

在线选择器在处理 chunk `X_j` 时只能读取：

```text
S
历史 capsule C_1...C_{j-1}
当前 chunk X_j
```

禁止读取：

```text
未来 chunk
R
Q
gold answer
测试集人工 phrase list
```

允许使用 R→T attention 的方案必须单独标记为 `post-prefill/decode-only`，不能宣称减少流式 Prefill 峰值。

## 3. 数据

### 3.1 第一阶段

使用当前 240 条 `latent_handoff_v1`：

- exact fact：120；
- derived workflow：120；
- T：512、2048、8192、16384；
- R：32、128、512；
- 24 个 `(template,T,R)` 网格，每格 10 条。

按 variant 隔离：

```text
v00-v06：训练/规则开发，168 条
v07：验证，24 条
v08-v09：盲测，48 条
```

测试前冻结选择器代码、阈值和配置哈希。

### 3.2 必须新增的鲁棒性变体

- 证据位于 T 的开头、中间、末尾；
- 关键实体恰好跨 chunk 边界；
- JSON、日志、自然语言、表格四种格式；
- 多个相似实体，仅一个为正确值；
- 否定证据与冲突版本；
- 随机 IP、端口、UUID、request ID、路径、行号；
- 重复日志比例 0%、50%、90%；
- 问题询问低 surprisal 但关键的普通词。

## 4. Chunk 参数

第一轮固定：

```text
B ∈ {64, 128, 256}
overlap ∈ {0, 8}
```

每个 chunk 的保留预算：

```text
r/B ∈ {50%, 25%, 12.5%, 6.25%}
```

对应目标压缩率：

```text
2×、4×、8×、16×
```

Chunk 优先在换行、JSON 对象结尾和日志事件边界切分；另设强制固定边界条件用于测试跨边界鲁棒性。

## 5. 对照条件

每条样本至少运行：

1. `OneShot-Full`：完整前缀一次性 Prefill；
2. `Chunked-Full`：分块 Prefill，但不删除任何 KV；
3. `Drop-All-T`：删除全部 T；
4. `Random-K`：相同预算随机保留；
5. `Last-K`：相同预算只保留最近 token；
6. `Uniform-Stride`：等间隔保留；
7. `Entity-Only`：结构化实体 span 原样保留；
8. `Surprisal-TopK`：按 token surprisal 保留；
9. `Entity+Surprisal`：实体硬保护，其余预算按 surprisal；
10. `Oracle-Entity`：读取已知相关 span，仅作为上界。

所有选择条件必须使用相同保留 token 数，避免预算不公平。

## 6. 在线 ChunkPack 算法

```text
cache = KV(S)

for X_j in chunks(T):
    Prefill X_j using original logical positions
    score tokens using only S, prior capsules, X_j
    keep selected original K/V in original order
    delete remaining K/V of X_j immediately

Prefill R using original logical positions
answer Q using the compressed physical cache
```

第一阶段只选择原始 KV，不做数值平均，避免 RoPE 相位和数字损坏。后续再增加 memory slots 与 RoPE-aware pooling。

## 7. 主要指标

### 7.1 任务保持率

Full 本身可能答错，因此主指标使用 Full-correct 子集：

\[
\operatorname{RetentionAcc}
=
\frac{
\sum_i \mathbf 1[Full_i=1\land Pack_i=1]
}{
\sum_i \mathbf 1[Full_i=1]
}
\]

分别报告：

- `T_only_exact`；
- `T_only_derived`；
- `R_explicit_control`；
- `T_negative_evidence`。

### 7.2 分布指标

- gold sequence mean log-probability；
- gold-conditioned sequence KL；
- Full 与 Pack 的 greedy answer 一致率；
- 不再只使用第一 token KL 作为主要证据。

### 7.3 压缩和性能

\[
\text{T compression ratio}
=
\frac{|T|}{\sum_j|C_j|}
\]

记录：

- 最终物理 KV token 数；
- 流式 Prefill 峰值 KV token 数；
- CUDA peak memory；
- Prefill wall time；
- 首 token latency；
- Decode tokens/s；
- 每层实际 KV bytes。

## 8. 通过门槛

### Gate A：Chunked Full 正确性

- 所有 greedy answer 与 OneShot Full 一致；
- gold-conditioned sequence KL `< 1e-3`；
- Cache 长度和逻辑 position 不变量全部 PASS。

### Gate B：Oracle 容量上界

在 Full-correct 子集：

```text
4×：exact ≥95%，derived ≥90%
8×：exact ≥90%，derived ≥80%
```

如果 Oracle 都不能通过，则当前预算或 chunk 结构不足，不继续优化自动选择器。

### Gate C：自动选择器

在盲测集、4× 压缩下：

```text
T-only exact retention ≥90%
T-only derived retention ≥80%
控制题相对 Full 下降 ≤3 个百分点
至少比 Random-K 高 15 个百分点
```

### Gate D：性能

在 T=16384 条件：

- 最终 T-KV bytes 至少减少 4×；
- 流式 Prefill 峰值 KV 至少减少 2×；
- Decode attention 长度与实际保留比例一致；
- 记录 Python 原型开销，性能结论以 GPU CUDA timing 为准。

## 9. 统计方法

- 所有比较采用同一样本上的 paired evaluation；
- 对 accuracy delta 做 10,000 次 paired bootstrap；
- 报告 95% confidence interval；
- selector 比较使用 McNemar test；
- 同时报告每个 T 长度、chunk 大小、证据位置和场景家族的结果，禁止只报总体平均。

## 10. 执行顺序

### Phase 0：24 条 Canary

```text
B=64
压缩率={2×,4×,8×}
条件={Full,ChunkedFull,Drop,Random,Entity,Entity+Surprisal,Oracle}
```

用于排除实现错误和确定合理预算。

### Phase 1：240 条数据集

跑完整 `(T length,R length,chunk size,compression ratio)` 消融，得到 accuracy–compression 曲线。

### Phase 2：鲁棒性数据

加入证据位置、跨边界、冲突和格式变体，执行冻结后的盲测选择器。

### Phase 3：GPU 性能

RTX 4080 空闲后测量真实 CUDA 峰值、Prefill latency、首 token latency 和 Decode throughput。

## 11. 最终判定

只有同时满足以下条件才称为 ChunkPack 方法成立：

1. Chunked Full 与 OneShot Full 等价；
2. Oracle 在目标预算下证明信息容量足够；
3. 不读取 Q/gold 的自动选择器在盲测集通过 Gate C；
4. GPU 实测获得与压缩率一致的内存或速度收益；
5. 跨 chunk 边界和证据位置变化不会导致不可接受的性能崩溃。

