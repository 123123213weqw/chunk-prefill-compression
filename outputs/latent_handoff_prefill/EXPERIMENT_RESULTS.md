# Replay-Free KV Eviction：自然残留基线实验结果

## 1. 实验定位

本轮实验测试的是 **自然 KV 残留**，不是显式压缩算法。

```text
Full:   Prefill [S][T][R] -> 使用 KV[S,T,R]
Evict:  Prefill [S][T][R] -> 删除 KV[T] -> 使用历史条件化 KV[S,R]
Replay: 不输入 T，重新 Prefill [S][R]，同时保持 R/Q 原始逻辑位置
```

本轮没有实现 `Compressor(T) -> C`、memory token 训练、KV pooling、摘要蒸馏或压缩损失。因此结果只能回答：普通 Transformer 的后续 R-KV 是否自然保留了可解码的 T 信息。

## 2. 实验环境

- 模型：`Qwen/Qwen3-4B-Instruct-2507`
- 模型位置：`<MODEL_ROOT>/Qwen3-4B-Instruct-2507`
- GPU：NVIDIA GeForce RTX 4080 16 GB
- PyTorch：2.11.0+cu130
- Transformers：5.8.0
- dtype：FP16
- attention：SDPA
- 解码：greedy，`do_sample=false`
- T 长度：512、2048、8192、16384 tokens
- R 长度：32、128、512 tokens

## 3. Canary 样本

- 从 24 个 `(template,T,R)` 网格单元各选择一条，共 24 条样本；
- 12 条 exact-fact 样本；
- 12 条 derived-workflow 样本；
- nginx、数据库迁移、Python 测试、Kubernetes rollout 四个场景家族各 3 条；
- 每个条件执行 84 道题，三条件共 252 次 probe。

## 4. Cache 正确性

| 不变量 | 结果 |
|---|---:|
| Evict 后保留的 S/R KV 与 Full 对应切片逐字节一致 | PASS |
| Replay 的 S KV 与 Full 的 S KV 逐字节一致 | PASS |
| Replay 的 R KV 至少一层不同于 Full | PASS |
| Full 物理长度为 `\|S\|+\|T\|+\|R\|` | PASS |
| Evict/Replay 物理长度均为 `\|S\|+\|R\|` | PASS |
| 三条件 Q 的逻辑起始位置相同 | PASS |

Qwen3 共 36 层；单样本 smoke test 中 Replay 的 R-KV 有 35 层不同于 Full。这只能证明 T 改变了 R 的内部状态，不能证明变化中存在可解码答案。

## 5. 答案结果

### 5.1 分信息类别

| 信息类别 | 题数 | Full | Evict | Replay |
|---|---:|---:|---:|---:|
| `T_only_exact` | 36 | 35/36（97.22%） | 0/36（0%） | 0/36（0%） |
| `T_only_derived` | 12 | 7/12（58.33%） | 0/12（0%） | 0/12（0%） |
| `R_explicit_control` | 24 | 23/24（95.83%） | 22/24（91.67%） | 19/24（79.17%） |
| `T_negative_evidence` | 12 | 12/12（100%） | 12/12（100%） | 11/12（91.67%） |
| **全部题目** | **84** | **77/84（91.67%）** | **34/84（40.48%）** | **30/84（35.71%）** |

### 5.2 主指标：T-only

真正用于判断工具信息是否被 R-KV 继承的是：

```text
T_only_exact + T_only_derived = 48 道题
```

| 条件 | T-only 正确数 | 准确率 |
|---|---:|---:|
| Full | 42/48 | 87.50% |
| Evict | 0/48 | 0% |
| Replay | 0/48 | 0% |

因此本轮答案级结果为：

> **没有任何 T-only 答案通过自然残留的 R-KV 被恢复。**

### 5.3 控制题

| 条件 | 控制题正确数 | 准确率 |
|---|---:|---:|
| Full | 35/36 | 97.22% |
| Evict | 34/36 | 94.44% |
| Replay | 30/36 | 83.33% |

`Evict 40.48%` 与 `Replay 35.71%` 的总体差异全部来自控制题，不能用作 T 信息恢复证据。

## 6. 分布级观察

第一答案 token 上记录了：

```text
inheritance_gain = KL(Full || Replay) - KL(Full || Evict)
```

部分 probe 出现正值，例如：

| Probe | mean inheritance gain |
|---|---:|
| derived-root-cause | 1.332607 |
| workflow-state | 0.187413 |
| negative-causal-control | 0.163449 |
| explicit-control | 0.018585 |

这说明 Evict 的输出分布在部分问题上比 Replay 更接近 Full，但尚未转化为任何 T-only 贪心答案。该指标仅作为探索性信号，不能替代答案级成功率。

## 7. 数据与评分修正记录

单样本 smoke test 发现 `只输出 IP:端口` 会被模型理解为输出字面字段名。现已改为具体格式示例，例如 `192.0.2.1:22`，随后重新生成全部 240 条数据，并通过 token budget、哈希、证据唯一性、泄漏和网格完整性验证。

语义评分允许严格的同义表达和词序变化，但以下情况仍判错：

- 根因回答遗漏关键因果条件；
- configured/listening 端口顺序颠倒；
- 下一步动作与 R 中明确给出的动作不一致。

## 8. 结论

1. 三路预填充、KV 删除和位置控制实现正确；
2. T 确实改变了 R 的多层 KV；
3. 当前固定、泛化、teacher-forced 的 R 是弱载体；
4. Evict 在 48 道 T-only 题上为 0%，没有答案级 latent handoff 证据；
5. 当前实验应标记为 **自然残留失败基线**；
6. 下一阶段需要单独实现真正的 `T -> C` 压缩器，再复用本实验框架进行评估。

## 9. 结果文件

- `runs/canary-grid/results.jsonl`：逐样本、逐条件、逐 probe 原始回答与分数；
- `runs/canary-grid/summary.json`：汇总指标；
- `runs/canary-grid/run_metadata.json`：模型、环境及选样记录；
- `run_prefill_experiment.py`：实验实现；
- `rescore_results.py`：从原始回答重新执行确定性评分。

