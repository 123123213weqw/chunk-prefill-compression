# Latent Handoff 三路预填充实验器

实现 Qwen3 的 `Full / Evict / Replay` 三路 KV Cache 对照实验。

## 正确性原则

- 数据中的 `S/T/R/Q` 分别使用 `add_special_tokens=false` 编码；
- 加载时复算每段 token 数和 token-ID SHA-256；
- `R` 使用固定 token 做 teacher forcing；
- 所有 forward 都显式传入逻辑 `position_ids`；
- Evict 只删除各层 KV 的 T 物理切片，不重新计算 R；
- Replay 不输入 T，但 R 仍使用原来的逻辑位置；
- 不调用会根据物理缓存长度推断位置的普通 `model.generate()`；
- 每个 probe 后将 Cache 裁剪回同一个 prefix 快照。

## 运行位置

模型和 GPU 实验均在 `GPU_HOST` 上运行。本地只保存源码与结果。

模型：

```text
<MODEL_ROOT>/Qwen3-4B-Instruct-2507
```

远程实验目录：

```text
<PROJECT_ROOT>/outputs/latent_handoff_prefill
```

## 单样本 smoke test

```bash
python run_prefill_experiment.py \
  --model <MODEL_ROOT>/Qwen3-4B-Instruct-2507 \
  --dataset ./data/latent_handoff_v1.jsonl \
  --output-dir ./runs/smoke \
  --sample-id exact-t00512-r032-v00 \
  --overwrite
```

## 24 网格 canary

每个 `(template,T,R)` 网格选择一条；12 条 derived 样本额外保证四个场景家族各 3 条。

```bash
python run_prefill_experiment.py \
  --model <MODEL_ROOT>/Qwen3-4B-Instruct-2507 \
  --dataset ./data/latent_handoff_v1.jsonl \
  --output-dir ./runs/canary-grid \
  --canary-grid \
  --overwrite
```

## 输出

- `run_metadata.json`：运行环境与样本选择；
- `results.jsonl`：逐样本、逐条件、逐 probe 回答；
- `summary.json`：准确率、信息类别统计及 Cache 不变量结果。

如只修改确定性评分规则、不重新推理，可从已经保存的 raw answer 重算：

```bash
python rescore_results.py ./runs/canary-grid
```

Cache 正确性检查包括：

1. Full 物理长度为 `|S|+|T|+|R|`；
2. Evict/Replay 物理长度均为 `|S|+|R|`；
3. Evict 保留的 S/R KV 与 Full 逐字节一致；
4. Replay 的 S KV 与 Full 逐字节一致；
5. Replay 的 R KV 至少有一层不同于 Full；
6. 三路 Q 的逻辑起始位置始终为 `|S|+|T|+|R|`。
