# Latent Handoff Dataset v1

这是用于验证 Replay-Free KV Eviction 的第一版自建数据集生成器。

## 固定规模

- `exact_fact`：120 条。
- `derived_workflow_state`：120 条。
- 总计：240 条。
- 工具块 `T`：512、2048、8192、16384 tokens。
- 后缀载体 `R`：32、128、512 tokens。
- 每个 `(template, T length, R length)` 组合：10 条。

所有 token 长度均使用 `Qwen/Qwen3-4B-Instruct-2507` tokenizer 实际计算，不使用字符数近似。

## 生成

```bash
python generate_dataset.py \
  --model <MODEL_ROOT>/Qwen3-4B-Instruct-2507 \
  --output-dir ./generated
```

生成器会在写出数据前执行完整验证，任何检查失败都会终止：

- 样本 ID 唯一性；
- T/R 精确 token budget；
- token ID 哈希复算；
- R 中的答案泄漏；
- gold evidence 的存在性和唯一性；
- physical/logical KV 布局；
- 24 个全因子网格是否完整。

## 文件

- `generated/exact_fact_v1.jsonl`
- `generated/derived_workflow_v1.jsonl`
- `generated/latent_handoff_v1.jsonl`
- `generated/validation_report.json`

每个样本都已经将 `S/T/R/Q` 序列化为 Qwen ChatML 文本，并记录 token 数与 token-ID SHA-256。实验端必须采用：

```text
encode each segment with add_special_tokens=false
then concatenate token IDs
```

这种方式可以保证 Full、Evict 和 Replay 使用完全相同的 `S/R/Q` token IDs。
