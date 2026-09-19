# Latent Handoff Long-Text Dataset v2

该数据集用于评测长文本条件下的 chunk 选择与 Replay-Free KV 压缩，不再把关键证据固定放在 `T` 开头。

## 固定规模

- 4 个模板：2 个精确检索模板、2 个跨 chunk 数学模板。
- `T` 长度：4096、8192、16384、32768 tokens。
- 证据位置：begin、quarter、middle、three_quarter、end。
- 每个格子 3 个确定性随机种子。
- 总计 `4 × 4 × 5 × 3 = 240` 条样本、720 个 probes。
- `R` 固定为 128 tokens。

每条样本包含：

1. 一个 `T_only_exact` 或 `T_only_derived` 主问题；
2. 一个 yes/no 平衡的 `T_negative_evidence` 反事实问题；
3. 一个答案明确位于 `R` 的 `R_explicit_control` 正控制。

## 正确性约束

生成器会在落盘前执行以下检查：

- 使用目标模型 tokenizer 精确生成 T/R token budget；
- 复算每个 S/T/R/Q 的 token ID SHA-256；
- 通过 `derivation_program` 重新计算 gold；
- 所有 evidence 在所属段中恰好出现一次；
- 证据实际位置与目标位置误差不超过 3.5%；
- 数学模板所需事实跨越至少两个 64-token chunk；
- 反事实 yes/no 类别严格平衡；
- R 不泄露 T-only 答案。

## 数据划分

- `longtext_v2_development.jsonl`：seed 0，80 条，用于选择器调参。
- `longtext_v2_test.jsonl`：seed 1/2，160 条，只用于最终评测。
- `longtext_v2_smoke16.jsonl`：每个模板 × 每个长度的 middle-position 样本，共 16 条。
- `longtext_v2.jsonl`：完整 240 条。

不得使用 test split 调整 selector 阈值。自动 selector 仍不得读取 Q、gold、probe evidence 或未来 T。

## 生成

目标 tokenizer：`Qwen/Qwen3-4B-Instruct-2507`。

```bash
python generate_longtext_dataset.py \
  --model <MODEL_ROOT>/Qwen3-4B-Instruct-2507 \
  --output-dir ./generated
```

评测端必须分别编码每个序列化 segment：

```text
encode(segment, add_special_tokens=false)
```

然后拼接 token IDs，不能重新序列化整段对话。
