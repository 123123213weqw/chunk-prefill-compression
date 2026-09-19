# Chunk Prefill Compression

研究 Transformer 的 **chunk 表示压缩与 prefill 实际计算减少**。基于 Qwen3-4B-Instruct-2507，包含数学原型、真实 attention 重建、层间 hidden-state 缩短和可复核的开发实验。

> **当前结论：确实少算、能够提速，但信息保真不足。不是无损压缩，也不是已验证可部署的方法。**

## 最新真实模型实验

RTX 4080 / FP16 / Qwen3-4B-Instruct-2507。前 12 层完整处理原始 token；在层间把 T 的每个 64-token chunk 变为 32 个表示，再运行后 24 层。S/R/Q 和解码 token 保持完整。

| 指标 | Full | 层间相邻均值合并 |
|---|---:|---:|
| 字段读取正确 | 4/4 | **2/4** |
| 后缀控制题正确 | 4/4 | 4/4 |
| 线性层 FLOPs（实际算子形状推算） | 100% | **67.21%** |
| Prefill 中位耗时 | 612–621 ms | **410–417 ms** |

- 两个计时样本、每方法 1 warmup + 5 repeats；相对 b1024/b4096 中较快原生基线，prefill 时间减少 **32.8%–32.9%**，约 **1.49×**。
- 正确性只有 **4 条开发样本 / 2 个反事实样本对**。不能将其当作正式泛化评测，也不能由向量误差换算答题准确率。
- 未压缩 layerwise 实现与原生模型的被检缓存、首次 logits 和 8 个 probe 的生成 token 序列完全一致。
- 上层 QKV/MLP 确实接收更少输入行；这不是先计算全部 KV 再丢弃。
- 运行保存了失败版本：最初的 mask 策略导致 identity gate 失败，修正后未放宽门槛。

[完整报告](outputs/prefill_compute_goal/layer_shrink_v0/runs/4080-smoke-20260909-v1/REPORT_ZH.md) · [独立审计结果](outputs/prefill_compute_goal/layer_shrink_v0/runs/4080-smoke-20260909-v1/AUDIT.json) · [mask 修正说明](outputs/prefill_compute_goal/layer_shrink_v0/MASK_FIX_20260909.md)

## 代码结构

| 路径 | 内容 |
|---|---|
| `outputs/prefill_compute_goal/layer_shrink_v0/` | 当前主线：逐层变长 KV、hidden-state 合并、自检、真实模型 runner、报告审计 |
| `outputs/prefill_compute_goal/cost_model.py` | 解析计算量账本，**不是实测速比** |
| `outputs/kv_representation_v0/` | 带权 KV 合并数学 toy、单 chunk / 多 chunk attention 重建 |
| `outputs/validation_v3/` | 合成数据生成、严格 JSON 评分、配对评测与冻结控制 |
| `outputs/chunkpack_validation_v1/` | 历史 chunk prefill / KV 选择实验与共享依赖 |
| `outputs/latent_handoff_*` | 更早的数据和实验代码 |

目录保留研究过程中的命名；`prefill_compute_goal` 只是目录名，不依赖任何任务调度服务。历史代码和文档可能引用未分发的私有中间产物，公开主入口如下。

## 安装与小模型自检

建议 Python 3.11。先按硬件环境安装合适的 PyTorch，再安装其余依赖：

```sh
python -m pip install -r requirements.txt
python outputs/prefill_compute_goal/layer_shrink_v0/test_engine.py
python outputs/kv_representation_v0/toy_compression.py
python outputs/prefill_compute_goal/cost_model.py
```

小模型自检不下载 Qwen 权重，测试随机小 Qwen 的因果性、cache 恢复、mask 策略与真实算子行数。它不测应用正确率。

历史真实 GPU 实验环境为 PyTorch `2.11.0+cu130` / Transformers `5.8.0`。不同硬件、依赖和内核可能改变数值与性能；runner 会先做原生对齐检查。

## 重跑真实 4K 开发实验

自行取得并遵守 Qwen3-4B-Instruct-2507 的模型许可，将完整 Hugging Face 模型目录放到本地。仓库不分发模型和 tokenizer 权重资产。

```sh
export MODEL_DIR=/path/to/Qwen3-4B-Instruct-2507
python outputs/prefill_compute_goal/layer_shrink_v0/run_experiment.py \
  --model "$MODEL_DIR" \
  --dataset outputs/validation_v3/dataset/smoke4.jsonl \
  --out local-runs/layer-shrink-4k
```

真实模型 runner 要求 CUDA，不悄悄回退 CPU。输出目录不可覆盖。报告审计：

```sh
python outputs/prefill_compute_goal/layer_shrink_v0/report_experiment.py \
  local-runs/layer-shrink-4k
```

复核已发表的这一轮数据，不需模型权重或 GPU：

```sh
python outputs/prefill_compute_goal/layer_shrink_v0/report_experiment.py \
  outputs/prefill_compute_goal/layer_shrink_v0/runs/4080-smoke-20260909-v1
```

正式测试集仍未公开。`smoke4.jsonl` 为程序生成的开发数据；不包含私人真实业务记录。生成器公开，因此任何可由它重建的旧 split 都不能再被当作外部保密盲测；后续正式评测需要独立的新测试数据。

## 发布范围与局限

- 提交代码、方案、开发 smoke4 和最新实验的审计所需小型 JSON/source 快照。
- 不提交模型权重、tokenizer 资产、原始 Q/K/V 张量、批量 attention 观测、认证信息、SSH 配置、服务器日志或完整数据集。
- 历史报告中的私有文件系统前缀已脱敏。最新实验 `source/` 的代码和 smoke 数据保持字节不变，原始 SHA256 可复核；manifest 中环境路径是占位符。
- 个别历史报告/审计脚本需要未分发的中间产物，不保证仅靠此快照可以审计全部历史阶段。
- 尚未实施后续“第 24 层后 64→32”或“第 12 层后 64→48”对照，不能将建议当作结果。

发布文件清单及 SHA256：[`PUBLIC_SNAPSHOT.json`](PUBLIC_SNAPSHOT.json)。
