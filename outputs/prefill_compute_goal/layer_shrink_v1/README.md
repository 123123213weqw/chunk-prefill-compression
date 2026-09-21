# 层间缩短 v1：晚压缩 vs 轻压缩

执行前固定协议见 [PROTOCOL.md](PROTOCOL.md)。不改写 v0 实验记录，不使用 Goal。

## 本轮问题

旧条件 `mean_d12_k32` 同轮重跑；比较新增的 `mean_d24_k32` 与 `mean_d12_k48`。
两个新条件具有相同的理论线性计算预算，但压缩深度、表示数量不同。
固定 64→48 为四元组 `[a,b,c,d] → [(a+b)/2,c,d]`，不以答案选择位置。

## 文件

- `engine.py`：层间缩短、原始逻辑位置、变长 per-layer cache。
- `specs.py`：冻结条件及开发数据 SHA256。
- `run_experiment.py`：两种 split 的原生对齐门槛、正确性、实际形状、并发遥测、性能。
- `report_experiment.py`：独立复算评分、gold、记录覆盖、计算量及计时；自动生成中文报告。
- `test_engine.py`：14 项小随机 Qwen CPU 测试；不代表真实任务正确率。
- `test_report.py`：6 项独立审计基础测试；提供完成 run 后的 8 项抗篡改测试。

## 自检

```sh
python outputs/prefill_compute_goal/layer_shrink_v1/test_engine.py
python outputs/prefill_compute_goal/layer_shrink_v1/test_report.py
```

## 真实模型实验

从项目根目录执行；需要 CUDA、完整本地 Qwen 模型与固定 smoke4。不会下载模型，不回退 CPU。
输出目录必须从未存在。历史环境 Python 3.11 / Torch 2.11.0+cu130 / Transformers 5.8.0。

```sh
export MODEL_DIR=/path/to/Qwen3-4B-Instruct-2507
export RUN_DIR=local-runs/depth-budget-4k
python outputs/prefill_compute_goal/layer_shrink_v1/run_experiment.py \
  --model "$MODEL_DIR" --dataset outputs/validation_v3/dataset/smoke4.jsonl --out "$RUN_DIR"
python outputs/prefill_compute_goal/layer_shrink_v1/report_experiment.py "$RUN_DIR"
RUN_DIRECTORY="$RUN_DIR" python outputs/prefill_compute_goal/layer_shrink_v1/test_report.py
```

runner 冻结脚本、协议、数据和评分器。审计会拒绝不完整记录、重复样本、被更改的 gold、
错误计算量或与原始 trial 不一致的中位数。审计 PASS 只表示记录一致，不表示压缩方法有效。

## 本轮结果（2026-09-19，已完成）

结果：[中文报告](runs/4080-depth-budget-20260919-v1/REPORT_ZH.md)、[审计 JSON](runs/4080-depth-budget-20260919-v1/AUDIT.json)。

| 条件 | readout | control | 字段命中 | 线性计算量 / Full |
|---|---:|---:|---:|---:|
| 原生 Full | 4/4 | 4/4 | 14/14 | 100% |
| 旧方案 d12 / 64→32 | 2/4 | 4/4 | 10/14 | 67.21% |
| 晚压缩 d24 / 64→32 | 4/4 | 4/4 | 14/14 | 83.61% |
| 轻压缩 d12 / 64→48 | 4/4 | 4/4 | 14/14 | 83.61% |

- 两个新方案均少算约 16.39% 线性运算，并通过这 4 条开发数据。不是全部 attention FLOPs。
- 旧方案的库存字段错误在本轮复现；两个新方案在库存和延迟模板均为 2/2。
- 本机与 4080 的 14 项小随机模型测试全部通过。
- 真实模型两种 split 共 16 项 identity 检查通过，缓存与首次 logits 最大相对 RMS 为 0。
- 48 条正确性记录、24 组形状审计、84 条含 warmup 的计时记录已独立复核。
- 本机与 4080 的 14 项审计测试（6 基础 + 8 完成 run / 抗篡改）均通过。
- GPU 并发任务仍在，未中断他人进程。本轮耗时非隔离，仅供诊断，不作可靠提速结论。
- 只有旧开发样本 4 条 / 2 对，不能称无损或鲁棒；两个候选均可进入更大的、预先固定的开发集对照。
- 正式测试集未读取；未启动 Goal、CPU 卸载或滑动 chunk。
