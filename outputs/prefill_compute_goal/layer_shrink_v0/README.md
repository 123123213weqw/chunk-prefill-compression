# 层间 hidden-state 缩短原型 v0

## 状态（2026-09-09）

已按普通请求完成真实 Qwen3-4B / RTX 4080 的 4K 开发预实验，没有恢复 Goal。

- v0 被真实 identity gate 拦截：mask/SDPA 路径差异。失败记录完整保留。
- 修正 mask 策略后，v1 的缓存、首次 logits 和 8 个 probe 生成 token 全部对齐。
- 均值合并真实线性 FLOPs 减少约 32.79%，prefill 时间减少约 32.8–32.9%（两个计时样本，分别 5 次测量中位数）。
- 但 readout 从 Full 4/4 降为 2/4；不能称可用、鲁棒或无损。control 均 4/4。
- 结果：runs/4080-smoke-20260909-v1/REPORT_ZH.md，独立复核 AUDIT.json。

本地与 4080 CPU 的随机小 Qwen 10 项测试通过；V100 CPU 的 9 项是修正前的历史测试，不能冒称当前版本跨环境验证。

## 实现

- engine.py：保留原始逻辑位置；每层独立 cache 长度；T 在 split_depth 后相邻两 hidden state 合并，或保留组末位置作为删除对照。S/R/Q/decode 不缩短。
- 缩短后剩余层真的接收更少输入行，并重新计算 Q/K/V/MLP；没有先跑完深层再删缓存。
- mask 根据每层 family 的实际 key 逻辑位置构造，不能用第 0 层 cache 长度套用所有层。
- 不加 log(count) 偏置：hidden-state 合并不是先前相同-key KV 合并公式的精确场景。任何加偏置变体需另列实验。
- prefix snapshot 只保存每层长度与 logical_next；probe 后逐层 crop，不将不同层裁成同一长度。不保存/恢复完整 CPU cache。

## 小模型测试

```sh
python3 test_engine.py
```

10 项：
1. identity 与原生 Qwen 的 prefill cache 和后续 logits 对齐；
2. 均值/端点和逻辑位置映射；
3. 不同层 cache 长度、实际七个线性算子输入行数；
4. future token 扰动不影响此前表示；
5. 64 与 128 block 分区对齐；
6. probe 顺序独立、prefix cache 位级不变；
7. 错误物理/逻辑位置及不完整 chunk 拒绝；
8. decode 完成追加后异常的回滚；
9. decoder layer 内 KV 更新后、元数据更新前异常的回滚；
10. 原生 / identity 的 mask 省略策略、dtype、内容逐调用对齐。

测试模型随机初始化：4 层、hidden 64、vocab 127、FP32 CPU，split_depth=2。它验证结构正确性，不具有字段读取能力，也不代表真实模型数值误差/吞吐。

## 真实模型流水线（已运行；重跑须新目录）

```sh
python -u run_experiment.py \
  --model <MODEL_ROOT>/Qwen3-4B-Instruct-2507 \
  --dataset <PROJECT_ROOT>/outputs/validation_v3/dataset/smoke4.jsonl \
  --out runs/4080-smoke-v0
```

- 必须有 CUDA，不自动回退 CPU；固定 development smoke4，不读 held-out。
- 第一门：4 条样本原生 Full vs layerwise identity；每层缓存相对 RMS <0.005，首次 Q logits 相对 RMS <0.005，KL <0.001，readout/control 的 greedy token 序列一致。任一失败即停，不把实现误差记为算法误差。
- 条件：native_full、layerwise_identity、pair_mean_d12、endpoint_d12；固定 block=1024，前 12 层完整，后 24 层 T 长度减半。
- 全部条件用相同 strict JSON 评分（validation_v3/methods.py 原样依赖并保存快照）。数学计算不作为主指标；当前只跑 readout/control。
- ShapeAudit 钩子记录各层 Q/K/V/O、gate/up/down 实际输入形状和线性 FLOPs。钩子不参与计时。
- 性能另跑：每对取 member 0、1 warmup + 5 repeats，方法顺序随机。
- 原生性能基线直接调用未修改 HF model.model，不带 engine 的位置元数据追踪开销；同时报告 b1024 与 b4096，不能只选慢基线。
- 两个计时段：S/T/R prefill；Q 到第一个 token。另报相加总 TTFT。包括在线合并、mask、位置计算和 CPU→GPU 输入传输；模型加载不计入。
- 不测完整答案延迟或 decode 吞吐；不声称已经完成 Goal 的全部验证。

失败、OOM、输出超长都保留。manifest 存参数、环境、代码和数据哈希，run 目录不可覆盖。
