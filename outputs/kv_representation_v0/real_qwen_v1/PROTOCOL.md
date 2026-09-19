# 真实 Qwen chunk KV 表示压缩：4080 开发预实验

## 本轮范围

- 使用现有 Qwen3-4B-Instruct-2507，FP16 权重，RTX 4080 单卡；不量化、不训练、不改 checkpoint。
- AutoModel 完整运行原始 token 的 Transformer，256-token 分块 prefill，不计算不需要的 LM logits。
- 来自 v3 smoke4 的 4 条 4096-token 开发样本：两类数学模板、两个反事实样本对。不是 4 个独立文档，更不是正式泛化测试。
- 层 0/9/18/27/35；每层 8 个 KV head，全部 32 个 GQA query head。
- 按 T 相对位置固定 chunk 7/31/55（零起始），每块 64 token。固定选择不使用 gold 或查询。
- 对应 chunk 后距离 1/16/64/256/1024 的可用真实 token 查询，以及 readout 问题最后 8 个 token 查询；取并集后为每个 chunk 使用所有后续可见查询。距离桶样本量并不平衡。

## 方法与数学控制

1. mean_no_mass：相邻等长组的 K/V 均值，不加质量偏置，作为错误基线。
2. mean_mass：相同分组，加 log(组大小)。
3. cluster_mass：仅使用该 chunk 的 post-RoPE K 做确定性 farthest-point 初始化和 20 次 Lloyd 迭代；组内 K/V 均值，加 log(组大小)。
4. cluster_mass_fp16：方法 3 的 K/V 原型保存为 FP16，再升为 FP32 做评估；bias 仍是 FP32。

每种方法固定 m=32/16，每个 KV head 独立构造，原型共享给其 4 个 query head。未来 query 仅评分，不进入聚类、原型生成或参数拟合。原型 K 不重复应用 RoPE。

压缩每次只替换一个完整 chunk，其他可见历史（包括 query 自身）全部保持原始 KV。不在 chunk 内提前使用含未来 token 的压缩表示。

## 数值验证

- 拦截 SDPA 入口，提取实际 post-RoPE Q 与 attention 输出，不改变 forward 运算。
- 从 Full cache 读取 K/V，用 FP32 显式 masked attention 重建。每个样本/层/KV head 的重建与真实 FP16 SDPA 输出相对 RMS <0.005 才继续。
- CPU 自检：相同 key 合并精确性、缺失质量偏置可检出、分区重组等价、重复 key 下无空簇、未来 masked KV 扰动不影响输出。
- 编码长度与 token-id SHA256 必须与冻结开发数据一致。

## 输出与解释

- mixed_relative_l2：保持其余 Full memory 时，单个 query/head 输出的相对 L2 误差。
- chunk_relative_l2：只看这个 chunk 的归一化 attention 输出误差，避免低 attention mass 掩盖信息损失。
- chunk_logZ_error：组对 softmax 分母贡献的对数误差。
- original_chunk_mass、mass_abs_error：原始 chunk 权重以及压缩后的权重绝对误差。
- 按层和 query 距离报告均值、p95、p99；query/head 有相关性，不把观测行数当独立样本量，不做独立观测 CI。
- storage.json 记录表示张量字节数（不含容器/索引/allocator）。FP32 原型 64→32 未必减少相对原 FP16 的字节数；FP16 版本才可接近 2×/4× chunk 字节压缩。
- compression_timings.json 是服务器 CPU 参考聚类耗时，不是优化 GPU 开销；不推导端到端加速。

## 本轮不能回答

不测正式测试集、压缩后整模型生成准确率、同时压缩全部历史的效果、跨层累计误差、部署内核和速度。即使 mixed error 很低，也不能由单 chunk 预实验推出全缓存压缩成功。

## 运行与文件

服务器目录：`<PROJECT_ROOT>/outputs/kv_representation_v0/real_qwen_v1/`

运行命令：

```sh
python -u run.py \
  --model <MODEL_ROOT>/Qwen3-4B-Instruct-2507 \
  --dataset smoke4.jsonl --out runs/4080-pilot
```

source/ 冻结实际执行代码与数据，manifest.json 保存哈希、环境与最终状态。V100 的失败尝试保留在独立 runs/v100-pilot，不混入结果。
