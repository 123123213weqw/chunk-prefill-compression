# Chunk Prefill Compression

研究稠密 Transformer 的 **chunk 表示压缩与首轮 prefill 计算减少**。在中间层缩短历史表示，前面层保留完整 KV，后面层处理更少位置；不依赖前缀重复，不是完整计算后才删除 KV，也没有证明无损或原生 context 扩窗。

**2026-09-21 更新：公开实验代码、已使用的合成数据、逐条回答、原始计时、缓存/算子形状检查和冻结源码。完整结果包括 v1～v4；v5 是明确标记的进行中快照。**

## 实验索引

| 阶段 | 数据与用途 | 状态 / 报告 |
|---|---|---|
| v0 | 4K smoke pilot；含失败的实现门槛记录 | [历史报告](outputs/prefill_compute_goal/layer_shrink_v0/runs/4080-smoke-20260909-v1/REPORT_ZH.md) |
| v1 | 压缩深度与线性计算预算试验 | [完成](outputs/prefill_compute_goal/layer_shrink_v1/runs/4080-depth-budget-20260919-v1/REPORT_ZH.md) |
| v2 | 96 条开发样本，4K/8K/16K | [完成](outputs/prefill_compute_goal/layer_shrink_v2/runs/4080-dev96-20260919-v1/REPORT_ZH.md) |
| v3 | 4 类任务，校准32 / 验证32；固定、自适应、随机对照 | [4B 完成](outputs/prefill_compute_goal/compression_depth_v3/runs/4080-20260920-v1/qwen4b/REPORT.md) · [0.6B 完成](outputs/prefill_compute_goal/compression_depth_v3/runs/4080-20260920-v1/qwen06b/REPORT.md) |
| v4 | 256 条新种子样本；16 份旧文档的逐 chunk 干预 | [完成：13,360 单元](outputs/prefill_compute_goal/long_validation_v4/runs/4080-20260920-24h-v1/results/REPORT.md) |
| v5 | 冻结候选复验；两种 prefill 分块；关闭诊断的独立计时 | [阶段快照：507/3,200 单元](outputs/prefill_compute_goal/confirmatory_v5/runs/4080-20260921-v1-snapshot-507/results/REPORT.md) |

**v5 不是最终结果。** 本次快照截至 2026-09-21 15:17（Asia/Shanghai），有 62 条样本已完成全部方法和两种分块的配对比较；尚无正式计时结果。快照不随服务器后台任务自动变化。

## v4：256 条新样本的结果

Qwen3-4B-Instruct-2507，RTX4080，FP16 SDPA，4K/16K 合成文本。

| 方法 | 正确 /256 | Full 原正确、压缩后错误 | Full 原错误、压缩后正确 |
|---|---:|---:|---:|
| Full | 202 | — | — |
| 第18层，64→48 均值 | 182 | 29 | 9 |
| 第27层，64→32 均值 | 199 | 9 | 6 |
| 第30层，64→16 均值 | 205 | 1 | 4 |
| 第27层，64→32 保留组末表示 | 203 | 2 | 3 |
| 重建误差自适应 | 190 | 23 | 11 |
| 随机早压 / 晚压 | 193 | 18 | 9 |

- 上表压缩条件匹配 T 部分投影/MLP 工作量节省12.5%，**不是总计算或实际耗时相同**。
- 总分相同或略高不等于无损。置信区间、错误转移、分任务结果在 `summary.json` 中。
- 候选是在 v4 后选定，必须用 v5 的新样本复验；不能把选择时的最高分当作泛化保证。
- 重建误差自适应没有显示出超过随机或固定晚压的优势。
- v4 单 chunk 干预来自已暴露的旧文档，不是新验证集；9,472次干预不是9,472个独立样本。
- 部分任务的 Full 基线低于75%，不适合据此声称稳定保真。所有失败样本仍保留。
- v3 有正式 prefill 计时；v4 是带诊断开销的机制验证，**不能把 v4 runtime 当作生产加速**。
- 同模板新种子不等于新领域，同家族不同规模不等于跨架构泛化。0.6B 原始基线过低，不能用于强泛化结论。

## 数据与复核

每个 run 包含原始记录、协议、冻结源码、源文件哈希和报告。v4/v5 原始记录位于 `results/units/*.json`；GitHub 网页可能无法一次列出全部文件，建议 clone 或下载仓库 ZIP。

安装依赖（CPU 报告审计不需要下载模型）：

```sh
python -m pip install -r requirements.txt
```

复核 v4 完整记录：

```sh
RUN=outputs/prefill_compute_goal/long_validation_v4/runs/4080-20260920-24h-v1
python "$RUN/frozen_source/report.py" --run "$RUN/results"
```

复核 v5 阶段快照，**必须保留 `--partial`**：

```sh
RUN=outputs/prefill_compute_goal/confirmatory_v5/runs/4080-20260921-v1-snapshot-507
python "$RUN/frozen_source/report.py" --run "$RUN/results" --partial
```

复核 v3：

```sh
RUN=outputs/prefill_compute_goal/compression_depth_v3/runs/4080-20260920-v1
python "$RUN/frozen_source/report.py" --run "$RUN/qwen4b"
python "$RUN/frozen_source/report.py" --run "$RUN/qwen06b"
```

重跑 v5（自行准备模型，CUDA必需；新输出目录不可覆盖）：

```sh
python outputs/prefill_compute_goal/confirmatory_v5/run.py \
  --model /path/to/Qwen3-4B-Instruct-2507 \
  --out local-runs/v5-reproduction --hours 12
```

模型、tokenizer 权重不随仓库分发。历史实验环境为 PyTorch2.11.0+cu130 / Transformers5.8.0；不同软硬件可能改变数值和性能，runner 的 identity gate 不允许跳过。`<PROJECT_ROOT>`、`<MODEL_ROOT>` 等是脱敏占位符，重跑时使用自己的路径。

## 发布边界与可追溯性

- 公开本轮使用的合成文本、gold、逐项输出、误差/形状记录、正式计时和源码。
- 不上传模型权重、KV大张量、私人服务器日志、SSH/认证/环境文件；未使用的旧锁定测试 split 仍不分发。
- 原始数据不含私人真实业务内容。公开数据已经可见，不应继续称作外部保密盲测。
- 文件系统前缀、主机别名和GPU UUID已脱敏。相应公开 `source_sha256` 已重建；**不能称脱敏源码与私有原始源码字节一致**。
- [原始与公开哈希映射](PUBLIC_PROVENANCE_20260921.json)保留修改前指纹与修改后文件哈希；[公开审计](PUBLIC_AUDIT.json)记录重新复核的范围；[文件清单](PUBLIC_SNAPSHOT.json)覆盖本次发布。

`prefill_compute_goal` 是历史目录名，不需要任何任务调度服务。尚未证明通用无损压缩或模型原生上下文扩展。
