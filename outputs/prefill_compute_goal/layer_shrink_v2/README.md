# v2：96 条开发集扩展验证

## 范围

保持 v1 `engine.py` 字节不变，扩展到 96 条开发数据 / 48 个反事实样本对：
两模板 × 三长度（4K/8K/16K）× 两布局 × 每格四对。

- Full、两个 identity 边界、旧方案 d12/k32、新方案 d24/k32 和 d12/k48。
- 每长度 32 条依次执行，三个阶段不改变算法或根据成绩选样本。
- 全部 1152 条 readout/control 评分、384 个 identity probe、576 组实际算子形状。
- 性能固定选每格 instance 0 / member 0，共 12 文档，每方法 1 warmup + 5 repeats，共 504 trials。
- 2000 次分层 pair-cluster bootstrap；同时报告 Full 对→压缩错与反方向变化。
- 正式测试未读取，数学计算不纳入主指标；不使用 Goal。

## GPU 清场

本轮用户明确授权停止其他 GPU 任务。启动前已向唯一确认的 flybrain GPU compute 进程发送 SIGTERM，
确认其退出；随后无其他 compute PID，GPU 利用率为 0%。未停止 SSH 或显示服务，未修改自动启动配置。
runner 在每阶段启动检查隔离，在每个性能 trial 前后记录遥测；若新任务出现则标记污染，不自动终止新任务。

## 自检

```sh
python outputs/prefill_compute_goal/layer_shrink_v2/test_engine.py
python outputs/prefill_compute_goal/layer_shrink_v2/test_matrix.py
python outputs/prefill_compute_goal/layer_shrink_v2/test_report.py
```

- 14 项随机小 Qwen 测试；只验证实现，不验证真实任务正确率。
- 8 项矩阵/统计测试，包含临时合成的完整 32 条审计 fixture 和 14 项审计/抗篡改测试。
- fixture 明确标记为 SYNTHETIC_UNIT_TEST_ONLY，位于临时目录，不是实际 GPU 实验结果。
- 本地和 4080 的上述 engine / matrix 测试已通过。

## 有限自动流水线

从项目根目录执行，使用本地完整模型与已冻结的 development.jsonl；不下载权重。
输出目录不可覆盖，脚本会顺序执行三阶段、阶段审计、完成结果的抗篡改测试，以及最终汇总。
任一实现门槛或审计失败即停止并写状态；不会悄悄换条件或无限重试。

```sh
python outputs/prefill_compute_goal/layer_shrink_v2/pipeline.py \
  --model /path/to/Qwen3-4B-Instruct-2507 \
  --dataset outputs/validation_v3/dataset/development.jsonl \
  --out local-runs/dev96
```

输出结构：

- `PIPELINE.json`：监督进程状态与各阶段状态。
- `4096/`、`8192/`、`16384/`：每阶段的 manifest、进度、数据复验、源码快照、逐条答案、真实形状、计时、审计和中文报告。
- `REPORT_ZH.md`、`AGGREGATE.json`：仅三阶段全部成功且审计通过后生成的汇总。
- 阶段日志 / 审计日志 / 抗篡改测试日志保留在根输出目录。

已于 2026-09-19 在 4080 启动 `runs/4080-dev96-20260919-v1`。
**启动不等于完成；实时状态以远程 PIPELINE.json 和阶段 progress.json 为准。**
本地运行结果目录是同步快照，不能当作实时状态。

## 结论边界

已有开发数据的覆盖扩展，不是新种子或新模板盲测。每格仅 4 对，置信区间可能退化。
形状计算量只覆盖七类线性算子，不是全部 FLOPs；计时文档只有 12 个，不代表整个 96 条的性能分布。
OOM/失败保留；审计 PASS 只说明记录一致，不说明算法有效或无损。
