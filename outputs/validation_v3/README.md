# v3 验证集与评测工具

固定工作目录：`<PROJECT_ROOT>/outputs/validation_v3`

## 文件

| 文件 | 用途 |
|---|---|
| `PROTOCOL.md` | 预注册设计、主次指标、门槛和解释限制 |
| `data.py` | 生成器 + 从实际 T 重解析 gold 的独立验证 |
| `dataset/all.jsonl` | 全部 240 条、120 对 |
| `dataset/development.jsonl` | 开发 96 条、48 对 |
| `dataset/test.jsonl` | 留出测试 144 条、72 对；默认禁止运行 |
| `dataset/smoke4.jsonl` | 两模板 4K 分散布局各一对，共 4 条，验证代码路径 |
| `dataset/calibration24.jsonl` | 每格一对的校准子集；不能代替完整校准解锁测试 |
| `dataset/manifest.json` | 数据文件、生成器和 tokenizer 的哈希 |
| `dataset/validation.json` | 数据正确性报告 |
| `methods.py` | 严格 JSON 评分、选择器、配对/随机种子统计 |
| `run.py` | calibration / baseline / offline / online / suite |
| `report.py` | 原始成绩、pair bootstrap、等实际预算对照、分层汇总 |
| `performance.py` | 固定工作量、warmup/重复测量的独立性能测试 |
| `freeze.py` | 开发门槛通过后冻结并解锁测试 |
| `test_methods.py` / `test_gpu.py` | CPU 单元测试 / GPU 实现不变量测试 |
| `remote_pipeline.sh` | 同步并顺序执行；不启动后台监控，不自动跑 held-out |

## 本地自检

```bash
cd <PROJECT_ROOT>/outputs/validation_v3
python3 -m unittest -v test_methods test_cli
python3 data.py --tokenizer tokenizer --output dataset --validate-only
```

已有数据不允许生成覆盖；需要重新生成时指定新的输出目录。tokenizer 文件从 4080 的实际模型目录同步，不下载或复制模型权重到本机。

## 4080 一键验证

```bash
bash <PROJECT_ROOT>/outputs/validation_v3/remote_pipeline.sh smoke
```

依次执行：远程数据复验、单元/GPU 测试、完整开发短证据校准、4 条 suite 全路径 smoke、性能 smoke，完成后同步回本地。失败会停止，日志保留。运行版本改变时用新的 RUN_LABEL，不覆盖原实验。

完整开发 baseline：

```bash
RUN_LABEL=v3-dev-review bash <PROJECT_ROOT>/outputs/validation_v3/remote_pipeline.sh development
```

该模式运行完整开发校准和完整开发 baseline，不自动消费测试集。若门槛失败，先在开发集定位，不筛掉错题，不调整 test。

## 远程手动命令

在 `<PROJECT_ROOT>/outputs/validation_v3`，Python 为 `python`，模型为 `<MODEL_ROOT>/Qwen3-4B-Instruct-2507`。

```bash
PY=python
MODEL=<MODEL_ROOT>/Qwen3-4B-Instruct-2507

$PY run.py --model "$MODEL" --dataset dataset/development.jsonl --out runs/dev-cal --phase calibration
$PY run.py --model "$MODEL" --dataset dataset/development.jsonl --out runs/dev-full --phase baseline
$PY freeze.py --calibration runs/dev-cal --baseline runs/dev-full --dataset-dir dataset --output LOCK.json

# 只有开发门槛通过且代码/参数一致才可运行：
$PY run.py --model "$MODEL" --dataset dataset/test.jsonl --out runs/test-frozen --phase suite --lock LOCK.json
$PY report.py --run-dir runs/test-frozen
```

结果目录包含 manifest/runtime/status、逐条件 events、逐样本 results，以及 summary/REPORT。进程中断后只有配置和代码哈希完全相同才能 --resume；不完整样本从头重跑，已完成 pair 不被重复计数。

## 注意

- 两数学模板的读取与计算是独立 probe，读取结果不喂给计算题。
- 原始 v1/v2 数据、代码和实验不修改。
- 当前选择器冻结为 v2 heuristic；本版改的是验证方法，不暗中根据新数据优化算法。
- 专用 performance 的固定 token decode 不是回答准确率评测。
- 测试集是方法内的工作流锁定，不是第三方盲测托管。
