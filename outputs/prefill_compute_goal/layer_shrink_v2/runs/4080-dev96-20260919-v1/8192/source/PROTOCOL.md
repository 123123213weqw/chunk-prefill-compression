# v2 完整开发矩阵：2026-09-19 执行前冻结

本版仅扩展验证覆盖，不改变 v1 engine 或任何合并规则。
不使用 Goal，不解锁/读取 test，不用 CPU 卸载或滑动 chunk。

## 数据 / 顺序

- 固定 development.jsonl：SHA256 c775d3fcaf2cc3a6f32b3c06d8b2ccec02e800d9146f4b8ad3c329c2f8e8c3c5。
- 96 条 / 48 对：2 模板 × 3 长度(4096,8192,16384) × 2 布局 × 4 个独立实例 × 2 反事实成员。
- 按 4K、8K、16K 顺序，每阶段 32 条 / 16 对；阶段间不修改代码、不选优或提前停止低分方法。
- 原来 smoke4 是本开发集子集；这是已有开发数据的覆盖扩展，不是全新盲测。
- 对全部 96 条重新解析 T 验证 gold、单字段干预、S/R/Q 一致、token 长度 / hash 和证据位置。
- 模型仅接收 S/T/R 与该 probe 的 Q；压缩器不接收 target_job、gold 或 evidence。

## 固定条件 / 实现门槛

native_full、identity_d12、identity_d24、mean_d12_k32、mean_d24_k32、mean_d12_k48。
合并规则与 v1 字节相同：k32 相邻均值；k48 每四个 [a,b,c,d]→[mean(a,b),c,d]。
Qwen3-4B-Instruct-2507、FP16、SDPA、block=1024、greedy、max_tokens=96。
两个 identity 边界对每条数据核验 KV RMS<0.005、首次 logits RMS<0.005、KL<0.001、完整 greedy token 序列一致。
门槛失败记录原因并停止该流水线（实现失效不能被当成算法损失）；门槛完成前不运行该长度的有损条件。
保留原生 b1024 HF 实现；不通过改原生计算路径达成对齐。

每条件/样本实际 hooks 汇总全部 36 层 Q/K/V/O/gate/up/down 的输入行数、维度和调用次数。
不存数百 MB 的重复逐调用 JSON，保存由真实 hooks 聚合的算子记录；独立复算各层行数及线性 FLOPs。
候选 OOM 记录 status=OOM、错误字符串，按全部样本分母计为未成功；同时报告可回答样本与失败数量。
不删除或替换失败数据；非 OOM 的意外异常使阶段 FAILED 并保留部分记录。失败阶段不伪造完整结论。

## 正确性 / 统计

主要为 readout 严格 JSON；control 辅助；不把数学计算混入主分数。
所有方法共享固定样本及 probe，Q/生成后按每层长度恢复 prefix。
报告全样本成功率、字段命中、格式错、封顶、OOM、完整 pair 两成员均对。
同时列 Full 对→压缩错、Full 错→压缩对，不用后者掩盖前者。
各长度、模板、布局分别报告；Full-correct 保持率仅辅助，分母 0 为 N/A。
主要损伤指标：method minus Full 的配对准确率差；2000 次分层 pair-cluster bootstrap，固定 seed=20260919。
按 template×length×layout 分层，保留每格原 pair 数；每对两成员始终一起重采样。
95% CI 为描述性开发统计，不做正式多重检验或无损保证；每格仅 4 对，CI 可能退化。
若 Full 在某格本身失败则保留并指出原生能力限制；不暗中筛选 Full-correct 样本作总分母。

## 性能独立测量

12 个预先固定代表文档：所有 template×length×layout 的 instance=0、member=0。
每长度 4 个，**不是全部 48 对都计时**；不根据正确性选取。
7 条件：原生 b1024 / b4096、两个 identity、三个压缩条件。
每文档每方法 1 warmup + 5 repeats；固定 seed 打乱顺序；每阶段 168 条 trial，总 504 条。
方法、总共 1152 条正确性 probe、384 条 identity probe、576 组 shape 审计均可核对覆盖。
实际耗时包含压缩、mask、位置与输入传输，不计模型加载、完整答案生成及 hooks。
每个样本的速度与同轮较快原生基线相比；不引用有并发干扰的 v1 时延作基线。

用户已明确授权停止 4080 其他 GPU 任务。执行前仅终止已识别的 compute 进程，不停 SSH/显示服务。
启动时要求无其他 compute PID；每 trial 前后记录 GPU 状态及外部 PID。
若重新出现并发或遥测失败，保留数据、标为非隔离，不自动杀新进程，不以污染数据宣称可靠加速。
边界采样不能证明采样间绝无干扰。

## 产物 / 中断

阶段目录不可覆盖；每阶段保存完整 dev96、selected sample IDs、代码 / 协议 / 模型 config 哈希。
每样本保存 gate；每 probe 保存评分；每方法保存 shape；每 trial 保存计时；状态 JSON 原子替换。
阶段成功后自动运行独立审计和抗篡改测试；三阶段全完成才输出汇总报告和全量 48 对统计。
后台 supervisor 自行顺序完成三阶段、审计、汇总；无调度器/常驻自动化，不会无限重试。
