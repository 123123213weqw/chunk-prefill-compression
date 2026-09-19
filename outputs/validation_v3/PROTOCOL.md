# v3 预注册协议

## 研究范围与冻结规则

研究对象是 Qwen3-4B-Instruct-2507（FP16）的 KV 选择性保留，不是可逆 token 编码，也不是训练过的 latent compressor。

本版刻意采用 **query-blind** 压缩：压缩时 S 不指出目标 job，自动方法不接收 Q、gold、target_job 或 evidence metadata。目标 job 在压缩完成后的 Q 才出现。因此任务是“未知未来查询的信息保留”，不能假设所有事实都能无损删除。任务已知的 selector 可另做后续实验，但不得混入本版。

算法基线继续使用 v2 的 lexical span score 与 fingerprint 去重规则，名称 semantic 不意味着真正理解语义。它会规范化数字/十六进制，可能错误合并不同 job 的事实；这属于被评估算法的缺陷，不依据测试答案临时修正。

生成器、实际数据、tokenizer 文件、运行参数和依赖代码均记录 SHA-256。改问题、模板、评分或选择器必须开新版本/运行目录，不能覆盖失败记录。

## 验证集

- 2 个模板：库存 `shards × items − rejected`；延迟 `base + retries × penalty − discount`。
- T 为 4096 / 8192 / 16384 个实际 tokenizer tokens。
- 两种布局：集中、全文分散。分散布局的相邻必要事实 token 起点间距 ≥0.15T。
- 每个单元 10 个实例，每实例两个反事实版本，共 240 条 / 120 对。
- 实例 0–3：development 96 条 / 48 对；实例 4–9：test 144 条 / 72 对。
- 配对成员 S/R/Q 一致、T token 长度及证据位置一致；只改一个目标字段的一行数值；读取和计算 gold 都必须改变。
- 不同 job 的干扰记录与目标记录使用相同字段、相同结构、相近数值范围。第一版不引入旧版本覆盖和自然文本。
- 不出现 evidence_bundle、样本编号、pair_id 或标签奇偶提示。自动 selector 不得到这些元数据。
- R 为固定中性后缀及正控制动作；不含目标作业或数值。R 实际长度固定，但不机械填充到 128。
- 同一 pair 不跨 split；所有统计以 pair 为独立单位。

每条含三个独立 probe：
1. readout：字段读取 JSON，整数类型严格匹配；主要能力。
2. calculation：仅输出 `{"result":整数}`，由模型自行计算；若校准失败，仅作为诊断。
3. control：R 中动作 JSON，检查保留后缀可读取。

所有 probe 从同一个前缀 KV 独立发起。前一个 probe 的问题、答案、teacher-forcing token 都 crop 掉；不允许给下一题提示。

## 阶段与准入门槛

1. **数据与实现自检**：原文重解析 gold、配对单变量干预、token/hash、分散间距；单元测试、GPU KV 选择/位置/crop 自检。
2. **calibration**：96 条开发样本，仅使用必要证据 short_T、原 S/R/Q。readout ≥95%、control ≥95%；calculation ≥90% 才能设为主要指标。
3. **baseline**：完整开发集，Full（b1024 无删除）、chunked_full（b64 无删除）、Drop-T cached-R、Drop-T replay-R。每种长度 Full readout ≥90% 才允许解锁正式测试。其余错误保留，不筛掉失败样本。
4. **离线选择**：完整 prefill 之后从同一 Full KV 快照选行，原 S/R KV 不重算。此阶段只研究“删哪些 KV”，不宣称节省 prefill。
5. **在线选择**：逐 64 token 前向并淘汰，后续 KV 的历史依赖会变化。另以最终相同位置从 Full KV 选取，隔离选择位置与生成历史的差异。
6. **系统性能**：独立运行 performance.py；不使用含 CPU snapshot 的离线诊断时间声称加速。

默认数学失败不阻止研究字段保留，但必须在正式锁文件中写入 `calculation_primary=false`。Full 字段能力失败则保持 test 锁定。开发/实现 smoke 不能充当正式测试通过。

## 方法与公平性

预算倍率 2× / 4× / 8×；随机种子 0–4，种子由 pair_id 和 replication 确定，配对两成员使用相同随机性。

离线：Semantic-span、Random-token、Random-span、Oracle-evidence。
- Semantic 的倍率只给预算上限，记录实际 K。
- 主要对照是 **x4 Semantic 与实际 K 完全一致的 5 个 Random-token 结果**。
- Random-span 用带随机顺序的 subset-sum 保留完整 token 行，尽量达到相同 K；这不是对所有 span 子集的均匀抽样。
- Random-span 不能达到 K 时报告缺口，并额外生成与它实际 K 完全一致的 Random-token。不能忽略预算差异。
- Oracle 允许知道 evidence，只是泄漏参考；预算容不下证据时记 BUDGET_INFEASIBLE，不截断后冒充完整 Oracle。

在线：冻结的 Semantic-span、5 个 Random-token、5 个因果 Random-span priority reservoir。
- token 随机位置在读取内容前生成；span reservoir 只接收当前已到达文本，不复活被删 KV。
- 在线各方法在名义预算下报告实际 KV 曲线；**不利用未来最终 K 来冒充事先可部署的等量在线随机基线**。
- 未完成行允许临时缓存，报告瞬时 peak；最终 K 不超过预算。长行最坏内存界不属于本版短记录测试结论。
- Full-history same positions 是事后机制诊断，不计入自动方法。

## 评分与统计

- 严格 JSON；字段集合、值与类型必须匹配。忽略 JSON 键顺序和包裹整段 JSON 的代码围栏。
- 不接受额外解释、重复字段、缺字段、额外字段、用字符串/浮点/bool 代替整数。
- 格式错误和内容错误分开记录；程序只计算 gold，不能将程序算出的模型输出改记为模型推导正确。
- 报告 readout/calculation/control 原始准确率、字段级命中、完整证据召回、反事实两成员同时正确比例。
- 相对 Full 的配对准确率差为主要损伤指标；Full-correct 保持率仅辅助，分母 0 为 N/A。
- 2000 次 pair-cluster bootstrap，配对成员不拆开。五个随机种子先在每个成员内平均，再按 pair 重采样，不扩张独立样本量。
- 主要假设：离线 x4、实际 K 相等时 Semantic 相对 Random-token 的 readout 差值。正式 test 中 CI 下界 >0 才称有统计上的优势；其他倍率/布局/数学仅分层或探索性。
- 同时报告资源与相对 Full 的损伤，不能仅凭“胜过随机”声称实用或无损。
- 开发 smoke 只有 2 对，bootstrap 可能退化，任何 CI 都不能作为泛化证据。
- 分块实现诊断同时报告 sequence KL 与字面一致性；最大 KL<1e-3 是诊断阈值，不能单凭措辞差异判为任务错误。
- 未运行为 NOT_TESTED，未满足分母/覆盖为 NOT_EVALUABLE，失败为 FAILED/OOM；不完整 pair 不进统计，但保留 events 供故障定位。

## 系统测量

- 同一模型、dtype、设备；记录 GPU 状态，不终止其他用户进程。
- 默认每个方法 1 次 warmup、5 次测量；同一 sample/trial 内随机化方法顺序。
- 固定生成 64 tokens（忽略 EOS），报告 prefix 时间、query TTFT、后续 63 token 吞吐、总时间、KV bytes 和峰值 allocated/reserved。
- 主比较使用共同 b64 的未压缩与在线方法；另报 b1024 未压缩基线，避免因选择慢基线夸大加速。
- 在线选择器、同步、KV 搬移和 Python 开销计入端到端计时。按实际 KV bytes 报告；不将一次运行的最大值说成稳定收益。
- 32K 不在本版正确性矩阵，后续单独解决内存容量；不得把分块 Full 偷换名称为 one-shot。

## 测试集冻结

freeze.py 必须检查同一套代码、同一模型下完整的开发校准和开发 baseline，以及门槛。产生 LOCK.json 后，run.py 才接受 test.jsonl；代码/数据/参数/模型改变均拒绝。test 只运行完整 suite（2/4/8、五个种子）。不因测试结果调整阈值。

这只是本地工作流保护，不是密码学意义的盲测托管：数据生成者可以读取 test 文件。若用于论文，还需外部保管或独立生成最终测试实例。
