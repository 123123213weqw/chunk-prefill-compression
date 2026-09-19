# Prefill 计算减少：一手资料与本项目判断

检索日期：2026-09-08。当前只完成代表性路线筛查；未完成论文实现复现，不把论文数值作为本项目结果。

| 路线 | 一手来源支持的机制 | 本项目判断（推论，不是论文结论） |
|---|---|---|
| 层间缩短序列 | Funnel-Transformer 逐步缩短 hidden-state 序列以减少计算，并用 decoder 恢复逐 token 表示。 | 直接命中“少量表示跑后续层”；原论文并不证明冻结的因果 Qwen 可直接均值池化后保持检索精度。 |
| 动态 token pruning | LazyLLM 在 prefill/decode 选择重要 token 的 KV 计算，并允许先前被裁掉的 token 在后续步骤重新参与。 | 可作真正减少 token 运算的对照；选择性保留与 chunk 信息合并必须区分，恢复成本也需计入。 |
| 稀疏 attention | FlexPrefill 根据输入和 head 动态调整稀疏 attention 模式与预算。 | 主要减少 attention 连接计算；不自动消除所有 token 的 MLP/QKV 成本，收益受上下文长度和算子占比影响。 |

来源：
- [Funnel-Transformer, arXiv:2006.03236](https://arxiv.org/abs/2006.03236)
- [LazyLLM, arXiv:2407.14057](https://arxiv.org/abs/2407.14057)
- [FlexPrefill, arXiv:2502.20766](https://arxiv.org/abs/2502.20766)
- [FlexPrefill 作者代码库](https://github.com/ByteDance-Seed/FlexPrefill)

上述来源当前提供机制依据，不提供我们模型、硬件、数据、query-blind 条件下的准确率或加速保证。首个原型将是独立的训练前诊断，不冒称这几篇论文的完整复现。
