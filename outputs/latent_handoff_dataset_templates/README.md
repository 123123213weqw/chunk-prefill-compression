# Latent Handoff Dataset Templates v1

本目录固定第一阶段自建数据集的两个模板：

1. `01_exact_fact_template.json`：测试未被后缀文本复述的精确事实是否仍存在于旧后缀 KV 中。
2. `02_derived_workflow_template.json`：测试派生结论、因果依据和 Agent 工作流状态的保留程度。

## 固定实验定义

每条样本都由四段组成：

\[
[S][T][R][Q]
\]

- `S`：始终保留的 system/user 上下文。
- `T`：准备删除的长工具输出。
- `R`：在读取 `T` 后形成、且在所有条件下 token IDs 完全相同的后缀载体。
- `Q`：删除之后输入的探针问题。

三个核心缓存条件为：

\[
C_{full}=KV(S,T,R)
\]

\[
C_{evict}=\operatorname{Select}_{S\cup R}(KV(S,T,R))
\]

\[
C_{replay}=KV(S,R;\operatorname{pos}(R)=\operatorname{pos}_{original}(R))
\]

删除以后：

\[
L_{physical}=|S|+|R|
\]

但下一个 token 的位置必须为：

\[
p_{next}=|S|+|T|+|R|
\]

## 核心指标

淘汰损失：

\[
L_{evict}=D_{KL}(p_{full}\Vert p_{evict})
\]

Replay 损失：

\[
L_{replay}=D_{KL}(p_{full}\Vert p_{replay})
\]

隐式继承收益：

\[
G_{inherit}=L_{replay}-L_{evict}
\]

若 `G_inherit > 0`，说明 history-conditioned 的旧 `R-KV` 比重新计算的相同 `R` 更接近完整缓存。

## 第一版数据规模

每个模板使用：

- 4 个工具输出长度：512、2K、8K、16K tokens；
- 3 个后缀载体长度：32、128、512 tokens；
- 每个组合 10 个随机变体。

因此每个模板生成 120 条样本，两个模板共 240 条。第一轮只运行 512 和 2K token 子集，验证实现后再扩展到 8K 和 16K。

## 不可改变的控制条件

- Full、Evict、Replay 使用完全相同的 `S/R/Q` token IDs。
- Replay 中 `R` 保持原始 logical positions。
- Evict 只删除 `T` 对应的 K/V 行，不重新计算 `S` 或 `R`。
- 第一轮使用 greedy decoding，不引入采样噪声。
- 精确事实模板中的目标字段禁止出现在 `R`。
- 派生状态模板必须同时包含显式工作流控制题和仅依赖 `T` 的隐式信息题。
