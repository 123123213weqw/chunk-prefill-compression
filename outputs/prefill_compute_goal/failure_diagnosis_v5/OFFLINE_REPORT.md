# v5 失败案例：GPU 干预前的离线诊断

状态：证据定位已完成，GPU 干预尚未运行；不能声称已恢复答案。

全部三个失败是 narrative/16K，仓库 Amber→Dune，箱子数保持正确。

| 样本 | 角色 | 正确仓库/箱数 | 原压缩输出 | 仓库证据 chunk | 箱数证据 chunk |
|---|---|---|---|---|---|
| 1155f1c2b9ca39381f7f5844 | loss | {'warehouse': 'Amber', 'boxes': 46} | {   "warehouse": "Dune",   "boxes": 46 }  | [38] | [204] |
| df95590d8ecd480de63b0a95 | matched_correct | {'warehouse': 'Amber', 'boxes': 57} | {   "warehouse": "Amber",   "boxes": 57 }  | [38] | [204] |
| 18c5665358c364ff70111471 | loss | {'warehouse': 'Amber', 'boxes': 62} | {   "warehouse": "Dune",   "boxes": 62 }  | [38] | [204] |
| 73a05237143cc0773a4a9152 | matched_correct | {'warehouse': 'Amber', 'boxes': 66} | {   "warehouse": "Amber",   "boxes": 66 }  | [38] | [204] |
| 3615d03937e9ef228782b28f | loss | {'warehouse': 'Amber', 'boxes': 57} | {   "warehouse": "Dune",   "boxes": 57 }  | [37] | [204] |
| db38fac84d1584897de1e7b7 | matched_correct | {'warehouse': 'Dune', 'boxes': 66} | {   "warehouse": "Dune",   "boxes": 66 }  | [38] | [204] |

索引从0开始，chunk长64 token。prepared.json 记录每个证据 token、合并组、组内偏移和原逻辑位置锚点。

固定队列：108 单元。每份文档重新验证 Full identity gate 和压缩基线；证据 chunk 单独及联合改为 keep32/64，并设置同数量、相同预算的非证据 placebo chunk。

这是使用题目和答案定位证据的事后机制诊断，不是 query-blind 路由算法，也不能用于宣称泛化准确率。正确对照只存在2份同为 Amber 的样本，第3份匹配任务/长度但仓库为 Dune；详见 selection.json。

一次恢复只能证明该配置的干预影响输出，不能单独分离内容混合、位置锚定、全局注意力归一化等机制。未恢复也不等于证据已经不可逆丢失。
