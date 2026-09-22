# v5 局部恢复诊断结果

审计 PASS；完成 108/108 单元；3个失败案例 + 3个正确对照。

这是答案可见、事后选样的机制诊断，不是新验证集准确率。所有位置为 T 内从0开始的 chunk 编号。

## 1155f1c2b9ca39381f7f5844（loss）

Gold: {"warehouse": "Amber", "boxes": 46}
Full: {   "warehouse": "Amber",   "boxes": 46 }
压缩: {   "warehouse": "Dune",   "boxes": 46 }
仓库证据 chunks [38]；箱数证据 chunks [204]。

| 范围 | keep | 证据chunk | 正确 | placebo chunk | 正确 | 证据/对照正确-vs-foil logP差 |
|---|---:|---|---|---|---|---|
| chunk_38 | 32 | [38] | False | [37] | False | -8.3995 / -10.0249 |
| chunk_38 | 64 | [38] | True | [37] | False | -1.8055 / -10.1342 |
| chunk_204 | 32 | [204] | False | [203] | False | -9.9151 / -9.7433 |
| chunk_204 | 64 | [204] | False | [203] | False | -9.8060 / -9.3372 |
| warehouse_sentence | 32 | [38] | False | [37] | False | -8.3995 / -10.0249 |
| warehouse_sentence | 64 | [38] | True | [37] | False | -1.8055 / -10.1342 |
| all_evidence | 32 | [38, 204] | False | [36, 37] | False | -8.4306 / -9.9467 |
| all_evidence | 64 | [38, 204] | True | [36, 37] | False | -1.8679 / -9.5249 |

## df95590d8ecd480de63b0a95（matched_correct）

Gold: {"warehouse": "Amber", "boxes": 57}
Full: {   "warehouse": "Amber",   "boxes": 57 }
压缩: {   "warehouse": "Amber",   "boxes": 57 }
仓库证据 chunks [38]；箱数证据 chunks [204]。

| 范围 | keep | 证据chunk | 正确 | placebo chunk | 正确 | 证据/对照正确-vs-foil logP差 |
|---|---:|---|---|---|---|---|
| chunk_38 | 32 | [38] | True | [37] | True | 12.6651 / 6.7259 |
| chunk_38 | 64 | [38] | True | [37] | True | 17.6756 / 6.7735 |
| chunk_204 | 32 | [204] | True | [203] | True | 7.1016 / 6.7886 |
| chunk_204 | 64 | [204] | True | [203] | True | 7.8990 / 6.7411 |
| warehouse_sentence | 32 | [38] | True | [37] | True | 12.6651 / 6.7259 |
| warehouse_sentence | 64 | [38] | True | [37] | True | 17.6756 / 6.7735 |
| all_evidence | 32 | [38, 204] | True | [36, 37] | True | 12.6806 / 8.1006 |
| all_evidence | 64 | [38, 204] | True | [36, 37] | True | 17.6004 / 11.5217 |

## 18c5665358c364ff70111471（loss）

Gold: {"warehouse": "Amber", "boxes": 62}
Full: {   "warehouse": "Amber",   "boxes": 62 }
压缩: {   "warehouse": "Dune",   "boxes": 62 }
仓库证据 chunks [38]；箱数证据 chunks [204]。

| 范围 | keep | 证据chunk | 正确 | placebo chunk | 正确 | 证据/对照正确-vs-foil logP差 |
|---|---:|---|---|---|---|---|
| chunk_38 | 32 | [38] | False | [37] | False | -14.6915 / -17.4259 |
| chunk_38 | 64 | [38] | False | [37] | False | -10.0671 / -17.4727 |
| chunk_204 | 32 | [204] | False | [203] | False | -17.3790 / -17.0349 |
| chunk_204 | 64 | [204] | False | [203] | False | -17.4886 / -16.4571 |
| warehouse_sentence | 32 | [38] | False | [37] | False | -14.6915 / -17.4259 |
| warehouse_sentence | 64 | [38] | False | [37] | False | -10.0671 / -17.4727 |
| all_evidence | 32 | [38, 204] | False | [36, 37] | False | -14.7696 / -17.1292 |
| all_evidence | 64 | [38, 204] | False | [36, 37] | False | -10.2845 / -16.3477 |

## 73a05237143cc0773a4a9152（matched_correct）

Gold: {"warehouse": "Amber", "boxes": 66}
Full: {   "warehouse": "Amber",   "boxes": 66 }
压缩: {   "warehouse": "Amber",   "boxes": 66 }
仓库证据 chunks [38]；箱数证据 chunks [204]。

| 范围 | keep | 证据chunk | 正确 | placebo chunk | 正确 | 证据/对照正确-vs-foil logP差 |
|---|---:|---|---|---|---|---|
| chunk_38 | 32 | [38] | True | [37] | True | 5.1951 / 0.9449 |
| chunk_38 | 64 | [38] | True | [37] | True | 9.8836 / 0.9453 |
| chunk_204 | 32 | [204] | True | [203] | True | 1.1953 / 0.8515 |
| chunk_204 | 64 | [204] | True | [203] | True | 3.4448 / 0.8204 |
| warehouse_sentence | 32 | [38] | True | [37] | True | 5.1951 / 0.9449 |
| warehouse_sentence | 64 | [38] | True | [37] | True | 9.8836 / 0.9453 |
| all_evidence | 32 | [38, 204] | True | [36, 37] | True | 5.3513 / 2.1638 |
| all_evidence | 64 | [38, 204] | True | [36, 37] | True | 10.3208 / 3.9763 |

## 3615d03937e9ef228782b28f（loss）

Gold: {"warehouse": "Amber", "boxes": 57}
Full: {   "warehouse": "Amber",   "boxes": 57 }
压缩: {   "warehouse": "Dune",   "boxes": 57 }
仓库证据 chunks [37]；箱数证据 chunks [204]。

| 范围 | keep | 证据chunk | 正确 | placebo chunk | 正确 | 证据/对照正确-vs-foil logP差 |
|---|---:|---|---|---|---|---|
| chunk_37 | 32 | [37] | True | [36] | False | 4.9954 / -0.0670 |
| chunk_37 | 64 | [37] | True | [36] | False | 9.1477 / -0.0667 |
| chunk_204 | 32 | [204] | True | [203] | False | 0.6785 / -0.0355 |
| chunk_204 | 64 | [204] | True | [203] | False | 2.5607 / -0.0982 |
| warehouse_sentence | 32 | [37] | True | [36] | False | 4.9954 / -0.0670 |
| warehouse_sentence | 64 | [37] | True | [36] | False | 9.1477 / -0.0667 |
| all_evidence | 32 | [37, 204] | True | [35, 36] | True | 5.2716 / 0.7144 |
| all_evidence | 64 | [37, 204] | True | [35, 36] | True | 9.2721 / 2.2771 |

## db38fac84d1584897de1e7b7（matched_correct）

Gold: {"warehouse": "Dune", "boxes": 66}
Full: {   "warehouse": "Dune",   "boxes": 66 }
压缩: {   "warehouse": "Dune",   "boxes": 66 }
仓库证据 chunks [38]；箱数证据 chunks [204]。

| 范围 | keep | 证据chunk | 正确 | placebo chunk | 正确 | 证据/对照正确-vs-foil logP差 |
|---|---:|---|---|---|---|---|
| chunk_38 | 32 | [38] | True | [37] | True | 33.4738 / 33.4426 |
| chunk_38 | 64 | [38] | True | [37] | True | 33.7548 / 33.5205 |
| chunk_204 | 32 | [204] | True | [203] | True | 33.3800 / 33.4738 |
| chunk_204 | 64 | [204] | True | [203] | True | 33.3017 / 33.5048 |
| warehouse_sentence | 32 | [38] | True | [37] | True | 33.4738 / 33.4426 |
| warehouse_sentence | 64 | [38] | True | [37] | True | 33.7548 / 33.5205 |
| all_evidence | 32 | [38, 204] | True | [36, 37] | True | 33.4112 / 33.3173 |
| all_evidence | 64 | [38, 204] | True | [36, 37] | True | 33.5830 / 33.2080 |

## 解释边界
- 同一案例的多个干预、重复chunk集合不是独立样本。
- keep32/64 同时改变表示、数量与位置锚点，不能单独据此归因 RoPE 或均值混合。
- placebo 使用同数量非证据chunk；恢复若非证据chunk也有效，应考虑全局注意力/数值与上下文干扰。
- 证据句包含标识符和属性值；没有只保护答案 token。路由仍使用已知证据，是 oracle 诊断。
- 未恢复不证明信息彻底丢失；恢复不证明通用无损。
- 全部耗时带诊断开销，不作为 prefill 加速结果。
