# v7 chunk数量扫描

审计PASS；228/228原子单元，6份已暴露文档×4条固定压缩顺序；0与256端点共用，不重复计为独立GPU试验。

count是256个T chunk中压缩到16个token的数量。每层投影/MLP相关T token节省 = count×48×6；不是总计算或实际加速。

| 样本 | 原角色 | 顺序 | 0/16/32/64/96/128/160/192/224/240/256 的正确性（✓/×） | 首错grid点 | 翻转次数 | 错后恢复 |
|---|---|---|---|---:|---:|---|
| 1155f1c2b9ca39381f7f5844 | loss | prefix | ✓✓✓✓✓✓✓✓××× | 224 | 1 | False |
| 1155f1c2b9ca39381f7f5844 | loss | suffix | ✓✓✓✓✓✓✓✓✓✓× | 256 | 1 | False |
| 1155f1c2b9ca39381f7f5844 | loss | random_A | ✓✓✓✓✓✓✓✓××× | 224 | 1 | False |
| 1155f1c2b9ca39381f7f5844 | loss | random_B | ✓✓✓✓✓✓✓✓✓✓× | 256 | 1 | False |
| df95590d8ecd480de63b0a95 | matched_correct | prefix | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| df95590d8ecd480de63b0a95 | matched_correct | suffix | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| df95590d8ecd480de63b0a95 | matched_correct | random_A | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| df95590d8ecd480de63b0a95 | matched_correct | random_B | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| 18c5665358c364ff70111471 | loss | prefix | ✓✓✓×××××××× | 64 | 1 | False |
| 18c5665358c364ff70111471 | loss | suffix | ✓✓✓✓✓✓✓✓✓✓× | 256 | 1 | False |
| 18c5665358c364ff70111471 | loss | random_A | ✓✓✓✓✓✓✓×××× | 192 | 1 | False |
| 18c5665358c364ff70111471 | loss | random_B | ✓✓✓✓✓✓✓✓××× | 224 | 1 | False |
| 73a05237143cc0773a4a9152 | matched_correct | prefix | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| 73a05237143cc0773a4a9152 | matched_correct | suffix | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| 73a05237143cc0773a4a9152 | matched_correct | random_A | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| 73a05237143cc0773a4a9152 | matched_correct | random_B | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| 3615d03937e9ef228782b28f | loss | prefix | ✓✓✓✓✓✓✓✓✓✓× | 256 | 1 | False |
| 3615d03937e9ef228782b28f | loss | suffix | ✓✓✓✓✓✓✓✓✓✓× | 256 | 1 | False |
| 3615d03937e9ef228782b28f | loss | random_A | ✓✓✓✓✓✓✓✓✓✓× | 256 | 1 | False |
| 3615d03937e9ef228782b28f | loss | random_B | ✓✓✓✓✓✓✓✓✓✓× | 256 | 1 | False |
| db38fac84d1584897de1e7b7 | matched_correct | prefix | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| db38fac84d1584897de1e7b7 | matched_correct | suffix | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| db38fac84d1584897de1e7b7 | matched_correct | random_A | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |
| db38fac84d1584897de1e7b7 | matched_correct | random_B | ✓✓✓✓✓✓✓✓✓✓✓ | None | 0 | False |

## 解读边界

- 图格代表预先选定的离散压缩数量，不是连续真实阈值；同一顺序下新增chunk是嵌套的。
- 如果出现错后恢复，单调阈值假设不成立；所有翻转逐项记录在summary.json。
- Prefix/suffix/random是chunk数量与位置的机制对照，并非模型可用的选择器；每份文档的4条轨迹不是4份独立样本。
- 只对3个原失败案例和3个正确对照做事后扫描，不可用于独立准确率或泛化宣称。
- 计时含hooks及评分，不能代替干净prefill测速。
