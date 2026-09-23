# v6 反向局部压缩诊断

审计 PASS；完成 36/36 单元。六份文档来自已暴露v5，不能作独立泛化准确率。

与先前“压缩所有chunk、只保护指定chunk”相反，本次“保留所有chunk、只压缩指定chunk”。每个证据干预有相同压缩预算的非证据对照。

| 样本 | 角色 | Full | 全压缩 | 只压仓库证据 | 只压非证据对照 | 只压两处证据 | 只压两个非证据对照 |
|---|---|---|---|---|---|---|---|
| 1155f1c2b9ca39381f7f5844 | loss | True | False | True | True | True | True |
| df95590d8ecd480de63b0a95 | matched_correct | True | True | True | True | True | True |
| 18c5665358c364ff70111471 | loss | True | False | True | True | True | True |
| 73a05237143cc0773a4a9152 | matched_correct | True | True | True | True | True | True |
| 3615d03937e9ef228782b28f | loss | True | False | True | True | True | True |
| db38fac84d1584897de1e7b7 | matched_correct | True | True | True | True | True | True |

## 与先前保护实验并读

| 失败样本 | 全压缩 | 全压缩但保护仓库证据 | 仅压仓库证据 | 全压缩但保护两处证据 | 仅压两处证据 |
|---|---|---|---|---|---|
| 1155f1c2b9ca39381f7f5844 | False | True | True | True | True |
| 18c5665358c364ff70111471 | False | False | True | False | True |
| 3615d03937e9ef228782b28f | False | True | True | True | True |

## 限制

- 本轮干预使用答案定位；每个干预与对照预算相等，但不同基线（全保留/全压缩）不能直接比较绝对压缩率。
- 只压缩某个chunk后模型出错，表明在该上下文中该局部变化足以改变输出；不能单独归因于信息内容、RoPE位置、注意力权重归一化或数值误差。
- 只压证据和只压非证据若都正确，不能推断原全局压缩的错误来自某一单个chunk；可能是交互。
- 多个条件共享相同文档，不是独立样本。logP对照使用紧凑JSON teacher forcing，不是校准置信度；有诊断开销的时间不是加速数字。
