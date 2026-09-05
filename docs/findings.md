# 复现发现（跨 benchmark）

来源：AppWorld（appworld 分支）与 WebWalkerQA（webwalkerqa 分支）两次独立复现，
backbone 均为混元 hy3（冻结、纯 API 推理）。数字层面的完整记录见
[`appworld_repro.md`](appworld_repro.md) 与 [`webwalkerqa_repro.md`](webwalkerqa_repro.md)；
本文只沉淀跨 benchmark 稳定成立的结论。

## 1. 记忆只对"能力圈边缘"的任务有效

WebWalkerQA 最终验证（170 held-out 成对）按难度分层：

| difficulty | n | baseline | memory | lift |
|---|---|---|---|---|
| medium | 114 | 39.5% | 46.5% | **+7.0pp** |
| easy | 21 | 66.7% | 66.7% | 0.0pp（天花板已满） |
| hard | 35 | 34.3% | 31.4% | −2.9pp（够不着） |

记忆是把 agent 从"不会"推到"会"的杠杆，只在能力边界处起作用。整体 lift
（+4.1pp）偏小的直接原因：大多数 held-out 任务不在边界上。评估记忆系统时，
**分层报告比单一 acc 更有信息量**。

## 2. 检索端饱和、转化端受限（两个 benchmark 一致）

- WebWalkerQA：venus 阶段 hit rate 爬到 0.97、run #5 冠军 hit=0.933，acc 仍停在 ~40%；
  最终验证 85.9% 任务成功注入记忆（平均 1.65 条），token 开销仅 +2.7%，lift 仍不显著
- AppWorld：同样观察到 hit rate 高位平台与 acc 脱钩

即：**检索"找到了"不等于"用对了"**。瓶颈在 base 模型把注入记忆转化为行动的能力，
不在记忆系统的检索召回。改进方向应指向注入内容的精度与时机（judge 过滤、
contrastive 匹配），而不是继续提高召回。

## 3. 对进化搜索本身的保留：架构优势可能被噪声高估

- 搜索轮 30 任务折 1 SD ≈ 9pp；R1→R3 的 +10pp 进步 ≈ 1 个标准差
- 同架构同 fold 重测（r1_c1 作为冠军重注入即 r3_c0）：33.3% → 26.7%，跌 6.6pp
- Final runoff 两个候选 acc **精确打平**（0.350 vs 0.350），靠 fitness 0.343 vs 0.337 决胜
- 最终 170 任务验证 1 SD ≈ 3.8pp，+4.1pp 的 lift 恰在噪声边界（McNemar p=0.296）

复现者视角的判断：**"记忆带来正向增益"成立；"搜索找到的最优架构显著优于其他架构"
证据不足**。论文若在同等噪声量级上报告架构差异，需检查其统计功效。

## 4. 判分端差异 > 记忆效应

同一批 170 任务、同一模型，仅判分端不同：

| 判分端 | baseline acc |
|---|---|
| taiji（taiji judge） | 50.0% |
| venus judge | 39.4% |

差 ~10pp，是记忆 lift（4pp）的 2.5 倍。**任何跨实验/跨论文的 acc 比较必须锁定
判分端**；成对实验（lift 差值）内部自洽即可信。LLM-as-judge 的尺度漂移是这一类
复现的第一误差源。

## 5. 绝对分与 lift 大致解耦（可迁移性证据）

| | base | baseline acc | memory lift |
|---|---|---|---|
| 论文 | Qwen3.5-122B-A10B | 67.6% | +4.9pp |
| 本复现 | hy3 | 41.8% | +4.1pp |

绝对分差 26pp（纯 base 能力差距），lift 却接近。记忆系统的相对贡献不随 base 强弱
崩塌——这是 AutoMem 类"train-free 推理时记忆"方案可迁移性的正面证据。

## 6. 工程教训（详 见 webwalkerqa_repro.md 事故复盘）

1. **协议指纹机制**：prompt 树/端点/行为参数/源码哈希任一改动 = 全 run 作废。
   改动前必须 `scripts/verify_digest.py` 预检；运行期间零改动纪律
2. **双缺陷叠加的静默死亡**：models.py 重试耗尽后隐式 `return None`（缺陷 A）+
   异常处理器内引用不存在的 `LogLevel.WARNING`（缺陷 B）→ 错误信息退化为
   `{'error': 'WARNING'}`，已加固的重试完全失效。异常处理器自身必须保证不抛
3. **QPM 饥饿**：agent 吃满限流时 judge 饿死 → fail-closed 门禁整体作废候选。
   judge 分流到独立无限流端点解决
4. **数据只在本地一份 = 最高风险**：runs/ 与 data/ 均被 .gitignore 排除，
   1.2G 实验数据在沙箱多次无预告杀进程的环境中裸奔。结论性子集应入库
   （见 `results/`），全量本地冷备

## 一句话总结

**记忆系统的增益是真实但窄带的（边界任务 +7pp），瓶颈在 base 模型的记忆利用率
而非检索；架构搜索的"最优性"大概率是噪声；判分端与统计功效是这个方向所有实验
的第一误差源。**
