# WebWalkerQA 复现记录（webwalkerqa 分支）

目标：在论文表现较强的 WebWalkerQA 上用混元 hy3 完整复现 AutoMem 的进化搜索（论文数字：无记忆 67.6% → AutoMem 72.5%，lift +4.9，Qwen3.5-122B-A10B backbone）。

## 配置（两个阶段共用）

| 项 | 值 |
| --- | --- |
| 数据 | WebWalkerQA 680 任务（HF `callanwu/WebWalkerQA`，免授权），`data/webwalkerqa/webwalkerqa_main.jsonl` |
| 切分 | warmup 0 / search 60 / validation 110 / test 170（对齐论文 Table 6：无独立 warmup、首轮播种记忆池） |
| 搜索 | 6 轮 × 3 候选，配对显著性检验 α=0.1，final runoff |
| 工具栈 | duckduckgo 搜索 + crawl4ai 爬取（零外部凭证；抽样爬取成功率 5/8，失败集中于国内 .edu.cn 站点） |
| 判分 | hy3 judge（WebWalkerQA runner 原生 LLM 判分） |

## 阶段一：venus 代理（2026-08-25，被限额中断，结果作为参照系）

任务模型/proposer/诊断/判分全部 hy3 @ `v2.open.venus.oa.com`，并发 6。

**完成的部分**（数据因端点切换触发协议指纹清空而丢失，以下数字来自运行日志）：

- 无记忆 baseline：**43.5%**（74 easy / 96 memory-sensitive，170 任务）
- 6 轮搜索全部完成，**六轮冠军全部收敛于同一架构家族**：
  `extract=[tip, trajectory] / storage=json / retrieval=hybrid / management=lightweight`
- 逐轮 hit rate：0.767 → 0.922 → 0.953 → 0.956 → 0.967 → 0.973（canonical 池 0 → 176 单元）
- 逐轮冠军 acc（fold 交替）：40.0% → 43.3% → 39.3% → 38.9% → 38.3% → 38.7%
- Final runoff：**r5_c2 胜出**（同家族但 retrieval=**mmr**，36.7% vs r1_c0 的 33.3%）——
  validation 裁决否决了 search 批领跑的 hybrid 家族，M3 机制生效
- 中断点：final validation 的 held-out 基线跑到 160/170 时 **venus 日限额 200 元烧穿**

**venus 阶段的观察**：
1. 重复导航模式 → tip/trajectory 记忆高度可迁移（R1 即 lift +10 点）
2. hit rate 爬到 0.97+ 但 acc 停在 ~40%：检索端饱和、转化端受限（与 AppWorld 结论一致）
3. 搜索很快收敛（R1 提出的家族六轮未被推翻，后期改进主要靠池变厚）

## 事故复盘（两条，均已转化为操作纪律）

1. **运行中改代码 → 协议指纹清空**（AppWorld 阶段，损失 4h）：指纹包含 prompt 树与包源码
2. **换 API 端点 → 协议指纹清空**（本阶段，损失 ~20h）：指纹的 `endpoints` 组件包含全部
   api_base；`behavior_args` 组件包含 concurrency/max_rounds/切分参数/max_steps 等全部
   行为参数。**任何一项改动 = 整个 run 作废重来**

纪律：①改 .env/代码前先 `cp -al` 备份 run 目录；②重启命令必须与首次启动一字不差；
③运行期间不碰任何实验条件。

## 阶段二：taiji OpenAPI 全量重跑（2026-08-27 15:40 以 no_think 模式重启，进行中）

- 端点：`http://api.taiji.woa.com/openapi/v2`（OpenAI 兼容，Bearer key 见 repo `.env`）
- 限流：~10 QPM（服务端硬上限，错误信息确认，可申请提额；429 由重试吸收）
- 并发 8；**reasoning_effort=no_think**（commit 56b6355，env `LLM_REASONING_EFFORT` 门控）——
  high 思考模式单次调用 30-90s 是延迟瓶颈（~8 调用/分钟），no_think 后转为 QPM 硬顶
  （~10 调用/分钟），全量 ETA ~3-5 天。曾中途诊断：阶段二第一版（high thinking）跑到
  baseline 116/170 后被放弃，备份在 runs/search/webwalkerqa-full.bak-highthink
- 阶段二期间的 high-thinking 版 baseline 部分结果（116/170 有效、通过率 52.6%）
  备份在 runs/search/webwalkerqa-full.bak-highthink，待 no_think 版 baseline 完成后
  可量化 thinking 模式对无记忆基线的影响
- 页面磁盘缓存（`.cache/`）在指纹清空中幸存，爬取近乎免费
- 页面缓存复用使阶段二与阶段一的爬取条件一致

## 已知工程注记

- 引擎的 `--concurrency>1` 硬性守卫已在本分支放宽为警告（见 commit f0783e3 的理由说明）
- 代理限流/限额错误在模型包装层表现为 `reasoning_content NoneType`，不优雅；
  runner 的 fail-closed 完成门禁 + 断点续跑可兜底（error 任务不计入完成，resume 精准补跑）
- 协议指纹组件清单（engine.py `_compute_eval_protocol_signature`）：prompt 树、
  runtime policy、query planner/composer prompts、L1/L2 指令、四模型 ID、execution_mode、
  endpoints（全部 api_base + 各角色是否配置）、environment_controls、behavior_args、split

## 结果（待补）

- [ ] taiji 全量 baseline
- [ ] 六轮搜索轨迹
- [ ] runoff 胜者与最终架构
- [ ] final validation（170 held-out）：无记忆 vs 带记忆 → memory lift
- [ ] 与 venus 阶段参照系的对比（同名模型、不同端点的行为差异）
