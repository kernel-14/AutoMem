# AppWorld 复现记录（appworld-full）

日期：2026-08-24 ~ 2026-08-25。复现仓库：kernel-14/AutoMem fork（appworld 分支）。

## 配置

| 项 | 值 |
| --- | --- |
| 任务模型（四角色同） | 混元 hy3（OpenAI 兼容内部代理） |
| 数据 | AppWorld train+dev 共 147 任务（`data/appworld/appworld_train_dev.jsonl`） |
| 切分 | warmup 20 / search 50 / validation 37 / test 40（seed 42，fold 轮换 2×25） |
| 搜索 | 5 轮 × 3 候选，配对显著性检验 α=0.1，final runoff（前 2 名 × 完整 50 任务） |
| 判分 | AppWorld 确定性状态/答案断言（无 judge LLM） |
| 总量 | 682 个任务结果，124M rollout tokens，约 17h 墙钟（含两次中断恢复） |

## 结果

| 指标 | 数值 |
| --- | --- |
| 无记忆基线（全 107 任务） | 23.4%（25 easy / 82 memory-sensitive） |
| held-out test（40 任务）无记忆 | 20.0%（8/40） |
| held-out test 带记忆（胜者架构） | 22.5%（9/40） |
| **memory lift（held-out）** | **+2.5 点** |
| 检索 hit rate（R1 冠军 → R2 冠军） | 0.44 → 0.88 |

**胜出架构（runoff 决赛 16% vs 14% 胜出）**：

```json
{
  "extract_types": ["tip", "trajectory", "workflow", "insight"],
  "storage_routing": {"tip": "json", "trajectory": "json", "workflow": "json", "insight": "json"},
  "retrieval": "hybrid",
  "management": "lightweight"
}
```

## 逐轮 best-so-far（fold 交替）

| 轮 | fold | acc | lift | 架构要点 |
| --- | --- | --- | --- | --- |
| R1 | 1 | 12.0% | +4.0 | [tip] 单类型 |
| R2 | 2 | 28.0% | +0.0 | 四类编码 ← 最终冠军 |
| R3 | 1 | 16.0% | +1.3 | 四类（fold1: 12→16） |
| R4 | 2 | 24.0% | −4.0 | 五类+shortcut（fold2 回落） |
| R5 | 1 | 20.0% | +2.0 | 五类（fold1: 16→20） |
| runoff | 全 50 | 16.0% | — | r2_c2 胜 r5_c1 |

## 经验账本（最终）

- **Principle P001**：扩展 extract_types 超出 [tip] 以覆盖任务族、降低空检索率（证据：R1 empty_retrieval_rate=0.56，extraction_gap=7）
- **Dead end**：[tip,trajectory,shortcut] + tool_manager（R1_c2 评测失败）
- 10 条 open questions 累积

## 观察

1. **机制完整复现**：propose→评测→FGMD 归因→账本更新→champion 注入→配对检验→runoff→held-out 验证全链条按论文设计运转；配对显著性检验拒绝了 R5 对 R2 的噪声性"改进"。
2. **支持"架构随任务分布定制"**：AppWorld 上搜出"广覆盖编码 + json/hybrid + 轻管理"，与论文 GAIA（图存储+对比检索）、WebWalkerQA（tool_manager）均不同——个人助理分布偏好"记全"而非"记精"。
3. **lift 偏弱但检索端起效**：+2.5 点在 40 任务上只有 1 个任务的差距（统计弱）；hit rate 翻倍说明瓶颈在注入后的利用（FGMD 分类中的 injection_bad），而非检索。
4. **工程**：三次中断（引擎父进程 .env 未加载、faiss-cpu 缺失、运行中改代码触发协议指纹清空）均靠断点续跑恢复；运行期间不可修改仓库代码。

## 复现命令

```bash
python -m automem.search.engine \
  --run_name appworld-full --output_dir runs/search --benchmark appworld \
  --infile data/appworld/appworld_train_dev.jsonl \
  --model hy3 --search_model hy3 --judge_model hy3 --diagnosis_model hy3 \
  --max_rounds 5 --num_candidates 3 \
  --warmup_n 20 --search_n 50 --validation_n 37 --test_n 40 \
  --final_validation --resume
```
