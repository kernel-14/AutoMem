# WebWalkerQA run #5 (`webwalkerqa-full-c3`) 最终验证数据

复现记录见 [`docs/webwalkerqa_repro.md`](../../docs/webwalkerqa_repro.md)。

## 来源

- 引擎：AutoMem（webwalkerqa 分支），backbone hy3（agent/proposer 走 taiji，judge 走 venus，两臂同一 judge）
- 数据：WebWalkerQA 680 任务中 `final_test_indices` 的 **170 个 held-out 任务**（与 baseline 同批，成对比较）
- run 目录：`runs/search/webwalkerqa-full-c3/final_validation/`（2026-09-05 14:34 完赛）
- 本目录为该目录的**无改动子集**，仅排除 `run.log`（26M，含内网 API 端点 URL，无复现价值）

## 内容

| 路径 | 说明 |
|---|---|
| `baseline_tasks/*.json` | 无记忆臂，170 任务逐个结果（`success`/`task_score`/`judgement`/`difficulty`/`language` 等） |
| `baseline_tasks/report.txt` | baseline 臂汇总（acc 41.76%，71/170） |
| `tasks/*.json` | 记忆臂（冠军架构 r6_c1），170 任务逐个结果 |
| `results.jsonl` | 记忆臂 170 行 JSONL 汇总（每行一任务，含完整 trajectory） |
| `validation_result.json` | 引擎最终验证输出（best_config_id、架构、final_test_indices、acc/lift） |
| `runtime_config.json` | 评测协议配置（已确认不含任何密钥） |
| `storage/store_json/memory_db.json` | 冠军架构最终记忆池（312 units） |

## 核心数字

| 臂 | acc |
|---|---|
| baseline（无记忆） | 41.76%（71/170） |
| memory（r6_c1） | 45.88%（78/170） |
| raw lift | **+4.1pp**（McNemar 校正 p=0.296，不显著） |

## 复现统计

```python
import json, glob
def acc(d, pat="*.json"):
    files = glob.glob(f"{d}/{pat}")
    return sum(str(json.load(open(f))["success"]).lower() == "true" for f in files) / len(files)
print(acc("baseline_tasks"), acc("tasks"))   # 0.4176 0.4588
```

成对比较以 json 内 `item_index` 对齐（两臂同批任务）。

## 数据保真说明

任务 json 中出现的 `10.176.63.160`（复旦官网"会议室预订（内网）"链接）是
WebWalkerQA 数据集自身的页面内容，非本复现的基础设施信息，故保留原样。
