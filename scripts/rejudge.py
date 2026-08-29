#!/usr/bin/env python
"""Offline re-judge for unjudged WebWalkerQA task checkpoints.

Uses the runner's own judge_webwalkerqa_answer (same model, same prompt,
same exact-match fallback) to fill in verdicts for tasks whose judge call
was starved by rate limiting. Only touches task jsons on disk; no repo
source changes (digest-safe: script lives in /tmp).

Run with the repo venv python from the repo root. Env must be loaded.
"""

import glob
import json
import os
import sys
import time

sys.path.insert(0, "/data/workspace/AutoMem-repro/src")
from dotenv import load_dotenv

load_dotenv("/data/workspace/AutoMem-repro/.env", override=False)

from automem.benchmarks.webwalkerqa.runner import (  # noqa: E402
    _normalize_answer_for_judge_fallback,
    judge_webwalkerqa_answer,
)

TASKS_GLOB = sys.argv[1] if len(sys.argv) > 1 else (
    "/data/workspace/AutoMem-repro/runs/search/webwalkerqa-full/baseline/tasks/*.json"
)
JUDGE_MODEL = os.environ.get("DEFAULT_JUDGE_MODEL", "hy3")


def main() -> None:
    files = sorted(glob.glob(TASKS_GLOB))
    fixed, still_unjudged, already_ok = 0, 0, 0
    for path in files:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("status") != "success" or data.get("judge_unjudged") is not True:
            already_ok += 1
            continue
        question = data.get("question") or ""
        golden = data.get("golden_answer") or ""
        pred = data.get("agent_result")
        verdict = None
        for attempt in range(4):
            try:
                res = judge_webwalkerqa_answer(
                    question, golden, pred, model=JUDGE_MODEL
                )
                verdict = (res.get("judgement") or "").strip().lower()
                if verdict in ("correct", "incorrect"):
                    break
            except Exception as exc:  # noqa: BLE001
                print(f"  judge exception: {exc}")
            time.sleep(20)  # breathe within QPM before retry
        if verdict not in ("correct", "incorrect"):
            # replicate the runner's exact-match rescue for judge infra errors
            pred_norm = _normalize_answer_for_judge_fallback(pred)
            gold_norm = _normalize_answer_for_judge_fallback(golden)
            if pred_norm and pred_norm == gold_norm:
                verdict, data["judge_fallback"] = "correct", "exact_match"
            else:
                still_unjudged += 1
                print(f"still unjudged: {os.path.basename(path)}")
                continue
        data["judgement"] = verdict
        data["judge_unjudged"] = False
        data["task_score"] = 1.0 if verdict == "correct" else 0.0
        data["success"] = verdict == "correct"
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False)
        os.replace(tmp, path)
        fixed += 1
        print(f"re-judged {os.path.basename(path)}: {verdict}")
        time.sleep(7)  # ~8 QPM
    print(f"\nsummary: fixed={fixed}, still_unjudged={still_unjudged}, already_ok={already_ok}")


if __name__ == "__main__":
    main()
