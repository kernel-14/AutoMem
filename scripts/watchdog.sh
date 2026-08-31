#!/bin/bash
# Auto-resume the WebWalkerQA AutoMem search engine if the sandbox killed it.
# Installed in system crontab (survives CodeBuddy session restarts).
# Set /tmp/NO_AUTORESUME to temporarily disable (e.g. during offline re-judge).
set -u
cd /data/workspace/AutoMem-repro
LOG=/tmp/automem_watchdog.log
ts() { date '+%F %T'; }

if [ -e /tmp/NO_AUTORESUME ]; then
  exit 0
fi

if pgrep -f "automem.search.engine --run_name webwalkerqa-full" >/dev/null 2>&1; then
  exit 0
fi

# Engine not running. If the run already finished, do nothing.
if grep -qE "Final validation complete|Search complete|All rounds complete" runs/webwalkerqa-full.console.log 2>/dev/null; then
  echo "$(ts) run appears finished; not resuming" >> "$LOG"
  exit 0
fi

echo "$(ts) engine dead -> resuming" >> "$LOG"
nohup .venv/bin/python -u -m automem.search.engine \
  --run_name webwalkerqa-full --output_dir runs/search \
  --benchmark WebWalkerQA --infile data/webwalkerqa/webwalkerqa_main.jsonl \
  --model hy3 --search_model hy3 --judge_model hy3 --diagnosis_model hy3 \
  --max_rounds 6 --num_candidates 3 --warmup_n 0 --search_n 60 \
  --validation_n 110 --test_n 170 --concurrency 8 --final_validation --resume \
  >> runs/webwalkerqa-full.console.log 2>&1 &
echo "$(ts) resumed pid $!" >> "$LOG"
