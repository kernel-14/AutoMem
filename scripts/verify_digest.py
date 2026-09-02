#!/usr/bin/env python
"""Offline preflight for the eval-protocol digest.

Why this exists: the engine refuses to resume a search when the protocol
digest changes, and the digest hashes EVERY .py file under automem/ and
flashoagents/ plus a list of behaviour args (including --concurrency). So a
mid-run source edit silently costs a full replay. Run this BEFORE editing
source or changing launch flags to see exactly what breaks resume.

There are two digests:
  * digest          -> gates overall resume (includes --concurrency)
  * baseline_digest -> gates --baseline_from reuse; deliberately EXCLUDES
                       concurrency, so a concurrency-only change can still
                       reuse a completed baseline.

Usage:
    python scripts/verify_digest.py                 # vs saved protocol file
    python scripts/verify_digest.py --concurrency 3 # what-if launch flags
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Replicate engine module init: load .env BEFORE importing engine code paths
from dotenv import load_dotenv

load_dotenv(".env", override=False)

from automem.search import engine  # noqa: E402  (imports run load_dotenv again, harmless)

RUN_DIR = "runs/search/webwalkerqa-full"

BASE_ARGV = [
    "--run_name", "webwalkerqa-full",
    "--output_dir", "runs/search",
    "--benchmark", "WebWalkerQA",
    "--infile", "data/webwalkerqa/webwalkerqa_main.jsonl",
    "--model", "hy3", "--search_model", "hy3", "--judge_model", "hy3", "--diagnosis_model", "hy3",
    "--max_rounds", "6", "--num_candidates", "3",
    "--warmup_n", "0", "--search_n", "60", "--validation_n", "110", "--test_n", "170",
    "--concurrency", "8",
    "--final_validation",
]


def compute_sig(overrides: dict) -> dict:
    argv = list(BASE_ARGV)
    for key, value in overrides.items():
        flag = "--" + key
        if flag in argv:
            argv[argv.index(flag) + 1] = str(value)
        else:
            argv += [flag, str(value)]
    sys.argv = ["python"] + argv
    args = engine.parse_args()
    engine._apply_benchmark_split_defaults(args)
    engine._validate_search_args(args)
    from automem.search.protocol import ProtocolConfig

    proto = ProtocolConfig.resolve(args)
    args.val_every = proto.val_every
    args._protocol = proto
    return engine._compute_eval_protocol_signature(
        eval_model=(args.model or "").strip(),
        protocol=args._protocol,
        args=args,
    )


if __name__ == "__main__":
    overrides: dict = {}
    rest = sys.argv[1:]
    for i in range(0, len(rest) - 1, 2):
        overrides[rest[i].lstrip("-")] = rest[i + 1]

    saved = json.loads(Path(RUN_DIR, "eval_protocol.json").read_text(encoding="utf-8"))
    baseline_done = json.loads(
        Path(RUN_DIR, "baseline", "baseline_done.json").read_text(encoding="utf-8")
    )

    saved_sig = compute_sig({})
    print(f"reproduce saved run:      digest={saved_sig['digest']}  saved={saved['digest']}  "
          f"{'MATCH' if saved_sig['digest'] == saved['digest'] else 'MISMATCH'}")

    if overrides:
        new_sig = compute_sig(overrides)
        print(f"what-if {overrides}:")
        print(f"  digest          = {new_sig['digest']}  "
              f"{'SAME (resume ok)' if new_sig['digest'] == saved['digest'] else 'CHANGED (replay required)'}")
        print(f"  baseline_digest = {new_sig['baseline_digest']}  "
              f"{'SAME (baseline reusable via --baseline_from)' if new_sig['baseline_digest'] == baseline_done['baseline_protocol_digest'] else 'CHANGED (baseline must be rerun)'}")
        print(f"  (saved baseline_digest = {baseline_done['baseline_protocol_digest']})")
