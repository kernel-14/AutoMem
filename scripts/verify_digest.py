#!/usr/bin/env python
"""Verify the eval-protocol digest before/after the models.py patch.

Replicates engine main()'s arg/env state exactly, computes the digest, and
compares against the digest saved in runs/search/webwalkerqa-full/eval_protocol.json.
Run from repo root with the venv python.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Replicate engine module init: load .env BEFORE importing engine code paths
from dotenv import load_dotenv

load_dotenv(".env", override=False)

from automem.search import engine  # noqa: E402  (imports run load_dotenv again, harmless)

ARGV = [
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


def compute_digest() -> str:
    sys.argv = ["python"] + ARGV
    args = engine.parse_args()
    engine._apply_benchmark_split_defaults(args)
    engine._validate_search_args(args)
    from automem.search.protocol import ProtocolConfig

    proto = ProtocolConfig.resolve(args)
    args.val_every = proto.val_every
    args._protocol = proto
    sig = engine._compute_eval_protocol_signature(
        eval_model=(args.model or "").strip(),
        protocol=args._protocol,
        args=args,
    )
    return sig["digest"]


if __name__ == "__main__":
    saved = json.loads(
        Path("runs/search/webwalkerqa-full/eval_protocol.json").read_text(encoding="utf-8")
    )["digest"]
    current = compute_digest()
    print(f"saved:   {saved}")
    print(f"current: {current}")
    print("MATCH" if saved == current else "MISMATCH")
