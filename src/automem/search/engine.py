#!/usr/bin/env python
"""
automem_search.py — AutoMem: Automatic Memory Architecture Search for LLM Agents.

Replaces the old prompt-optimization loop with direct architecture space exploration
guided by an LLM that receives rich per-task attribution feedback.

Key differences from the old loop:
  - Searches the architecture configuration directly (not a meta-prompt).
  - Maintains a Pareto front across accuracy / memory_lift / hit_rate / token_eff.
  - Keeps a canonical memory pool shared across all rounds.
  - Provides the LLM with per-task attribution (extraction_gap, retrieval_miss, etc.)
  - Generates K=3 diverse candidates per round with explicit diversity roles.

Data splits:
  warmup    (profile_n, default 19)  — one-time, seeds canonical memory pool
  search    (optimization_n, default 100, sample batch_size/round)
  validation (validation_n, default 30, fixed)
  test      (final_test_n, default 15, held-out)

Usage:
    python -m automem.search.engine \\
        --run_name automem_v1 --infile /path/to/tasks.jsonl --max_rounds 8

    # Resume after interruption
    python -m automem.search.engine \\
        --run_name automem_v1 --infile /path/to/tasks.jsonl --resume

    # Dry run (skip eval subprocesses)
    python -m automem.search.engine \\
        --run_name automem_v1 --infile /path/to/tasks.jsonl --dry_run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# Match the benchmark runners: load .env so the parent process's proposer /
# diagnosis model calls resolve credentials. override=False keeps a sourced
# shell environment authoritative.
load_dotenv(override=False)

from automem.architecture.compiler import ArchitectureCompiler, RuntimeConfig
from automem.architecture.models import ArchitectureSpec
from automem.architecture_space import (
    RECOMMENDED_ARCHITECTURE_SPACE,
    get_valid_managements,
    get_valid_retrievals,
)
from automem.contracts import FitnessWeights, compute_fitness
from automem.data_split import DataSplitConfig, create_level_aware_split, create_default_split
from automem.llm_utils import load_prompt, parse_json_response, render_prompt
from automem.memory_schema import MemoryUnit
from automem.search.pareto_front import ParetoEntry, ParetoFront
from automem.search.attribution import (
    run_posthoc_audit, save_attribution_report,
)
from automem.evaluation.aggregation import build_evaluation_report, load_task_results
from automem.evaluation.utils import (
    dataset_file_sha256,
    task_identity_digest,
    task_result_validation_error,
)
from automem.endpoints import resolve_openai_endpoint
from automem.resources import prompt_path, read_prompt_bytes
from automem.runtime import DEFAULT_RUNTIME_POLICY, MemoryContextComposer, QueryPlanner

# ---------------------------------------------------------------------------
# Installed resources and user-writable output defaults
# ---------------------------------------------------------------------------
SEARCH_PROMPT = prompt_path("meta", "architecture_search.txt")
DEFAULT_RESULTS_BASE = Path("runs") / "search"
BENCHMARK_RUNNER_MODULES = {
    "gaia": "automem.benchmarks.gaia.runner",
    "webwalkerqa": "automem.benchmarks.webwalkerqa.runner",
    "xbench-deepsearch": "automem.benchmarks.xbench_deepsearch.runner",
    "xbench_deepsearch": "automem.benchmarks.xbench_deepsearch.runner",
    "xbench": "automem.benchmarks.xbench_deepsearch.runner",
    "appworld": "automem.benchmarks.appworld.runner",
}
DEFAULT_SPLIT_SIZES = (19, 100, 30, 15)
XBENCH_DEFAULT_SPLIT_SIZES = (10, 70, 10, 10)
# AppWorld train split has 90 tasks. Small-budget reproduction defaults:
# warmup / search / validation / test.
APPWORLD_DEFAULT_SPLIT_SIZES = (10, 40, 20, 20)

WARMUP_ARCHITECTURE = {
    "extract_types": ["tip"],
    "storage_routing": {"tip": "json"},
    "retrieval": "hybrid",
    "management": "lightweight",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("automem")


# ======================================================================
# CLI
# ======================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AutoMem: Automatic Memory Architecture Search")
    p.add_argument("--run_name", required=True, help="Unique name for this run")
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_RESULTS_BASE),
        help="Parent directory for search runs (default: runs/search)",
    )
    p.add_argument("--max_rounds", type=int, default=8, help="Max search rounds (default: 8)")
    p.add_argument("--num_candidates", type=int, default=3, help="Candidates per round (default: 3)")
    p.add_argument("--model", type=str, default=None, help="Task-agent model id override")
    p.add_argument(
        "--search_model",
        type=str,
        default=None,
        help="Architecture-proposer model id override",
    )
    p.add_argument("--warmup_n", type=int, default=19, help="Warm-up tasks (default: 19)")
    p.add_argument("--search_n", type=int, default=100, help="Search pool size (default: 100)")
    p.add_argument("--batch_size", type=int, default=50,
                   help="Fixed search batch size, sampled once and reused across all rounds (default: 50)")
    p.add_argument("--search_batch_seed", type=int, default=42,
                   help="Seed for sampling the fixed search batch (default: 42)")
    p.add_argument("--token_cap_per_task", type=int, default=30000,
                   help="Per-task token cap for fitness normalization (default: 30000)")
    p.add_argument("--latency_cap_per_task", type=float, default=60.0,
                   help="Per-task latency cap in seconds for fitness normalization (default: 60)")
    p.add_argument("--validation_n", type=int, default=30, help="Validation tasks (default: 30)")
    p.add_argument("--test_n", type=int, default=15, help="Final test tasks (default: 15)")
    p.add_argument("--max_steps", type=int, default=40, help="Agent max steps (default: 40)")
    p.add_argument("--token_budget", type=int, default=8192, help="Token budget per task (default: 8192)")
    p.add_argument("--concurrency", type=int, default=1, help="Eval concurrency (default: 1)")
    p.add_argument("--infile", type=str, required=True,
                   help="Task metadata path (GAIA/WebWalker JSONL or xBench CSV)")
    p.add_argument("--eval_script", type=str, default=None,
                   help="Optional custom eval runner path. By default the installed "
                        "runner matching --benchmark is launched with python -m.")
    p.add_argument("--data_split", type=str, default=None,
                   help="Path to a custom data_split JSON (optimization/validation/final_test indices). "
                        "Overrides the GAIA default split lookup; use for xBench.")
    p.add_argument("--baseline_from", type=str, default=None,
                   help="Reuse a prior run's no-memory baseline instead of re-running it "
                        "(~1h + tokens saved). Point at a previous run's baseline/ dir or its "
                        "baseline_done.json. Reused only when task indices, backbone model, and "
                        "baseline protocol digest match exactly.")
    p.add_argument("--judge_model", type=str, default=None,
                   help="Judge model id forwarded to the eval runner (--judge_model). "
                        "Default: runner's own default. For xBench use qwen3.5-122b-a10b.")
    p.add_argument("--search_prompt", type=str, default=None,
                   help="Path to the proposer (architect) prompt template. "
                        "Defaults to the architecture_search.txt resource shipped "
                        "inside the automem package.")
    p.add_argument("--benchmark", type=str, default="GAIA",
                   help="Benchmark name shown to the proposer ({{ benchmark_name }} in "
                        "the prompt). Default GAIA (unchanged behaviour). Set to e.g. "
                        "xBench-DeepSearch / WebWalkerQA so the proposer optimises for "
                        "the right task distribution.")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted run")
    p.add_argument("--dry_run", action="store_true", help="Skip eval subprocesses")
    p.add_argument("--final_validation", action="store_true",
                   help="Evaluate the runoff winner on the held-out final-test split")
    p.add_argument("--diagnosis_model", type=str, default=None,
                   help="Model for layer diagnosis (default: DIAGNOSIS_MODEL or gpt-5.5)")
    p.add_argument("--random_search", action="store_true",
                   help="Sample candidates uniformly at random from the architecture space "
                        "instead of calling an LLM. Also disables layer diagnosis. "
                        "Used as the baseline ablation for LLM-driven search.")
    p.add_argument("--random_search_seed", type=int, default=123,
                   help="Seed for random candidate sampling (default: 123)")
    p.add_argument("--disable_elitism", action="store_true",
                   help="Disable champion injection (no elitism). Each round's "
                        "candidates are entirely freshly sampled / proposed; "
                        "the Pareto best is not auto-included. Used in "
                        "pure-random baseline experiments — random should not "
                        "inherit AutoMem's elitism for free.")
    p.add_argument("--no_canonical_import", action="store_true",
                   help="Skip import_canonical_to_storage; every candidate "
                        "starts with an EMPTY memory store. Memory still "
                        "accumulates within a candidate's 50 task subprocess "
                        "(task N's extraction is visible to task N+1), but "
                        "not across candidates / rounds. Combined with "
                        "--warmup_n 0 this produces a zero-memory-init random "
                        "baseline — useful for sanity-check, NOT comparable "
                        "to AutoMem main results (which uses canonical pool).")
    p.add_argument("--discovery_thresholds", type=str,
                   default="0.55,0.60,0.65,0.70",
                   help="Comma-separated accuracy thresholds for the "
                        "rounds_to_discovery tracker (default: 0.55,0.60,0.65,0.70). "
                        "Records the first round_id at which cumulative_max_accuracy "
                        "first crosses each threshold; useful for comparing search "
                        "algorithms by speed-to-discovery, not just final max.")
    p.add_argument("--no_ledger", action="store_true",
                   help="Disable Experience Ledger (structured cross-round "
                        "memory built up via diagnosis LLM after each round). "
                        "When set, the proposer sees no accumulated principles "
                        "— used for ablation comparison vs ledger-on runs.")
    p.add_argument("--obs_graph_enabled", action="store_true",
                   help="Enable the Observation Graph: a rule-based structural "
                        "experience map (task-pattern x extract-combo x retriever "
                        "performance) accumulated each round and shown to the "
                        "architecture Proposer as extra context. Does NOT change "
                        "runtime extraction/retrieval/storage/management; it only "
                        "augments the Proposer's prompt. Run with and without this "
                        "flag (same seed/split) for a clean paired ablation.")
    args = p.parse_args()
    split_flags = ("--warmup_n", "--search_n", "--validation_n", "--test_n")
    argv = sys.argv[1:]
    args._split_sizes_explicit = any(
        token == flag or token.startswith(f"{flag}=")
        for token in argv
        for flag in split_flags
    )
    return args


def _apply_benchmark_split_defaults(args: argparse.Namespace) -> None:
    """Apply the current xBench split only when the user supplied no split."""

    benchmark = str(getattr(args, "benchmark", "") or "").strip().lower()
    current = tuple(
        getattr(args, field)
        for field in ("warmup_n", "search_n", "validation_n", "test_n")
    )
    if (
        "xbench" in benchmark
        and not getattr(args, "data_split", None)
        and not getattr(args, "_split_sizes_explicit", False)
        and current == DEFAULT_SPLIT_SIZES
    ):
        (
            args.warmup_n,
            args.search_n,
            args.validation_n,
            args.test_n,
        ) = XBENCH_DEFAULT_SPLIT_SIZES
        logger.info(
            "Applied current xBench split defaults: warmup/search/validation/test=%s",
            "/".join(str(value) for value in XBENCH_DEFAULT_SPLIT_SIZES),
        )
    elif (
        benchmark == "appworld"
        and not getattr(args, "data_split", None)
        and not getattr(args, "_split_sizes_explicit", False)
        and current == DEFAULT_SPLIT_SIZES
    ):
        (
            args.warmup_n,
            args.search_n,
            args.validation_n,
            args.test_n,
        ) = APPWORLD_DEFAULT_SPLIT_SIZES
        logger.info(
            "Applied AppWorld split defaults: warmup/search/validation/test=%s",
            "/".join(str(value) for value in APPWORLD_DEFAULT_SPLIT_SIZES),
        )


def _validate_search_args(args: argparse.Namespace) -> None:
    """Reject numeric settings that otherwise fail late or change semantics."""

    positive = (
        "max_rounds",
        "num_candidates",
        "search_n",
        "batch_size",
        "max_steps",
        "token_budget",
        "concurrency",
        "token_cap_per_task",
        "latency_cap_per_task",
    )
    for field in positive:
        value = getattr(args, field, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"--{field} must be greater than zero")
    for field in ("warmup_n", "validation_n", "test_n"):
        value = getattr(args, field, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"--{field} must be a non-negative integer")
    if getattr(args, "final_validation", False) and args.test_n == 0:
        raise ValueError("--final_validation requires --test_n greater than zero")


# ======================================================================
# Model loading
# ======================================================================

def _resolve_search_model(args: Any = None) -> Tuple[str, str]:
    """Resolve the proposer model once for both execution and protocol signing."""

    cli_model = str(getattr(args, "search_model", "") or "").strip()
    env_model = (os.environ.get("SEARCH_MODEL") or "").strip()
    default_model = (os.environ.get("DEFAULT_MODEL") or "").strip()
    if cli_model:
        return cli_model, "--search_model"
    if env_model:
        return env_model, "SEARCH_MODEL"
    if default_model:
        return default_model, "DEFAULT_MODEL"
    return "qwen-plus", "default"


def load_model(args: argparse.Namespace):
    """Load the search (architecture proposer) LLM model.

    The task model and proposer are separate protocol roles. ``--model`` is
    forwarded only to the task runner; the proposer resolves from the explicit
    ``--search_model`` override, then SEARCH_MODEL, then DEFAULT_MODEL.
    """
    from flashoagents.models import OpenAIServerModel

    model_id, source = _resolve_search_model(args)
    api_key, api_base = resolve_openai_endpoint("SEARCH")
    model = OpenAIServerModel(model_id=model_id, api_base=api_base, api_key=api_key)
    logger.info("Proposer LLM: %s (api_base=%s, source=%s)", model_id, api_base, source)
    return model


def load_model_by_id(model_id: str):
    """Load an LLM model by explicit ID through the generic endpoint pair."""
    from flashoagents.models import OpenAIServerModel

    api_key, api_base = resolve_openai_endpoint()
    return OpenAIServerModel(model_id=model_id, api_base=api_base, api_key=api_key)


def load_diagnosis_model(model_id: Optional[str] = None):
    """Load the diagnosis-only LLM (defaults to gpt-5.5 via the dedicated
    DIAGNOSIS_API_BASE/DIAGNOSIS_API_KEY/DIAGNOSIS_MODEL env vars).

    This is used for: (a) per-round layer_diagnosis + reasoning subclass
    detection, and (b) Experience Ledger updates. Decoupled from the search
    LLM (qwen) so that diagnosis can use a stronger model without impacting
    the proposer's API budget or model identity.
    """
    from flashoagents.models import OpenAIServerModel

    resolved_model_id = model_id or os.environ.get("DIAGNOSIS_MODEL") or "gpt-5.5"
    api_key, api_base = resolve_openai_endpoint("DIAGNOSIS")
    return OpenAIServerModel(
        model_id=resolved_model_id, api_base=api_base, api_key=api_key
    )


def _initialize_search_models(args: argparse.Namespace) -> Tuple[Any, Any]:
    """Initialize proposer/diagnosis models only when a search round remains."""

    if args.random_search:
        logger.info("random_search=True: skipping LLM load and layer diagnosis.")
        return None, None
    if args.dry_run:
        return None, None

    try:
        model = load_model(args)
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize the architecture proposer; refusing to "
            "silently change a real search into dry-run mode"
        ) from e

    diagnosis_model = None
    try:
        diagnosis_model = load_diagnosis_model(args.diagnosis_model)
        logger.info(
            "Diagnosis LLM: %s (via DIAGNOSIS_API_BASE)",
            args.diagnosis_model
            or os.environ.get("DIAGNOSIS_MODEL")
            or "gpt-5.5",
        )
    except Exception as e:
        logger.warning(
            "Failed to load diagnosis model from DIAGNOSIS_* env (%s). "
            "Falling back to load_model_by_id with --diagnosis_model + "
            "OPENAI_* env. Ledger updates and layer diagnosis still run.",
            e,
        )
        try:
            fallback_model = (
                args.diagnosis_model
                or os.environ.get("DIAGNOSIS_MODEL")
                or "gpt-5.5"
            )
            diagnosis_model = load_model_by_id(fallback_model)
            logger.info("Diagnosis LLM (fallback): %s", fallback_model)
        except Exception as fallback_error:
            logger.warning(
                "Fallback diagnosis model load also failed (%s). Layer "
                "diagnosis + ledger update will be disabled this run.",
                fallback_error,
            )
    return model, diagnosis_model


# ======================================================================
# Directory management
# ======================================================================

def _atomic_write_json(path: Path, payload: Any) -> None:
    """Write one JSON checkpoint without exposing a truncated final path."""

    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def setup_run_dir(args: argparse.Namespace) -> Path:
    run_name = Path(args.run_name)
    if run_name.name != args.run_name or args.run_name in {".", ".."}:
        raise ValueError("--run_name must be a single directory name")
    run_dir = Path(args.output_dir).expanduser() / run_name
    if args.resume and not run_dir.is_dir():
        raise FileNotFoundError(f"Cannot resume missing run directory: {run_dir}")
    if run_dir.is_dir() and not args.resume and any(run_dir.iterdir()):
        raise FileExistsError(
            f"Run directory is not empty: {run_dir}. "
            "Choose a new --run_name or pass --resume to continue it."
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "canonical").mkdir(exist_ok=True)
    return run_dir


def setup_round_dir(run_dir: Path, round_id: int, candidate_id: int) -> Path:
    d = run_dir / f"round_{round_id}" / f"candidate_{candidate_id}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tasks").mkdir(exist_ok=True)
    return d


# ======================================================================
# Task loading
# ======================================================================

def load_tasks(infile: Optional[str] = None) -> List[Dict[str, Any]]:
    if not infile:
        raise ValueError("A benchmark task file is required; pass --infile PATH")
    path = Path(infile).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    tasks = []
    if path.suffix.lower() == ".csv":
        # xBench CSV: id,prompt,answer,reference_steps,canary  (prompt/answer are
        # base64+XOR-encrypted with the per-row canary). We best-effort decode
        # the prompt so layer diagnosis has readable questions; the eval runner
        # does its own decoding. No GAIA "Level" field -> uniform batch sampling.
        import csv as _csv
        import base64 as _b64
        def _xor(data: bytes, key: str) -> bytes:
            kb = key.encode("utf-8")
            return bytes(b ^ kb[i % len(kb)] for i, b in enumerate(data)) if kb else data
        seen_task_ids: set[str] = set()
        with open(path, "r", encoding="utf-8-sig") as f:
            for row_number, row in enumerate(_csv.DictReader(f), start=2):
                t = dict(row)
                task_id = str(row.get("id") or "").strip()
                if not task_id:
                    raise ValueError(f"xBench CSV row {row_number} has an empty id")
                if task_id in seen_task_ids:
                    raise ValueError(f"xBench CSV contains duplicate id: {task_id!r}")
                seen_task_ids.add(task_id)
                key = row.get("canary", "")
                try:
                    t["Question"] = _xor(_b64.b64decode(row["prompt"]), key).decode("utf-8")
                except Exception:
                    t["Question"] = ""
                t["task_id"] = task_id
                tasks.append(t)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(json.loads(line))
    logger.info("Loaded %d tasks from %s", len(tasks), path)
    return tasks


# ======================================================================
# Data splits
# ======================================================================

def create_or_load_splits(run_dir: Path, tasks: List[Dict], args: argparse.Namespace) -> DataSplitConfig:
    """Load (or build) the four-phase data split for this run.

    Preference order:
      1. ``run_dir/data_split.json`` when resuming.
      2. An explicit ``--data_split`` supplied with the benchmark data.
      3. A deterministic level-aware split generated from the input tasks.
      4. A deterministic default split when level metadata is unavailable.
    """
    split_path = run_dir / "data_split.json"

    def _load_custom_split(path: Path) -> DataSplitConfig:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"--data_split {path} must contain a JSON object")
        required = {
            "optimization_indices", "validation_indices", "final_test_indices"
        }
        missing = sorted(required - set(payload))
        if missing:
            raise ValueError(f"--data_split {path} is missing fields: {missing}")
        split = DataSplitConfig.from_dict(payload)
        ok, errs = split.validate(total_tasks=len(tasks))
        if not ok:
            raise ValueError(f"--data_split {path} is invalid: {errs}")
        return split

    custom = getattr(args, "data_split", None)
    if split_path.exists() and args.resume:
        logger.info("Loading existing data split from %s", split_path)
        split = DataSplitConfig.load(str(split_path))
        ok, errs = split.validate(total_tasks=len(tasks))
        if not ok:
            raise ValueError(f"Saved data split {split_path} is invalid: {errs}")
        if custom:
            cp = Path(custom).expanduser()
            if not cp.is_file():
                raise FileNotFoundError(f"--data_split file not found: {cp}")
            requested = _load_custom_split(cp)
            if requested.to_dict() != split.to_dict():
                raise ValueError(
                    "--resume data split differs from the split persisted in the run"
                )
        return split

    # Explicit custom split (e.g. xBench) overrides the GAIA default lookup.
    if custom:
        cp = Path(custom).expanduser()
        if not cp.is_file():
            raise FileNotFoundError(f"--data_split file not found: {cp}")
        split = _load_custom_split(cp)
        split.save(str(split_path))
        logger.info(
            "Loaded custom data split from %s (search=%d val=%d test=%d)",
            cp, len(split.optimization_indices),
            len(split.validation_indices), len(split.final_test_indices),
        )
        return split

    try:
        split = create_level_aware_split(
            tasks,
            profile_n=args.warmup_n,
            optimization_n=args.search_n,
            validation_n=args.validation_n,
            final_test_n=args.test_n,
        )
    except (KeyError, ValueError):
        logger.warning("Level-aware split failed; using default split.")
        split = create_default_split(
            total_tasks=len(tasks),
            profile_n=args.warmup_n,
            optimization_n=args.search_n,
            validation_n=args.validation_n,
            final_test_n=args.test_n,
        )

    valid, errors = split.validate(total_tasks=len(tasks))
    if not valid:
        raise ValueError(f"Data split overlap: {errors}")
    split.save(str(split_path))
    logger.info(
        "Data split: warmup=%d, search=%d, validation=%d, test=%d",
        len(split.profile_indices), len(split.optimization_indices),
        len(split.validation_indices), len(split.final_test_indices),
    )
    return split


def _validate_persisted_indices(
    values: Any,
    *,
    label: str,
    allowed: set[int],
) -> List[int]:
    if not isinstance(values, list):
        raise ValueError(f"{label} must be a JSON list")
    if any(type(value) is not int for value in values):
        raise ValueError(f"{label} contains non-integer indices")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate indices")
    outside = sorted(set(values) - allowed)
    if outside:
        raise ValueError(f"{label} contains indices outside the optimization split: {outside}")
    return list(values)


def load_or_create_search_batch(
    run_dir: Path, split: DataSplitConfig, args: argparse.Namespace,
    tasks: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    """Sample a fixed batch once from the search pool and persist it.

    All rounds evaluate candidates on the *same* tasks — this eliminates the
    cross-round batch_baseline variance that previously caused memory_lift
    to fluctuate ~17 percentage points between rounds (see automem_v2 R3→R8).

    When *tasks* is provided AND each task has a "Level" field (GAIA), the
    sampling is STRATIFIED by Level so the batch maintains the same Level
    distribution as the search pool. This makes Conditional Lift / MOR
    metrics more comparable across runs and prevents the batch from being
    accidentally dominated by Level-1 trivial tasks. Falls back to uniform
    sampling when tasks=None or Level field is missing.
    """
    batch_path = run_dir / "search_batch.json"
    pool = list(split.optimization_indices)
    expected_size = min(args.batch_size, len(pool))
    if batch_path.exists():
        data = json.loads(batch_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Invalid search batch checkpoint: {batch_path}")
        indices = _validate_persisted_indices(
            data.get("indices"),
            label="search_batch.indices",
            allowed=set(pool),
        )
        if (
            len(indices) != expected_size
            or data.get("seed") != args.search_batch_seed
            or data.get("pool_size") != len(pool)
            or data.get("n") != len(indices)
        ):
            raise ValueError(
                f"Search batch checkpoint metadata does not match the current protocol: "
                f"{batch_path}"
            )
        logger.info("Loaded fixed search batch: %d tasks (seed=%s) from %s",
                    len(indices), data.get("seed"), batch_path)
        return indices

    k = expected_size
    rng = random.Random(args.search_batch_seed)

    # Try stratified sampling by GAIA Level when tasks metadata is available.
    stratified = False
    indices: List[int] = []
    if tasks:
        level_buckets: Dict[Any, List[int]] = {}
        all_levels_present = True
        for idx in pool:
            if idx >= len(tasks):
                all_levels_present = False
                break
            lvl = tasks[idx].get("Level")
            if lvl is None:
                all_levels_present = False
                break
            level_buckets.setdefault(lvl, []).append(idx)

        if all_levels_present and level_buckets:
            # Allocate sample per bucket proportional to bucket size with
            # exact-sum-to-k guarantee. Use floor + largest-remainder method
            # rather than max(1, round(..)) which can over-allocate when
            # k < number_of_buckets (Codex CR1).
            n_total = sum(len(v) for v in level_buckets.values())
            level_keys = sorted(level_buckets.keys())  # deterministic
            raw = {lvl: k * len(level_buckets[lvl]) / n_total for lvl in level_keys}
            allocations = {lvl: int(raw[lvl]) for lvl in level_keys}  # floor
            remaining = k - sum(allocations.values())
            # Distribute remaining slots to buckets with largest fractional
            # remainder, capped at bucket size, until we hit exactly k.
            order_by_remainder = sorted(
                level_keys,
                key=lambda l: (-(raw[l] - allocations[l]), -len(level_buckets[l])),
            )
            for lvl in order_by_remainder:
                if remaining <= 0:
                    break
                if allocations[lvl] < len(level_buckets[lvl]):
                    allocations[lvl] += 1
                    remaining -= 1
            # If still remaining (impossible unless every bucket smaller than
            # alloc — only when k > pool), accept; sampler will cap.

            sampled: List[int] = []
            for lvl, idxs in level_buckets.items():
                take = min(allocations[lvl], len(idxs))
                if take > 0:
                    sampled.extend(rng.sample(idxs, take))
            # Hard cap defensively to k in case rounding still misaligned.
            if len(sampled) > k:
                sampled = rng.sample(sampled, k)
            indices = sorted(sampled)
            stratified = True
            logger.info(
                "Stratified search batch by Level: %s (totals %d / %d pool)",
                {lvl: allocations[lvl] for lvl in level_keys},
                len(indices), n_total,
            )

    if not indices:
        indices = sorted(rng.sample(pool, k))

    payload = {
        "indices": indices,
        "seed": args.search_batch_seed,
        "n": len(indices),
        "pool_size": len(pool),
        "stratified_by_level": stratified,
    }
    _atomic_write_json(batch_path, payload)
    logger.info("Created fixed search batch: %d tasks (seed=%d, pool=%d, stratified=%s)",
                len(indices), args.search_batch_seed,
                len(pool), stratified)
    return indices


def load_or_create_search_folds(
    run_dir: Path, split: DataSplitConfig, args: argparse.Namespace,
    n_folds: int,
    tasks: Optional[List[Dict[str, Any]]] = None,
) -> List[List[int]]:
    """Protocol-v2 M2: stratified fold partition of the search pool.

    Round t evaluates fold (t-1) mod n_folds, so within-round comparisons
    stay paired on identical tasks while the champion's pooled estimate
    accumulates across folds — a generalization signal the legacy fixed
    batch cannot provide. Each fold targets args.batch_size tasks; the
    sampled union therefore covers up to batch_size * n_folds tasks of
    the optimization pool (baseline already scores the full pool, so
    per-task memory_lift is available on every fold).

    The partition is persisted to search_folds.json on first creation and
    reloaded verbatim afterwards (resume safety).
    """
    from automem.search import protocol as _p2

    folds_path = run_dir / "search_folds.json"
    pool = list(split.optimization_indices)
    if folds_path.exists():
        data = json.loads(folds_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("folds"), list):
            raise ValueError(f"Invalid search folds checkpoint: {folds_path}")
        folds = [
            _validate_persisted_indices(
                values,
                label=f"search_folds.folds[{index}]",
                allowed=set(pool),
            )
            for index, values in enumerate(data["folds"])
        ]
        flattened = [value for fold in folds for value in fold]
        expected_union_size = min(len(pool), args.batch_size * n_folds)
        if (
            len(folds) != n_folds
            or any(not fold for fold in folds)
            or len(flattened) != len(set(flattened))
            or len(flattened) != expected_union_size
            or data.get("n_folds") != n_folds
            or data.get("seed") != args.search_batch_seed
            or data.get("pool_size") != len(pool)
            or data.get("fold_sizes") != [len(fold) for fold in folds]
        ):
            raise ValueError(
                f"Search folds checkpoint does not match the current protocol: "
                f"{folds_path}"
            )
        logger.info("Loaded %d search folds (sizes=%s, seed=%s) from %s",
                    len(folds), [len(x) for x in folds], data.get("seed"), folds_path)
        return folds

    if len(pool) < n_folds:
        raise ValueError(
            f"Search split has {len(pool)} tasks but the fixed protocol requires "
            f"at least {n_folds} non-empty folds"
        )
    folds = _p2.make_folds(
        pool=pool,
        n_folds=n_folds,
        seed=args.search_batch_seed,
        tasks=tasks,
        fold_size=args.batch_size,
    )
    payload = {
        "folds": folds,
        "n_folds": len(folds),
        "fold_sizes": [len(x) for x in folds],
        "seed": args.search_batch_seed,
        "pool_size": len(pool),
    }
    _atomic_write_json(folds_path, payload)
    logger.info("Created %d stratified search folds (sizes=%s, seed=%d, pool=%d)",
                len(folds), [len(x) for x in folds], args.search_batch_seed, len(pool))
    return folds


# ======================================================================
# Canonical pool management
# ======================================================================

def canonical_pool_path(run_dir: Path) -> Path:
    return run_dir / "canonical" / "pool.json"


_CANONICAL_STATE_SCHEMA = 2


def _empty_canonical_state() -> Dict[str, Any]:
    return {
        "schema_version": _CANONICAL_STATE_SCHEMA,
        "units": [],
        "applied_merges": [],
        "periodic_rounds": [],
        "graph_edges": [],
    }


def _load_canonical_state(run_dir: Path) -> Dict[str, Any]:
    """Load canonical state, accepting the legacy bare-list representation."""

    p = canonical_pool_path(run_dir)
    if not p.exists():
        return _empty_canonical_state()
    with open(p, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        state = _empty_canonical_state()
        state["units"] = payload
        return state
    if not isinstance(payload, dict) or not isinstance(payload.get("units"), list):
        raise ValueError(f"Invalid canonical state in {p}")
    state = _empty_canonical_state()
    state.update(payload)
    for key in ("applied_merges", "periodic_rounds", "graph_edges"):
        if not isinstance(state.get(key), list):
            raise ValueError(f"Invalid canonical state field {key!r} in {p}")
    state["schema_version"] = _CANONICAL_STATE_SCHEMA
    return state


def _save_canonical_state(run_dir: Path, state: Dict[str, Any]) -> None:
    """Atomically persist units and their idempotency/graph handoff metadata."""

    import tempfile

    p = canonical_pool_path(run_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".pool.json.tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, p)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


def load_canonical_pool(run_dir: Path) -> List[Dict[str, Any]]:
    return list(_load_canonical_state(run_dir)["units"])


def _canonical_state_digest(state: Dict[str, Any]) -> str:
    import hashlib

    payload = json.dumps(
        state, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _save_bound_canonical_snapshot(source_dir: Path, snapshot_dir: Path) -> str:
    state = _load_canonical_state(source_dir)
    digest = _canonical_state_digest(state)
    _save_canonical_state(snapshot_dir, state)
    _atomic_write_json(
        snapshot_dir / "snapshot_manifest.json",
        {"schema_version": 1, "sha256": digest},
    )
    return digest


def _load_bound_canonical_snapshot(snapshot_dir: Path) -> Tuple[Dict[str, Any], str]:
    state = _load_canonical_state(snapshot_dir)
    manifest_path = snapshot_dir / "snapshot_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = _canonical_state_digest(state)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("sha256") != digest
    ):
        raise ValueError("canonical snapshot manifest digest mismatch")
    return state, digest


def _compute_eval_protocol_signature(
    eval_model: str = "", protocol: Any = None, args: Any = None
) -> Dict[str, Any]:
    """Hash every material input used to produce or compare evaluations.

    A resumed search may only reuse checkpoints when this digest matches.  In
    particular, this includes the benchmark bytes, persisted split, runner and
    package source, resolved model ids, and CLI controls forwarded to the task
    agent.  Secrets are deliberately excluded.
    """
    import hashlib as _hashlib

    digest = _hashlib.sha256()
    src_root = Path(__file__).resolve().parents[2]

    def _update_component(name: str, value: str | bytes) -> None:
        raw = value if isinstance(value, bytes) else value.encode("utf-8")
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)

    def _file_digest(path: Path) -> str:
        file_hash = _hashlib.sha256()
        with open(path, "rb") as stream:
            while chunk := stream.read(1024 * 1024):
                file_hash.update(chunk)
        return file_hash.hexdigest()

    def _resolved_path(raw_path: str) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        return path.resolve()

    prompt_parts = [
        ("tips_prompt.txt",),
        ("insights_prompt.txt",),
        ("workflow_prompt.txt",),
        ("trajectory_prompt.txt",),
        ("shortcut_prompt.txt",),
        ("meta", "layer_diagnosis_fixed.txt"),
    ]
    for parts in prompt_parts:
        _update_component("prompt:" + "/".join(parts), read_prompt_bytes(*parts))

    # Search, diagnosis, ledger, classifier, and task-agent prompts are code
    # dependencies even when their paths are loaded indirectly by another
    # module. Hash the complete shipped prompt trees so resume cannot mix two
    # template revisions while the Python source stays unchanged.
    prompt_resource_hash = _hashlib.sha256()
    prompt_resource_count = 0
    for prompt_root in (
        src_root / "automem" / "prompts",
        src_root / "flashoagents" / "prompts",
    ):
        if not prompt_root.is_dir():
            continue
        for resource_path in sorted(prompt_root.rglob("*")):
            if not resource_path.is_file() or resource_path.suffix.lower() not in {
                ".txt",
                ".yaml",
                ".yml",
            }:
                continue
            relative = resource_path.relative_to(src_root).as_posix()
            prompt_resource_hash.update(relative.encode("utf-8"))
            prompt_resource_hash.update(b"\0")
            prompt_resource_hash.update(bytes.fromhex(_file_digest(resource_path)))
            prompt_resource_count += 1
    prompt_resource_sha = prompt_resource_hash.hexdigest()
    _update_component("prompt_resources_sha256", prompt_resource_sha)

    search_prompt_arg = (
        (getattr(args, "search_prompt", "") or "") if args is not None else ""
    )
    if search_prompt_arg:
        search_prompt_path = Path(search_prompt_arg).expanduser()
        if not search_prompt_path.is_absolute():
            search_prompt_path = Path.cwd() / search_prompt_path
        search_prompt_bytes = search_prompt_path.read_bytes()
        search_prompt_display = str(search_prompt_path.resolve())
    else:
        search_prompt_bytes = read_prompt_bytes("meta", "architecture_search.txt")
        search_prompt_display = "automem:prompts/meta/architecture_search.txt"
    _update_component("search_prompt", search_prompt_bytes)

    try:
        from flashoagents.memory import L1_MEMORY_INSTRUCTION_PREFIX, L2_ACKNOWLEDGE_INSTRUCTION
        _update_component("l1_memory_instruction", L1_MEMORY_INSTRUCTION_PREFIX)
        _update_component("l2_memory_instruction", L2_ACKNOWLEDGE_INSTRUCTION)
    except Exception:
        pass

    _update_component("runtime_policy", DEFAULT_RUNTIME_POLICY.digest)
    _update_component("query_planner_prompt", QueryPlanner._PROMPT)
    _update_component("memory_context_composer_system", MemoryContextComposer._SYSTEM)

    resolved_eval_model = (
        eval_model or os.environ.get("DEFAULT_MODEL") or "gpt-5"
    ).strip()
    resolved_search_model, _ = _resolve_search_model(args)
    judge_model_arg = (
        (getattr(args, "judge_model", "") or "") if args is not None else ""
    ).strip()
    benchmark_for_models = str(
        getattr(args, "benchmark", "gaia") if args is not None else "gaia"
    ).strip().lower()
    runner_judge_default = (
        os.environ.get("DEFAULT_MODEL") or "gpt-5"
        if benchmark_for_models == "gaia"
        else "gpt-5"
    )
    resolved_judge_model = (
        judge_model_arg
        or os.environ.get("DEFAULT_JUDGE_MODEL")
        or runner_judge_default
    ).strip()
    diagnosis_model_arg = (
        (getattr(args, "diagnosis_model", "") or "") if args is not None else ""
    ).strip()
    resolved_diagnosis_model = (
        diagnosis_model_arg or os.environ.get("DIAGNOSIS_MODEL") or "gpt-5.5"
    ).strip()
    execution_mode = (
        "dry_run" if args is not None and getattr(args, "dry_run", False) else "online"
    )
    _update_component("eval_model", resolved_eval_model)
    _update_component("search_model", resolved_search_model)
    _update_component("judge_model", resolved_judge_model)
    _update_component("diagnosis_model", resolved_diagnosis_model)
    _update_component("execution_mode", execution_mode)

    # Endpoint identity affects provider behaviour, but credentials must never
    # be written to eval_protocol.json or folded into diagnostic output.
    from urllib.parse import urlsplit, urlunsplit

    def _public_endpoint(raw_value: str) -> str:
        """Remove URL credentials, query strings, and fragments before signing."""

        raw_value = str(raw_value or "").strip()
        if not raw_value:
            return ""
        try:
            parsed = urlsplit(raw_value)
            host = parsed.hostname or ""
            if ":" in host and not host.startswith("["):
                host = f"[{host}]"
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
        except (TypeError, ValueError):
            return raw_value.split("?", 1)[0].split("#", 1)[0]

    generic_api_base = _public_endpoint(
        os.environ.get("OPENAI_API_BASE")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    task_endpoint_role = "TASK" if benchmark_for_models == "gaia" else "OPENAI"
    task_api_base = (
        _public_endpoint(os.environ.get("TASK_API_BASE") or generic_api_base)
        if task_endpoint_role == "TASK"
        else generic_api_base
    )
    diagnosis_role_configured = bool(
        os.environ.get("DIAGNOSIS_API_KEY")
        and os.environ.get("DIAGNOSIS_API_BASE")
    )
    endpoint_fields = {
        "generic_api_base": generic_api_base,
        "task_endpoint_role": task_endpoint_role,
        "task_api_base": task_api_base,
        "task_role_configured": bool(
            os.environ.get("TASK_API_KEY") and os.environ.get("TASK_API_BASE")
        ),
        "judge_api_base": _public_endpoint(
            os.environ.get("JUDGE_API_BASE") or generic_api_base
        ),
        "judge_role_configured": bool(
            os.environ.get("JUDGE_API_KEY") and os.environ.get("JUDGE_API_BASE")
        ),
        "search_api_base": _public_endpoint(
            os.environ.get("SEARCH_API_BASE") or generic_api_base
        ),
        "search_role_configured": bool(
            os.environ.get("SEARCH_API_KEY") and os.environ.get("SEARCH_API_BASE")
        ),
        "diagnosis_api_base": _public_endpoint(
            os.environ.get("DIAGNOSIS_API_BASE")
            if diagnosis_role_configured
            else generic_api_base
        ),
        "diagnosis_role_configured": diagnosis_role_configured,
        "jina_api_base": _public_endpoint(os.environ.get("JINA_API_BASE", "")),
        "mtu_api_base": _public_endpoint(os.environ.get("MTU_BASE_URL", "")),
    }
    _update_component("endpoints", json.dumps(endpoint_fields, sort_keys=True))

    from flashoagents.cache.config import CacheConfig

    cache_config = CacheConfig.from_env()
    proxy_value = (
        os.environ.get("CRAWL_PROXY")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("https_proxy")
        or os.environ.get("HTTP_PROXY")
        or os.environ.get("http_proxy")
        or ""
    )
    serper_keys = [
        value.strip()
        for value in os.environ.get("SERPER_API_KEYS", "").split(",")
        if value.strip()
    ]
    if not serper_keys and os.environ.get("SERPER_API_KEY"):
        serper_keys = ["configured"]
    environment_controls = {
        "force_stream": os.environ.get("FORCE_STREAM", "").strip().lower(),
        "task_token_cap": os.environ.get("TASK_TOKEN_CAP", "").strip(),
        "web_search_provider": os.environ.get("WEB_SEARCH_PROVIDER", "serper")
        .strip()
        .lower(),
        "web_access_provider": os.environ.get("WEB_ACCESS_PROVIDER", "jina")
        .strip()
        .lower(),
        "crawl_proxy": _public_endpoint(proxy_value),
        "search_cache_enabled": cache_config.enable_search_cache,
        "page_cache_enabled": cache_config.enable_page_cache,
        "cache_dir": str(cache_config.cache_dir.expanduser().resolve()),
        "search_cache_ttl_seconds": cache_config.search_ttl_seconds,
        "page_cache_ttl_seconds": cache_config.page_ttl_seconds,
        "freeze_cache": cache_config.freeze_cache,
        "serper_key_count": len(serper_keys),
        "jina_configured": bool(os.environ.get("JINA_API_KEY")),
        "github_configured": bool(os.environ.get("GITHUB_TOKEN")),
        "mtu_configured": bool(
            os.environ.get("MTU_API_KEY") and os.environ.get("MTU_BASE_URL")
        ),
    }
    _update_component(
        "environment_controls",
        json.dumps(environment_controls, sort_keys=True, separators=(",", ":")),
    )

    behavior_fields = (
        "benchmark",
        "max_rounds",
        "num_candidates",
        "warmup_n",
        "search_n",
        "batch_size",
        "search_batch_seed",
        "token_cap_per_task",
        "latency_cap_per_task",
        "validation_n",
        "test_n",
        "max_steps",
        "token_budget",
        "concurrency",
        "random_search",
        "random_search_seed",
        "disable_elitism",
        "no_canonical_import",
        "discovery_thresholds",
        "no_ledger",
        "obs_graph_enabled",
    )
    behavior = {
        field: getattr(args, field, None) if args is not None else None
        for field in behavior_fields
    }
    _update_component(
        "behavior_args",
        json.dumps(behavior, sort_keys=True, separators=(",", ":"), default=str),
    )

    split_payload = getattr(args, "_resolved_split", None) if args is not None else None
    if split_payload is not None:
        _update_component(
            "data_split",
            json.dumps(split_payload, sort_keys=True, separators=(",", ":")),
        )

    infile_display = ""
    infile_sha = ""
    asset_manifest: List[Dict[str, Any]] = []
    infile_arg = (getattr(args, "infile", "") or "") if args is not None else ""
    if infile_arg:
        infile_path = _resolved_path(infile_arg)
        if not infile_path.is_file():
            raise FileNotFoundError(f"Task file not found while signing protocol: {infile_path}")
        infile_display = str(infile_path)
        infile_sha = _file_digest(infile_path)
        _update_component("infile_path", infile_display)
        _update_component("infile_sha256", infile_sha)

        # GAIA attachments are material task inputs too. Only fingerprint
        # paths confined to the task file's directory; untrusted metadata
        # must not turn protocol signing into an arbitrary-file reader.
        for index, task in enumerate(
            getattr(args, "_loaded_tasks", []) if args is not None else []
        ):
            if not isinstance(task, dict):
                continue
            raw_asset = task.get("file_name")
            if not isinstance(raw_asset, str) or not raw_asset.strip():
                continue
            candidate = Path(raw_asset).expanduser()
            if not candidate.is_absolute():
                candidate = infile_path.parent / candidate
            resolved_asset = candidate.resolve()
            try:
                relative_asset = resolved_asset.relative_to(infile_path.parent)
            except ValueError:
                asset_manifest.append(
                    {"task": index, "path": raw_asset, "status": "outside_input_root"}
                )
                continue
            entry: Dict[str, Any] = {
                "task": index,
                "path": relative_asset.as_posix(),
            }
            if resolved_asset.is_file():
                entry["sha256"] = _file_digest(resolved_asset)
                entry["size"] = resolved_asset.stat().st_size
            else:
                entry["status"] = "missing"
            asset_manifest.append(entry)
        _update_component(
            "task_assets",
            json.dumps(asset_manifest, sort_keys=True, separators=(",", ":")),
        )

    eval_script_arg = (
        (getattr(args, "eval_script", "") or "") if args is not None else ""
    )
    if eval_script_arg:
        runner_path = _resolved_path(eval_script_arg)
        if not runner_path.is_file():
            raise FileNotFoundError(
                f"Custom eval runner not found while signing protocol: {runner_path}"
            )
        runner_display = str(runner_path)
        runner_sha = _file_digest(runner_path)
        _update_component("eval_runner_path", runner_display)
        _update_component("eval_runner_sha256", runner_sha)
    else:
        benchmark_key = str(behavior.get("benchmark") or "gaia").strip().lower()
        runner_display = BENCHMARK_RUNNER_MODULES.get(benchmark_key, "")
        runner_sha = ""
        _update_component("eval_runner_module", runner_display)

    # A package-source fingerprint also catches changes in transitive runner,
    # memory, grading, and search code that a single entry-point hash misses.
    source_hash = _hashlib.sha256()
    source_count = 0
    for package_name in ("automem", "flashoagents"):
        package_root = src_root / package_name
        if not package_root.is_dir():
            continue
        for source_path in sorted(package_root.rglob("*.py")):
            relative = source_path.relative_to(src_root).as_posix()
            source_hash.update(relative.encode("utf-8"))
            source_hash.update(b"\0")
            source_hash.update(bytes.fromhex(_file_digest(source_path)))
            source_count += 1
    source_sha = source_hash.hexdigest()
    _update_component("package_source_sha256", source_sha)

    import importlib.metadata as _metadata
    import platform as _platform

    dependency_names = (
        "filelock",
        "jinja2",
        "networkx",
        "numpy",
        "openai",
        "pandas",
        "pydantic",
        "scikit-learn",
        "sentence-transformers",
    )
    dependency_versions: Dict[str, str] = {}
    for dependency_name in dependency_names:
        try:
            dependency_versions[dependency_name] = _metadata.version(dependency_name)
        except _metadata.PackageNotFoundError:
            dependency_versions[dependency_name] = "not-installed"
    environment_fields = {
        "python": _platform.python_version(),
        "implementation": _platform.python_implementation(),
        "dependencies": dependency_versions,
    }
    _update_component(
        "execution_environment",
        json.dumps(environment_fields, sort_keys=True, separators=(",", ":")),
    )

    asset_manifest_sha = _hashlib.sha256(
        json.dumps(asset_manifest, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    baseline_material = {
        "benchmark": behavior.get("benchmark"),
        "eval_model": resolved_eval_model,
        "judge_model": resolved_judge_model,
        "max_steps": behavior.get("max_steps"),
        "token_budget": behavior.get("token_budget"),
        "infile_sha256": infile_sha,
        "task_assets_sha256": asset_manifest_sha,
        "eval_runner": runner_display,
        "eval_runner_sha256": runner_sha,
        "package_source_sha256": source_sha,
        "endpoints": endpoint_fields,
        "environment": environment_fields,
        "environment_controls": environment_controls,
    }
    baseline_digest = _hashlib.sha256(
        json.dumps(
            baseline_material,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:24]

    sig: Dict[str, Any] = {
        "eval_model": resolved_eval_model,
        "default_model": os.environ.get("DEFAULT_MODEL", ""),
        "search_model": resolved_search_model,
        "judge_model": resolved_judge_model,
        "diagnosis_model": resolved_diagnosis_model,
        "execution_mode": execution_mode,
        "runtime_policy_digest": DEFAULT_RUNTIME_POLICY.digest,
        "behavior": behavior,
        "post_search_actions": {
            "final_validation": bool(
                getattr(args, "final_validation", False) if args is not None else False
            )
        },
        "data_split": split_payload,
        "endpoints": endpoint_fields,
        "environment": environment_fields,
        "environment_controls": environment_controls,
        "baseline_digest": baseline_digest,
    }
    if protocol is not None:
        try:
            pf = protocol.digest_fields()
        except Exception:
            pf = None
        if pf:
            _update_component("evaluation_protocol", json.dumps(pf, sort_keys=True))
            sig["evaluation_protocol"] = pf
    sig["digest"] = digest.hexdigest()[:24]
    sig["runtime_info"] = {
        "search_prompt_path": search_prompt_display,
        "search_prompt_sha": _hashlib.sha256(search_prompt_bytes).hexdigest()[:12],
        "infile_path": infile_display,
        "infile_sha256": infile_sha,
        "task_asset_count": len(asset_manifest),
        "task_assets_sha256": asset_manifest_sha,
        "eval_runner": runner_display,
        "eval_runner_sha256": runner_sha,
        "package_source_sha256": source_sha,
        "package_source_files": source_count,
        "prompt_resources_sha256": prompt_resource_sha,
        "prompt_resource_files": prompt_resource_count,
    }
    return sig


def _run_canonical_periodic_ops(
    run_dir: Path, round_id: Optional[int] = None
) -> None:
    """Run mandatory periodic ops directly on canonical/pool.json.

    Codex Q4-1 fix (2026-04-28): R3-1 made candidate-side deactivations
    NOT propagate to canonical (to avoid one architecture's bad call
    silencing a unit forever). The flip side is that nothing prunes
    canonical — so the only safety net is to run the mandatory periodic
    ops (`utility_audit`, `size_capped_prune`, `quality_curation`)
    directly on canonical after the round-end sync.

    We back the canonical pool by an in-memory JsonStorage so the ops can
    operate on real `MemoryUnit` objects, then write the resulting state
    back to canonical/pool.json.
    """
    from automem.management.ops.utility_audit import UtilityAuditOp
    from automem.management.ops.size_capped_prune import SizeCappedPruneOp
    from automem.management.ops.quality_curation import QualityCurationOp
    from automem.storage.json_storage import JsonStorage

    state = _load_canonical_state(run_dir)
    round_key = str(round_id) if round_id is not None else None
    if round_key is not None and round_key in {
        str(value) for value in state["periodic_rounds"]
    }:
        logger.info("Canonical periodic ops already applied for round %s; skipping.", round_key)
        return

    canon = list(state["units"])
    if not canon:
        if round_key is not None:
            state["periodic_rounds"].append(round_key)
            _save_canonical_state(run_dir, state)
        return

    # Build an ephemeral JsonStorage backed by a tmpfile so the mandatory
    # ops have real .get_all / .update methods.
    # Codex Q5-A1 fix (2026-04-28): JsonStorage takes a CONFIG DICT, not a
    # path string. Must also call initialize() to load the file. The prior
    # broken code raised TypeError silently in the outer try/except, which
    # made the whole Q4-1 canonical-prune feature a no-op.
    import tempfile, json as _json
    with tempfile.TemporaryDirectory(prefix="canon_audit_") as td:
        store_dir = Path(td)
        db_path = store_dir / "memory_db.json"
        with open(db_path, "w") as f:
            _json.dump(canon, f)
        store = JsonStorage({"db_path": str(db_path)})
        store.initialize()

        # Quality curation first — adjusts confidence so utility_audit sees
        # current scores. Then utility_audit (with stale_unused enabled
        # ONLY here, see Codex Q4-6 in utility_audit.py for why). Then
        # size_capped_prune.
        op_configs = {
            "UtilityAuditOp": {"handle_stale_unused": True},
        }
        for op_cls in (QualityCurationOp, UtilityAuditOp, SizeCappedPruneOp):
            try:
                cfg = op_configs.get(op_cls.__name__, {})
                op = op_cls(store=store, config=cfg)
                res = op.execute({})
                logger.info(
                    "canonical periodic %s: deactivated=%d affected=%d",
                    op.op_name,
                    res.units_deleted, res.units_affected,
                )
            except Exception as e:
                raise RuntimeError(
                    f"canonical periodic {op_cls.__name__} failed: {e}"
                ) from e

        # Persist back. Note: only is_active and counter fields may have
        # changed; embeddings/content stay intact.
        #
        # Codex Q6-A6 fix (2026-04-28): only persist ACTIVE units. The
        # canonical pool is the source-of-truth that every candidate
        # imports at round start. Persisting inactive units sends them
        # back into vector/hybrid stores where they enter FAISS (Q5-A3
        # only compacts if the local store has compact()) and silently
        # consume top-k slots. pool_stats also counts them toward
        # available memory, biasing the architect's view of pool size.
        state["units"] = [u.to_dict() for u in store.get_all() if u.is_active]
        if round_key is not None:
            state["periodic_rounds"].append(round_key)
        _save_canonical_state(run_dir, state)
        logger.info("Canonical pool saved: %d units", len(state["units"]))


def save_canonical_pool(run_dir: Path, units: List[Dict[str, Any]]) -> None:
    state = _load_canonical_state(run_dir)
    state["units"] = units
    _save_canonical_state(run_dir, state)
    logger.info("Canonical pool saved: %d units", len(units))


def _make_storage_backend(store_type: str, store_dir: Path):
    """Instantiate and initialize a storage backend at ``store_dir``."""
    from automem.storage import (
        GraphStore,
        HybridStorage,
        JsonStorage,
        LLMGraphStore,
        VectorStorage,
    )

    store_dir = Path(store_dir)
    factories = {
        "json": lambda: JsonStorage({"db_path": str(store_dir / "memory_db.json")}),
        "vector": lambda: VectorStorage({"storage_dir": str(store_dir)}),
        "hybrid": lambda: HybridStorage({"storage_dir": str(store_dir)}),
        "graph": lambda: GraphStore({"storage_dir": str(store_dir)}),
        # Canonical-pool maintenance only loads or seeds persisted units. It
        # must not run extraction in the search coordinator process; task
        # runners construct the real backend with their injected task model.
        "llm_graph": lambda: LLMGraphStore(
            {"storage_dir": str(store_dir), "maintenance_mode": True}
        ),
    }

    if store_type not in factories:
        raise ValueError(f"Unknown store_type '{store_type}'")

    store = factories[store_type]()
    if not store.initialize():
        raise RuntimeError(f"Failed to initialize {store_type} store at {store_dir}")
    return store


def _discover_storage_artifacts(root_dir: Path) -> List[Tuple[str, Path]]:
    """Infer store backends from persisted artifacts under ``root_dir``."""
    discovered: List[Tuple[str, Path]] = []
    seen: set[Tuple[str, str]] = set()

    patterns = ("memory_db.json", "metadata.json", "graph.json")
    for pattern in patterns:
        for artifact in root_dir.rglob(pattern):
            store_dir = artifact.parent
            if pattern == "metadata.json":
                store_type = "vector"
            elif pattern == "graph.json":
                store_type = (
                    "llm_graph"
                    if store_dir.name == "store_llm_graph"
                    else "graph"
                )
            else:
                store_type = "hybrid" if (store_dir / "faiss.index").exists() else "json"

            key = (store_type, str(store_dir.resolve()))
            if key in seen:
                continue
            seen.add(key)
            discovered.append((store_type, store_dir))

    return discovered


def pool_stats(units: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a summary of the canonical pool for LLM context."""
    type_counts: Dict[str, int] = {}
    for u in units:
        mem_type = u.get("memory_type", u.get("type", "unknown"))
        type_counts[mem_type] = type_counts.get(mem_type, 0) + 1

    # Cold-start thresholds
    thresholds = {"json": 5, "vector": 20, "hybrid": 15, "graph": 30, "llm_graph": 50}
    safe_backends = [b for b, t in thresholds.items() if len(units) >= t]

    return {
        "total_units": len(units),
        "by_type": type_counts,
        "safe_backends": safe_backends,
        "cold_start_thresholds": thresholds,
        "warning": (
            "Pool has fewer than 20 units — avoid vector/graph backends."
            if len(units) < 20 else None
        ),
    }


def sync_tasks_to_canonical(
    run_dir: Path,
    tasks_dir: Path,
    round_start_pool_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
    merge_id: Optional[str] = None,
) -> int:
    """Export new MemoryUnits from tasks_dir back to the canonical pool.

    Counter merge model (revised 2026-05-13):
      • Candidate-local counters (usage / success / access / conflict) are
        zeroed at import_canonical_to_storage time, so each candidate's
        post-task counter is the genuine delta from this batch alone.
        sync therefore accumulates `existing += cand_val` directly — no
        subtraction needed. The N-candidate accumulation is intentional:
        canonical reflects total cross-candidate usage of each unit.
      • Replaces the prior delta = cand_val - round_start_val formula which
        assumed candidate-local values included round-start canonical. That
        assumption broke whenever a persistence bug in the candidate's local
        boost path (R3-2 fix era) ran cand_val far above round_start; the
        delta then re-amplified across each sync. Observed in run9:
        canonical max success_count grew to 87096 = 3 × 29844 (one bad
        candidate replicated across 3 rounds of sync).

      • is_active=False is NOT propagated to canonical (R3-1, kept).
        Re-activation IS propagated.
      • superseded_by / confidence / decay_weight take latest non-empty
        candidate value (R3-2 latest-wins, kept).
      • round_start_pool_by_id is unused but kept in signature for backwards
        compatibility; pre-2026-05-13 callers passed a round-start snapshot.
    """
    _ = round_start_pool_by_id  # accept-and-ignore for backwards compat
    state = _load_canonical_state(run_dir)
    stable_merge_id = merge_id or f"storage:{tasks_dir.resolve()}"
    if stable_merge_id in set(str(value) for value in state["applied_merges"]):
        logger.info("Canonical merge %s already applied; skipping.", stable_merge_id)
        return 0

    canon = list(state["units"])
    canon_by_id: Dict[str, Dict[str, Any]] = {u.get("id"): u for u in canon if u.get("id")}
    new_count = 0
    merged_count = 0
    candidate_graph_edges: List[Dict[str, Any]] = []

    _DELTA_FIELDS = (
        "usage_count",
        "success_count",
        "access_count",
        "conflict_count",
    )
    _LATEST_FIELDS = (
        "last_accessed",
        "decay_weight",
        "superseded_by",
        "confidence",
    )

    def _absorb(units: List[MemoryUnit]) -> None:
        nonlocal new_count, merged_count
        for u in units:
            unit_dict = u.to_dict()
            uid = unit_dict.get("id")
            if not uid:
                continue
            existing = canon_by_id.get(uid)
            if existing is None:
                canon.append(unit_dict)
                canon_by_id[uid] = unit_dict
                new_count += 1
                continue
            changed = False
            # ---- accumulate per-candidate deltas (candidate started at 0) ----
            for fld in _DELTA_FIELDS:
                if fld not in unit_dict:
                    continue
                cand_val = int(unit_dict.get(fld) or 0)
                if cand_val <= 0:
                    continue  # candidate did not advance the counter
                new_val = int(existing.get(fld, 0) or 0) + cand_val
                if new_val != existing.get(fld):
                    existing[fld] = new_val
                    changed = True
            # ---- is_active: only propagate re-activation (R3-1) ----
            cand_active = unit_dict.get("is_active")
            cur_active = existing.get("is_active", True)
            if cand_active is True and cur_active is False:
                existing["is_active"] = True
                changed = True
            # NOTE: deactivation is candidate-local; canonical's is_active
            # may only be flipped to False by canonical-level ops (e.g. when
            # quality_curation runs over canonical itself or via an
            # explicit utility_audit on canonical) — never as a side effect
            # of one candidate's pruning decision.
            # ---- latest-wins fields ----
            for fld in _LATEST_FIELDS:
                if fld not in unit_dict:
                    continue
                cand_val = unit_dict[fld]
                if cand_val in (None, ""):
                    continue
                if existing.get(fld) != cand_val:
                    existing[fld] = cand_val
                    changed = True
            if changed:
                merged_count += 1

    for store_type, store_dir in _discover_storage_artifacts(tasks_dir):
        try:
            store = _make_storage_backend(store_type, store_dir)
            _absorb(store.get_all(active_only=False))
            export_edges = getattr(store, "export_relation_edges", None)
            if callable(export_edges):
                candidate_graph_edges.extend(export_edges())
        except Exception as e:
            raise RuntimeError(
                f"Failed to load {store_type} store from {store_dir}: {e}"
            ) from e

    if candidate_graph_edges:
        by_key = {
            (
                str(edge.get("source_id") or ""),
                str(edge.get("target_id") or ""),
                str(edge.get("edge_type") or ""),
            ): edge
            for edge in state["graph_edges"]
            if isinstance(edge, dict)
        }
        for edge in candidate_graph_edges:
            key = (
                str(edge.get("source_id") or ""),
                str(edge.get("target_id") or ""),
                str(edge.get("edge_type") or ""),
            )
            if all(key):
                by_key[key] = edge
        state["graph_edges"] = [by_key[key] for key in sorted(by_key)]

    # Units, graph state, and the merge receipt are one atomic transaction.
    # A crash therefore cannot persist a counter delta without also persisting
    # the idempotency key that prevents it from being applied again on resume.
    state["units"] = canon
    state["applied_merges"].append(stable_merge_id)
    _save_canonical_state(run_dir, state)
    logger.info(
        "Synced %d new + %d merged units to canonical pool (total: %d, merge=%s)",
        new_count, merged_count, len(canon), stable_merge_id,
    )
    return new_count


def import_canonical_to_storage(run_dir: Path, runtime_config: RuntimeConfig) -> int:
    """Import canonical units into compiled stores via backend APIs.

    Only units enabled by the compiled ``extract_plan`` are imported, and each
    unit is routed to the backend specified by ``storage_routing``.

    Counter reset (2026-05-13): candidate-local counters (access_count,
    usage_count, success_count, conflict_count) are zeroed before insertion
    so that sync_tasks_to_canonical's "delta = cand_val" genuinely reflects
    THIS candidate's contribution over its 50 task batch. Without the reset,
    bugs in candidate-local persistence (observed in run9 where r8 candidates
    showed success_count=29844 carried over from canonical) cascade into
    canonical via sync's delta = cand - round_start, producing absurd values
    (run9 final max success_count=87096 = 3 × 29844). Confidence /
    decay_weight / is_active are NOT reset — those carry pool-level history.
    """
    state = _load_canonical_state(run_dir)
    raw_units = list(state["units"])
    if not raw_units:
        return 0

    try:
        units = [MemoryUnit.from_dict(u) for u in raw_units if isinstance(u, dict) and u.get("type")]
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize canonical pool: {e}") from e

    # Zero candidate-local counters so sync sees genuine per-candidate deltas.
    for u in units:
        u.access_count = 0
        u.usage_count = 0
        u.success_count = 0
        u.conflict_count = 0

    store_dirs: Dict[str, str] = {
        runtime_config.primary_storage_type: runtime_config.storage_dir
    }
    store_dirs.update(runtime_config.additional_stores)

    enabled_types = set(runtime_config.extract_plan.get("extract_types", []))
    routing = runtime_config.extract_plan.get("storage_routing", {})
    units_by_store: Dict[str, List[MemoryUnit]] = {}
    for unit in units:
        type_key = unit.type.value
        if enabled_types and type_key not in enabled_types:
            continue
        store_type = routing.get(type_key, runtime_config.primary_storage_type)
        if store_type not in store_dirs:
            logger.warning(
                "Canonical import fallback: no compiled store for type=%s route=%s; using primary=%s",
                type_key,
                store_type,
                runtime_config.primary_storage_type,
            )
            store_type = runtime_config.primary_storage_type
        units_by_store.setdefault(store_type, []).append(unit)

    total_new = 0
    for stype, store_units in units_by_store.items():
        if not store_units:
            continue
        spath = Path(store_dirs[stype])
        try:
            store = _make_storage_backend(stype, spath)
            added = store.add(store_units)
            import_edges = getattr(store, "import_relation_edges", None)
            if callable(import_edges) and state["graph_edges"]:
                imported_edges = import_edges(state["graph_edges"])
                logger.info(
                    "Imported %d canonical relation edge update(s) into %s store.",
                    imported_edges,
                    stype,
                )
            total_new += added
            logger.info(
                "Imported %d/%d canonical units into %s store at %s (total=%d)",
                added,
                len(store_units),
                stype,
                spath,
                store.count(),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to import canonical units into {stype} at {spath}: {e}"
            ) from e

    return total_new


# ======================================================================
# No-memory baseline
# ======================================================================

def run_baseline(run_dir: Path, split: DataSplitConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """Run no-memory evaluation on warmup + search + validation tasks.

    Covers all splits except test so that per-task memory_lift can be
    computed for any batch sampled during the search phase.

    Returns a baseline stats dict including per-task scores.
    """
    baseline_dir = run_dir / "baseline"
    done_marker = baseline_dir / "baseline_done.json"

    if done_marker.exists():
        logger.info("Baseline already done; loading from %s", done_marker)
        with open(done_marker, "r", encoding="utf-8") as f:
            return json.load(f)

    baseline_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = baseline_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    # Cover warmup + search + validation (all except test) so per-task
    # baseline scores are available for every task the search may sample.
    baseline_indices = sorted(set(
        list(split.profile_indices)
        + list(split.optimization_indices)
        + list(split.validation_indices)
    ))

    # Cross-run baseline reuse (2026-07-07): if --baseline_from covers the same
    # task indices, copy it in instead of re-running (~1h + tokens saved).
    # Reuse is fail-closed on exact indices, resolved model, and the dedicated
    # no-memory protocol digest (runner/source/input/endpoints/environment).
    _bfrom = getattr(args, "baseline_from", None)
    if _bfrom and not args.dry_run:
        src_marker = Path(_bfrom)
        if src_marker.is_dir():
            src_marker = src_marker / "baseline_done.json"
        if src_marker.exists():
            try:
                prior = json.load(open(src_marker, encoding="utf-8"))
                current_model = (
                    args.model or os.environ.get("DEFAULT_MODEL") or "gpt-5"
                )
                if prior.get("backbone_model") != current_model:
                    raise ValueError(
                        "backbone model mismatch: "
                        f"saved={prior.get('backbone_model')!r}, "
                        f"current={current_model!r}"
                    )
                current_baseline_digest = getattr(
                    args, "_baseline_protocol_digest", None
                )
                if (
                    not current_baseline_digest
                    or prior.get("baseline_protocol_digest")
                    != current_baseline_digest
                ):
                    raise ValueError(
                        "baseline protocol digest is missing or does not match"
                    )
                prior_idx = set(prior.get("baseline_indices") or [])
                if not prior_idx:
                    prior_idx = {int(k) for k in (prior.get("per_task_scores") or {})
                                 if str(k).lstrip("-").isdigit()}
                if prior_idx and set(baseline_indices) == prior_idx:
                    import shutil
                    src_tasks = src_marker.parent / "tasks"
                    _require_exact_task_results(
                        src_tasks,
                        baseline_indices,
                        "Source no-memory baseline",
                        getattr(args, "_task_dataset_sha256", None),
                    )
                    shutil.copytree(src_tasks, tasks_dir, dirs_exist_ok=True)
                    shutil.copy(src_marker, done_marker)
                    logger.info(
                        "[baseline_reuse] reused baseline from %s (covers %d/%d needed "
                        "indices; prior backbone=%s)", src_marker, len(prior_idx),
                        len(baseline_indices), prior.get("backbone_model", "?"))
                    return prior
                logger.warning(
                    "[baseline_reuse] %s does not cover all needed indices "
                    "(%d needed, %d available); running fresh baseline.",
                    src_marker, len(baseline_indices), len(prior_idx))
            except Exception as e:
                logger.warning("[baseline_reuse] failed to reuse %s (%s); running fresh baseline.",
                               src_marker, e)
        else:
            logger.warning("[baseline_reuse] no baseline_done.json at %s; running fresh baseline.", _bfrom)

    if not args.dry_run:
        # Codex CR2-4: never write a partial baseline checkpoint. If the
        # subprocess exits non-zero or produces fewer task_results than
        # expected, raise so resume cannot proceed against a corrupted
        # baseline_per_task lookup.
        ok = _run_eval_subprocess(
            tasks_dir=tasks_dir,
            task_indices=baseline_indices,
            runtime_config=None,  # no memory
            args=args,
            extra_flags=[],  # no memory_provider flag = no memory
        )
        if not ok:
            raise RuntimeError(
                f"Baseline subprocess failed (see {tasks_dir.parent / 'run.log'}). "
                "Refusing to write baseline_done.json with partial data — "
                "delete the baseline/ directory and re-run from scratch."
            )
        _require_exact_task_results(
            tasks_dir,
            baseline_indices,
            "No-memory baseline",
            getattr(args, "_task_dataset_sha256", None),
        )

    # Dry-run is a deterministic control path: it never needs runner output.
    if args.dry_run:
        task_results = [
            {
                "task_id": f"dry-{index}",
                "item_index": index + 1,
                "task_score": float((index * 17 + 3) % 5 != 0),
            }
            for index in baseline_indices
        ]
    else:
        task_results = load_task_results(str(tasks_dir))
    # The internal one-based item index is the protocol identity. External
    # task ids are display metadata and must not collapse two scored rows.
    task_scores = {
        str(r["item_index"]): float(r.get("task_score", 0.0))
        for r in task_results
    }

    # Build per-task score map keyed by item_index (for batch-specific lift)
    per_task_scores = {}
    for r in task_results:
        idx = str(r.get("item_index", ""))
        if idx:
            per_task_scores[idx] = float(r.get("task_score", 0.0))

    # Classify tasks
    easy = [tid for tid, s in task_scores.items() if s >= 1.0]
    memory_sensitive = [tid for tid, s in task_scores.items() if s < 1.0]

    total = len(task_scores)
    baseline_accuracy = sum(task_scores.values()) / total if total > 0 else 0.0

    stats = {
        "baseline_accuracy": round(baseline_accuracy, 4),
        "total_tasks": total,
        # Recorded so a later run can validate cross-run reuse (--baseline_from).
        "baseline_indices": list(baseline_indices),
        "backbone_model": args.model or os.environ.get("DEFAULT_MODEL") or "gpt-5",
        "baseline_protocol_digest": getattr(
            args, "_baseline_protocol_digest", None
        ),
        "easy_count": len(easy),
        "memory_sensitive_count": len(memory_sensitive),
        "easy_ids": easy[:20],  # cap for context size
        "memory_sensitive_sample": memory_sensitive[:20],
        "per_task_scores": per_task_scores,
        "note": (
            f"{len(memory_sensitive)}/{total} tasks failed without memory "
            f"(baseline acc={baseline_accuracy:.3f})"
        ),
    }

    _atomic_write_json(done_marker, stats)
    logger.info("Baseline complete: acc=%.3f (%d easy, %d memory-sensitive)",
                baseline_accuracy, len(easy), len(memory_sensitive))
    return stats


# ======================================================================
# Warm-up phase
# ======================================================================

def run_warmup(run_dir: Path, split: DataSplitConfig, args: argparse.Namespace) -> None:
    """Run initial architecture on warmup tasks to seed the canonical pool."""
    warmup_dir = run_dir / "warmup"
    done_marker = warmup_dir / "warmup_done.json"

    if args.warmup_n <= 0:
        logger.info("Warmup skipped (warmup_n=%d).", args.warmup_n)
        return

    tasks_dir = warmup_dir / "tasks"
    if done_marker.exists() and args.dry_run:
        logger.info("Synthetic warmup already done; skipping.")
        return

    if done_marker.exists():
        try:
            if done_marker.is_symlink() or not done_marker.is_file():
                raise ValueError("warmup marker must be a regular file")
            marker = json.loads(done_marker.read_text(encoding="utf-8"))
            canonical_state = _load_canonical_state(run_dir)
            if (
                not isinstance(marker, dict)
                or type(marker.get("n_tasks")) is not int
                or marker["n_tasks"] != len(split.profile_indices)
                or type(marker.get("pool_size")) is not int
                or marker["pool_size"] < 0
                or "warmup" not in {
                    str(value) for value in canonical_state["applied_merges"]
                }
            ):
                raise ValueError("warmup marker does not match persisted state")
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning(
                "Warmup marker is not reusable (%s); rebuilding its checkpoint.",
                exc,
            )
            if done_marker.is_dir() and not done_marker.is_symlink():
                import shutil as _shutil

                _shutil.rmtree(done_marker)
            else:
                done_marker.unlink(missing_ok=True)

    reuse_completed = False
    if not args.dry_run:
        reuse_completed = _reuse_or_reset_stateful_stage(
            tasks_dir=tasks_dir,
            expected_indices=list(split.profile_indices),
            reset_paths=[warmup_dir],
            stage="Warmup",
            required_state_paths=[warmup_dir / "storage"],
            expected_dataset_sha256=getattr(args, "_task_dataset_sha256", None),
        )
        if done_marker.exists() and reuse_completed:
            logger.info("Warmup already done with exact tasks and storage; skipping.")
            return

    warmup_dir.mkdir(parents=True, exist_ok=True)
    storage_dir = str(warmup_dir / "storage")

    runtime_config = _compile_architecture(WARMUP_ARCHITECTURE, storage_dir)
    if runtime_config is None:
        logger.error("Warmup architecture failed to compile; skipping warmup.")
        return

    tasks_dir.mkdir(exist_ok=True)

    if not args.dry_run and not reuse_completed:
        # Codex Q14-5 fix (2026-04-28): refuse to write the warmup
        # done-marker if the eval subprocess failed or produced fewer
        # task results than requested. A partial warmup with
        # warmup_done.json present causes every later --resume to skip
        # warmup with an empty/partial canonical pool, silently
        # changing every candidate's retrieval and lift downstream.
        ok = _run_eval_subprocess(
            tasks_dir=tasks_dir, task_indices=split.profile_indices,
            runtime_config=runtime_config, args=args,
        )
        if not ok:
            raise RuntimeError(
                f"Warmup subprocess failed (see {warmup_dir / 'run.log'}). "
                "Refusing to write warmup_done.json with partial data — "
                "delete the warmup/ directory and re-run."
            )
        _require_exact_task_results(
            tasks_dir,
            list(split.profile_indices),
            "Warmup",
            getattr(args, "_task_dataset_sha256", None),
        )
    elif not args.dry_run:
        _require_exact_task_results(
            tasks_dir,
            list(split.profile_indices),
            "Warmup",
            getattr(args, "_task_dataset_sha256", None),
        )

    sync_tasks_to_canonical(run_dir, warmup_dir, merge_id="warmup")

    pool = load_canonical_pool(run_dir)
    _atomic_write_json(
        done_marker,
        {"n_tasks": len(split.profile_indices), "pool_size": len(pool)},
    )
    logger.info("Warmup complete: %d tasks, %d units in pool", len(split.profile_indices), len(pool))


# ======================================================================
# Architecture compilation helper
# ======================================================================

def _compile_architecture(arch: Dict[str, Any], storage_dir: str) -> Optional[RuntimeConfig]:
    """Compile one strict public architecture to a runtime configuration."""
    compiler = ArchitectureCompiler(base_storage_dir=storage_dir)
    try:
        spec = ArchitectureSpec.from_search_dict(arch)
        return compiler.compile_spec(spec)
    except Exception as e:
        logger.error("Architecture compilation failed: %s | arch=%s", e, arch)
        return None


# ======================================================================
# Eval subprocess
# ======================================================================

def _indices_to_str(indices: List[int]) -> str:
    # eval script uses 1-based indexing (data[i-1]), split produces 0-based → add 1
    return ",".join(str(i + 1) for i in sorted(indices))


def _run_eval_subprocess(
    tasks_dir: Path,
    task_indices: List[int],
    runtime_config: Optional[RuntimeConfig],
    args: argparse.Namespace,
    extra_flags: Optional[List[str]] = None,
) -> bool:
    """Run the eval script as a subprocess. Returns True on success."""
    proc = _start_eval_subprocess(tasks_dir, task_indices, runtime_config, args, extra_flags)
    if proc is None:
        return True  # no tasks to run
    proc.wait()
    if hasattr(proc, "_log_file"):
        proc._log_file.close()
    if proc.returncode != 0:
        log_path = tasks_dir.parent / "run.log"
        logger.error("Eval subprocess failed (exit %d). See %s", proc.returncode, log_path)
        return False
    logger.info("Eval subprocess done.")
    return True


def _scan_task_result_indices(
    tasks_dir: Path,
    expected_dataset_sha256: Optional[str] = None,
) -> Tuple[set[int], List[int], List[str]]:
    """Read completed runner outputs and normalize their indices to zero-based."""
    counts: Dict[int, int] = {}
    invalid_files: List[str] = []
    for result_path in sorted(tasks_dir.glob("*.json")):
        if result_path.name == "extract_plan.json":
            continue
        try:
            if result_path.is_symlink() or not result_path.is_file():
                raise ValueError("task result must be a regular non-symlink file")
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            validation_error = task_result_validation_error(payload)
            if validation_error is not None:
                raise ValueError(validation_error)
            item_index = payload["item_index"]
            if result_path.name != f"{item_index}.json":
                raise ValueError("task result filename does not match item_index")
            if (
                expected_dataset_sha256
                and payload["task_identity"]
                != task_identity_digest(expected_dataset_sha256, item_index)
            ):
                raise ValueError("task result identity does not match current dataset")
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            invalid_files.append(result_path.name)
            continue
        normalized = item_index - 1
        counts[normalized] = counts.get(normalized, 0) + 1

    duplicates = sorted(index for index, count in counts.items() if count > 1)
    return set(counts), duplicates, invalid_files


def _require_exact_task_results(
    tasks_dir: Path,
    expected_indices: List[int],
    stage: str,
    expected_dataset_sha256: Optional[str] = None,
) -> None:
    """Fail closed unless runner outputs exactly cover the requested task set."""
    expected = set(expected_indices)
    if len(expected) != len(expected_indices):
        raise RuntimeError(f"{stage}: requested task indices contain duplicates")

    present, duplicates, invalid_files = _scan_task_result_indices(
        tasks_dir, expected_dataset_sha256
    )
    missing = sorted(expected - present)
    extras = sorted(present - expected)
    if missing or extras or duplicates or invalid_files:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extras:
            details.append(f"extras={extras}")
        if duplicates:
            details.append(f"duplicates={duplicates}")
        if invalid_files:
            details.append(f"invalid_files={invalid_files}")
        raise RuntimeError(
            f"{stage}: incomplete or mismatched task results ({'; '.join(details)})"
        )


def _reuse_or_reset_stateful_stage(
    tasks_dir: Path,
    expected_indices: List[int],
    reset_paths: List[Path],
    stage: str,
    required_state_paths: Optional[List[Path]] = None,
    expected_dataset_sha256: Optional[str] = None,
) -> bool:
    """Reuse an exact stage or remove every causally stateful partial artifact."""

    def _has_state(path: Path) -> bool:
        if path.is_symlink() or path.is_file():
            return True
        return path.is_dir() and any(path.iterdir())

    if not any(_has_state(path) for path in reset_paths):
        return False

    present, duplicates, invalid_files = _scan_task_result_indices(
        tasks_dir, expected_dataset_sha256
    )
    expected = set(expected_indices)
    if (
        len(expected) == len(expected_indices)
        and present == expected
        and not duplicates
        and not invalid_files
        and all(_has_state(path) for path in (required_state_paths or []))
    ):
        logger.info("%s: reusing an exact completed task set.", stage)
        return True

    import shutil as _shutil

    logger.warning(
        "%s: discarding partial state and replaying the full batch "
        "(present=%s expected=%s duplicates=%s invalid=%s).",
        stage,
        sorted(present),
        sorted(expected),
        duplicates,
        invalid_files,
    )
    for path in reset_paths:
        if path.is_dir() and not path.is_symlink():
            _shutil.rmtree(path)
        elif path.exists() or path.is_symlink():
            path.unlink()
    return False


def _start_eval_subprocess(
    tasks_dir: Path,
    task_indices: List[int],
    runtime_config: Optional[RuntimeConfig],
    args: argparse.Namespace,
    extra_flags: Optional[List[str]] = None,
) -> Optional[subprocess.Popen]:
    """Start the eval script as a non-blocking subprocess. Returns Popen or None."""
    if not task_indices:
        logger.warning("No task indices; skipping eval.")
        return None

    infile_path = Path(args.infile).expanduser()
    if not infile_path.is_file():
        raise FileNotFoundError(f"Benchmark task file not found: {infile_path}")
    infile = str(infile_path)
    results_file = tasks_dir.parent / "results.jsonl"

    eval_script = getattr(args, "eval_script", None)
    if eval_script:
        eval_path = Path(eval_script).expanduser()
        if not eval_path.is_file():
            raise FileNotFoundError(f"Custom eval runner not found: {eval_path}")
        launcher = [sys.executable, str(eval_path)]
    else:
        benchmark_key = str(getattr(args, "benchmark", "gaia")).strip().lower()
        runner_module = BENCHMARK_RUNNER_MODULES.get(benchmark_key)
        if runner_module is None:
            supported = ", ".join(sorted(BENCHMARK_RUNNER_MODULES))
            raise ValueError(
                f"No installed runner for benchmark '{args.benchmark}'. "
                f"Use one of [{supported}] or pass --eval_script."
            )
        launcher = [sys.executable, "-m", runner_module]

    cmd = [*launcher,
           "--infile", infile,
           "--outfile", str(results_file),
           "--task_indices", _indices_to_str(task_indices),
           "--max_steps", str(args.max_steps),
           "--token_budget", str(args.token_budget),
           "--concurrency", str(args.concurrency),
           "--direct_output_dir", str(tasks_dir)]

    if args.model:
        cmd.extend(["--model", args.model])
    if getattr(args, "judge_model", None):
        cmd.extend(["--judge_model", args.judge_model])

    if extra_flags:
        cmd.extend(extra_flags)
    elif runtime_config is not None:
        runtime_config_path = tasks_dir.parent / "runtime_config.json"
        runtime_config_path.write_text(
            json.dumps(runtime_config.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        cmd.extend([
            "--memory_provider", "modular",
            "--shared_memory_provider",
            "--enable_memory_evolution",
            "--runtime_config_json", str(runtime_config_path),
        ])
    # With no runtime config the runner's default ``None`` means no memory.

    log_path = tasks_dir.parent / "run.log"
    logger.info("Eval: %d tasks -> %s", len(task_indices), tasks_dir)

    lf = open(log_path, "w")
    try:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        proc._log_file = lf  # attach so caller can close after wait
        return proc
    except Exception as e:
        lf.close()
        logger.error("Eval subprocess start error: %s", e)
        raise


# ======================================================================
# Fitness computation
# ======================================================================

def _compute_per_task_mor(
    task_results: List[Dict[str, Any]],
    pool_units: List[Dict[str, Any]],
    relevance_threshold: float = 0.05,
) -> Dict[str, int]:
    """Per-task Memory Opportunity Rate (MOR) via TF-IDF oracle approximation.

    A task has MOR=1 if at least one pool unit's source_task_query / content
    has TF-IDF similarity >= threshold to the task question. This is an
    automatic, label-free proxy for "could the memory pool plausibly have
    helped this task" — used to compute Conditional Lift on the MOR=1 subset
    so reviewers cannot attack overall lift as a side-effect of task overlap.

    Returns a dict task_id -> 0/1.
    """
    from automem.search.attribution import _tokenize, _build_idf, _tf_idf_score

    if not pool_units:
        # F5 fix (codex review, 2026-05-16): use _stable_task_id so empty-pool
        # path keys match the reader (E2 fix).
        return {_stable_task_id(r, i): 0
                for i, r in enumerate(task_results)}

    # Codex Round 3 R3-8: replace blanket _flatten_strings with a curated
    # set of high-signal content fields. The previous flatten included
    # every step `action` ("web_search", "crawl_page", "open_url"), which
    # are tokens that appear in almost any GAIA task and pushed many tasks
    # to MOR=1 spuriously. Conditional Lift then collapsed toward overall
    # lift, defeating the whole point of the metric.
    _HIGH_SIGNAL_FIELDS = (
        # Tip
        "topic", "principle", "applicability",
        # Insight
        "root_cause_conclusion", "corrective_strategy", "detection_signal",
        # Trajectory
        "key_decision", "critical_observation", "reusable_anchor",
        # Workflow
        "chain_type", "final_format_check",
        # Shortcut
        "name", "description", "precondition",
    )

    def _extract_high_signal_text(content: Any, out: List[str]) -> None:
        """Pull only curated fields from content + nested workflow.steps.rationale."""
        if not isinstance(content, dict):
            return
        for k in _HIGH_SIGNAL_FIELDS:
            v = content.get(k)
            if isinstance(v, str) and v.strip():
                out.append(v)
        # Workflow rationales (one level of nesting, deliberately limited).
        for wf_key in ("agent_workflow", "search_workflow"):
            for s in content.get(wf_key, []) or []:
                if isinstance(s, dict):
                    for sk in ("rationale", "validation_criteria",
                               "query_formulation"):
                        v = s.get(sk)
                        if isinstance(v, str) and v.strip():
                            out.append(v)

    pool_texts = []
    for u in pool_units:
        parts: List[str] = []
        sq = u.get("source_task_query", "")
        if isinstance(sq, str):
            parts.append(sq)
        for k in ("use_when", "applicable_task_types"):
            v = u.get(k, [])
            if isinstance(v, list):
                parts.extend(str(x) for x in v if isinstance(x, str))
        _extract_high_signal_text(u.get("content", {}), parts)
        pool_texts.append(" ".join(parts))

    pool_tokens = [_tokenize(t) for t in pool_texts]
    idf = _build_idf(pool_tokens) if pool_tokens else {}

    out: Dict[str, int] = {}
    for i, r in enumerate(task_results):
        tid = _stable_task_id(r, i)         # E2 fix (2026-05-16)
        question = r.get("question", r.get("Question", "")) or ""
        q_tokens = _tokenize(question)
        if not q_tokens or not pool_tokens:
            out[tid] = 0
            continue
        max_sim = max(_tf_idf_score(q_tokens, pt, idf) for pt in pool_tokens)
        out[tid] = 1 if max_sim >= relevance_threshold else 0
    return out


def _stable_task_id(r: Dict[str, Any], i: int) -> str:
    """Derive a stable, unique task identifier.

    E2 fix (2026-05-16): writer/reader sites used different defaults
    (positional ``i`` vs ``""``) when a result was missing both ``task_id``
    and ``id``, silently dropping such tasks from MOR conditional lift.
    F6 fix (codex review): use ``is None`` instead of ``or`` chain so that
    falsy-but-present ids (integer ``0``, empty string) are preserved
    rather than treated as missing.
    """
    tid = r.get("task_id")
    if tid is None:
        tid = r.get("id")
    if tid is None:
        return f"_pos_{i}"
    return str(tid)


def _build_per_task_pass_fail_summary(
    tasks_dir: Path,
    baseline_per_task: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Build a compact per-task pass/fail list for validation checkpoints.

    Reads each tasks_dir/*.json and returns a list of dicts:
        {task_id, item_index, level, category, score, baseline_score,
         lift, golden_answer, agent_answer, status}

    Used by validation_checkpoint and final_validation so the user can
    eyeball "which exact tasks passed/failed" without grepping per-task
    JSONs. Per-task category uses the GAIA taxonomy classifier (see
    automem/task_taxonomy.py).
    """
    from automem.task_taxonomy import classify_gaia_task

    results = load_task_results(str(tasks_dir))
    summary: List[Dict[str, Any]] = []
    for r in results:
        score = float(r.get("task_score", 0.0) or 0.0)
        tid = str(r.get("task_id", ""))
        baseline_score = (baseline_per_task or {}).get(
            str(r.get("item_index", ""))
        )
        if baseline_score is None:
            baseline_score = (baseline_per_task or {}).get(tid)
        # Classify task by GAIA taxonomy. The result JSON keeps the
        # original Question + file_name in metadata, but result_aggregator
        # may not have preserved them — fall back to question + file_name
        # if present at top level.
        task_proxy = {
            "Question": r.get("question", ""),
            "file_name": r.get("file_name", "") or "",
        }
        category = classify_gaia_task(task_proxy)
        summary.append({
            "task_id": tid,
            "item_index": r.get("item_index"),
            "level": r.get("level"),
            "category": category,
            "score": score,
            "passed": score >= 1.0,
            "baseline_score": baseline_score,
            "lift": (
                round(score - baseline_score, 4)
                if baseline_score is not None else None
            ),
            "status": r.get("status", "unknown"),
            "question": (r.get("question") or "")[:200],
            "golden_answer": str(r.get("golden_answer", ""))[:200],
            "agent_answer": str(r.get("agent_result", ""))[:200],
        })
    # Stable sort: failures first (so user sees what's wrong at the top),
    # then by item_index for reproducibility.
    summary.sort(key=lambda x: (x["passed"], x["item_index"] or 0))
    return summary


class FitnessComputationError(Exception):
    """Raised by compute_candidate_fitness when fitness cannot be computed.

    E7 fix (2026-05-16): distinguishes 'evaluation failed' from 'agent got 0'.
    Caller catches and routes the candidate to failed_config_ids — never adds
    a zeroed entry to the Pareto front, where it would be indistinguishable
    from a legitimately failing-but-evaluated architecture.
    """

    def __init__(self, reason: str, details: str = ""):
        self.reason = reason
        self.details = details
        super().__init__(f"FitnessComputationError({reason}): {details}")


def _synthetic_candidate_metrics(
    architecture: Dict[str, Any],
    *,
    round_id: int,
    candidate_id: int,
    total_tasks: int,
    baseline_accuracy: float,
) -> Tuple[float, Dict[str, Any]]:
    """Return stable dry-run metrics without touching runners or task files."""
    import hashlib

    payload = json.dumps(
        {
            "architecture": architecture,
            "round_id": round_id,
            "candidate_id": candidate_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")

    def _value(offset: int, low: float, high: float) -> float:
        fraction = ((seed >> offset) & 0xFFFF) / 0xFFFF
        return round(low + (high - low) * fraction, 4)

    accuracy = _value(0, 0.50, 0.90)
    hit_rate = _value(16, 0.30, 0.90)
    token_eff = _value(32, 0.70, 1.00)
    memory_lift = round(accuracy - baseline_accuracy, 4)
    fitness = round(0.70 * accuracy + 0.20 * hit_rate + 0.10 * token_eff, 4)
    return fitness, {
        "accuracy": accuracy,
        "memory_lift": memory_lift,
        "hit_rate": hit_rate,
        "token_eff": token_eff,
        "fitness": fitness,
        "total_tasks": total_tasks,
        "synthetic": True,
    }



def compute_candidate_fitness(
    tasks_dir: Path,
    baseline_accuracy: float,
    baseline_per_task: Optional[Dict[str, float]] = None,
    weights: Optional[FitnessWeights] = None,
    round_id: int = 0,
    architecture: Optional[Dict[str, Any]] = None,
    token_cap_per_task: int = 30000,
    latency_cap_per_task: float = 60.0,
    pool_units: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute fitness for a candidate evaluation.

    When *baseline_per_task* is provided (item_index → score), memory_lift
    is computed against the same-task baseline instead of the aggregate.

    When *pool_units* is provided, also computes MOR (Memory Opportunity
    Rate) per task and Conditional Lift restricted to MOR=1 subset. This
    is the published metric reviewers will compare against — Overall Lift
    can be inflated by tasks with no chance of memory help, while
    Conditional Lift is robust to that confound.

    Returns (fitness_score, raw_metrics_dict).
    """
    task_results = load_task_results(str(tasks_dir))
    if not task_results:
        # E7 fix (2026-05-16): raise instead of returning (0.0, {}) which the
        # Pareto front cannot distinguish from a real zero-fitness candidate.
        # Caller adds the config to failed_config_ids (same path as subprocess
        # exit-nonzero) — guaranteeing no zeroed fitness pollutes the front.
        raise FitnessComputationError(
            reason="no_task_results",
            details=f"tasks_dir={tasks_dir} contained 0 task JSONs",
        )

    try:
        report = build_evaluation_report(
            tasks_dir=str(tasks_dir),
            architecture_decision_dict=architecture or {},
            round_id=round_id,
        )
    except Exception as e:
        # E7 fix (2026-05-16): same as above — raise rather than zero-out.
        raise FitnessComputationError(
            reason="build_report_failed",
            details=str(e),
        ) from e

    if weights is None:
        # C1 fix (2026-05-16): latency_penalty / token_cost_penalty previously
        # saturated to 1.0 on every candidate (avg_time ≫ 60s cap, total_tokens
        # ≫ 1.5M cap), making them constant offsets with zero effect on Pareto
        # ranking. Setting them to 0 here keeps fitness clearly accuracy-first
        # while empty_retrieval still differentiates architectures. Callers
        # wanting a relative cost penalty should pass a custom FitnessWeights
        # with weights anchored to the per-round baseline candidate's stats.
        weights = FitnessWeights(
            accuracy=1.0,
            latency_penalty=0.0,
            token_cost_penalty=0.0,
            empty_retrieval_penalty=0.05,
            unused_edge_penalty=0.0,
        )

    # Scale token_cap with batch size so norm_tokens does not saturate at 1.0
    # (previous global cap of 100k was exceeded on every batch, forcing
    # token_eff=0 across all 24 evaluated architectures in automem_v2).
    n_tasks = len(task_results) or 1
    batch_token_cap = float(token_cap_per_task * n_tasks)

    fitness_result = compute_fitness(
        report,
        weights,
        latency_cap=latency_cap_per_task,
        token_cap=batch_token_cap,
    )

    # Per-task memory_lift: compare against baseline scores for the same tasks
    if baseline_per_task:
        batch_baselines = []
        for r in task_results:
            idx = str(r.get("item_index", ""))
            if idx in baseline_per_task:
                batch_baselines.append(baseline_per_task[idx])
        batch_baseline_acc = (
            sum(batch_baselines) / len(batch_baselines)
        ) if batch_baselines else baseline_accuracy
    else:
        batch_baseline_acc = baseline_accuracy

    memory_lift = fitness_result.accuracy - batch_baseline_acc
    hit_rate = 1.0 - fitness_result.empty_retrieval_rate
    token_eff = 1.0 - fitness_result.normalized_token_cost

    raw_metrics = {
        "accuracy": fitness_result.accuracy,
        "memory_lift": memory_lift,
        "batch_baseline_accuracy": batch_baseline_acc,
        "hit_rate": hit_rate,
        "token_eff": token_eff,
        "fitness": fitness_result.fitness,
        "empty_retrieval_rate": fitness_result.empty_retrieval_rate,
        "total_tasks": len(task_results),
        "score_summary": report.score_summary,
    }

    # ------------------------------------------------------------------
    # Conditional Lift on MOR=1 subset (added 2026-04-28).
    # MOR (Memory Opportunity Rate) = fraction of tasks that *could*
    # plausibly benefit from the current pool, judged by TF-IDF similarity
    # between task question and pool unit source_task_query / content /
    # use_when triggers. Conditional Lift = lift restricted to that subset.
    # When pool_units is None or all tasks have MOR=0, conditional metrics
    # are reported as None so downstream code can decide how to display.
    # ------------------------------------------------------------------
    if pool_units is not None:
        mor_map = _compute_per_task_mor(task_results, pool_units)
        mor_n = sum(mor_map.values())
        mor_rate = mor_n / max(len(task_results), 1)
        if mor_n > 0:
            mor_acc_sum = 0.0
            mor_baseline_sum = 0.0
            counted = 0
            for i, r in enumerate(task_results):
                tid = _stable_task_id(r, i)        # E2 fix: same helper as writer
                if mor_map.get(tid, 0) != 1:
                    continue
                ts = float(r.get("task_score", 0.0))
                idx = str(r.get("item_index", ""))
                bs = (
                    baseline_per_task.get(idx, batch_baseline_acc)
                    if baseline_per_task else batch_baseline_acc
                )
                mor_acc_sum += ts
                mor_baseline_sum += bs
                counted += 1
            if counted > 0:
                mor_accuracy = mor_acc_sum / counted
                mor_baseline_acc = mor_baseline_sum / counted
                conditional_lift = mor_accuracy - mor_baseline_acc
            else:
                mor_accuracy = None
                mor_baseline_acc = None
                conditional_lift = None
        else:
            mor_accuracy = None
            mor_baseline_acc = None
            conditional_lift = None

        raw_metrics["mor_rate"] = round(mor_rate, 4)
        raw_metrics["mor_count"] = mor_n
        raw_metrics["conditional_lift"] = (
            round(conditional_lift, 4) if conditional_lift is not None else None
        )
        raw_metrics["mor_accuracy"] = (
            round(mor_accuracy, 4) if mor_accuracy is not None else None
        )
        raw_metrics["mor_baseline_accuracy"] = (
            round(mor_baseline_acc, 4) if mor_baseline_acc is not None else None
        )

    return fitness_result.fitness, raw_metrics


# ======================================================================
# LLM candidate generation
# ======================================================================

def generate_candidates(
    model,
    round_id: int,
    num_candidates: int,
    pareto: ParetoFront,
    pool_units: List[Dict],
    baseline_stats: Dict,
    last_round_details: Dict,
    cumulative_principles: str,
    prompt_path: str,
    max_rounds: int = 8,
    exploration_hints: str = "",
    obs_graph_json: str = "",
    benchmark_name: str = "GAIA",
) -> List[Dict[str, Any]]:
    """Call the search LLM to generate candidate architectures.

    obs_graph_json: optional serialized Observation Graph. When non-empty it
    is rendered into the prompt as additional structural context (the
    ``{% if obs_graph_json %}`` block in architecture_search.txt). Empty
    string => the prompt is byte-for-byte identical to the no-graph baseline.
    """
    pareto_ctx = pareto.to_llm_context(max_front=5, max_history=15)

    # Round-aware: starting from round 2, the runner auto-injects the current
    # Pareto best as candidate 0 ("champion") for elitism. The architect LLM
    # must produce only `num_candidates - 1` candidates in those rounds, with
    # candidate_id starting at 1.
    has_champion = (round_id > 1) and (pareto.best() is not None)
    num_candidates_for_llm = num_candidates - 1 if has_champion else num_candidates

    # Progress fraction drives the phase template (broad / mixed / pure).
    progress = round_id / max(1, max_rounds)

    template_vars = {
        "round_id": round_id,
        "max_rounds": max_rounds,
        "benchmark_name": benchmark_name,
        "progress": progress,
        "progress_pct": int(progress * 100),
        "num_candidates": num_candidates,
        "num_candidates_for_llm": num_candidates_for_llm,
        "pareto_top_k": min(5, pareto.size()),
        "pool_stats_json": json.dumps(pool_stats(pool_units), indent=2),
        "baseline_stats_json": json.dumps(baseline_stats, indent=2),
        "pareto_front_json": json.dumps(pareto_ctx["pareto_front"], indent=2),
        "history_table_json": json.dumps(pareto_ctx["history_table"], indent=2),
        "history_len": len(pareto_ctx["history_table"]),
        "last_round_json": json.dumps(last_round_details, indent=2),
        "cumulative_principles": cumulative_principles or "None yet.",
        # Smart-5 fix (2026-05-16): diversity-aware exploration hints. Tells
        # the proposer which option values have been under-tested across the
        # 4 architecture dimensions.
        "exploration_hints": exploration_hints or "",
        # Observation Graph (optional). Empty string keeps the prompt identical
        # to the no-graph baseline (the template guards it with {% if %}).
        "obs_graph_json": obs_graph_json or "",
    }

    # I1 fix (codex review, 2026-05-17): inline §2E.1 / §2E.2 / §2E.3 JSON
    # data so the LLM sees the synthesis output directly rather than having
    # to dig into the raw last_round_json. Without these the §2E.1-§2E.3
    # sections are just descriptions with no actual data.
    _candidates = last_round_details.get("candidates", []) or []
    template_vars["synthesized_verdicts_json"] = json.dumps([
        {
            "config_id": c.get("config_id"),
            "diversity_role": c.get("diversity_role", ""),
            "added_to_front": c.get("added_to_front", False),
            **(c.get("synthesized_verdict") or {}),
        }
        for c in _candidates if c.get("synthesized_verdict")
    ], indent=2, ensure_ascii=False)
    template_vars["round_level_verdict_json"] = json.dumps(
        last_round_details.get("round_level_verdict", {}),
        indent=2, ensure_ascii=False,
    )
    template_vars["supporting_detail_json"] = json.dumps([
        {
            "config_id": c.get("config_id"),
            "accuracy": (c.get("metrics") or {}).get("accuracy"),
            "layer_diagnosis": c.get("layer_diagnosis", {}),
            "memory_compliance": c.get("memory_compliance", {}),
            "breakdown": c.get("breakdown", {}),
            "attribution_diagnosis": c.get("attribution_diagnosis", ""),
            "architecture": c.get("architecture", {}),
        }
        for c in _candidates
    ], indent=2, ensure_ascii=False)
    template_vars["differential_diagnosis_json"] = json.dumps(
        last_round_details.get("differential_diagnosis", {}),
        indent=2, ensure_ascii=False,
    )

    template_str = load_prompt(prompt_path)
    filled = render_prompt(template_str, template_vars)

    last_raw = ""
    for attempt in range(1, 4):
        try:
            messages = [{"role": "user", "content": [{"type": "text", "text": filled}]}]
            response = model(messages)
            last_raw = response.content if hasattr(response, "content") else str(response)
            parsed = parse_json_response(last_raw)
            if parsed is not None and isinstance(parsed, list):
                # Codex Q14-3 fix (2026-04-28): the architect LLM may
                # return a list with malformed entries (None, strings,
                # missing candidate_id). The downstream round loop
                # calls cand.get(...)/int(...) without re-validating
                # and crashes. Filter to dicts with int candidate_id;
                # if everything is dropped, force fallback by retrying.
                valid = []
                for i, entry in enumerate(parsed):
                    if not isinstance(entry, dict):
                        logger.warning(
                            "Architect entry %d is not a dict (got %s); dropping.",
                            i, type(entry).__name__,
                        )
                        continue
                    cid = entry.get("candidate_id")
                    if cid is None:
                        # Auto-assign so downstream code doesn't crash.
                        entry["candidate_id"] = i
                    else:
                        try:
                            entry["candidate_id"] = int(cid)
                        except (TypeError, ValueError):
                            logger.warning(
                                "Architect entry %d has non-int candidate_id=%r; dropping.",
                                i, cid,
                            )
                            continue
                    if not isinstance(entry.get("architecture"), dict):
                        logger.warning(
                            "Architect entry %d missing dict 'architecture'; dropping.",
                            i,
                        )
                        continue

                    # ArchitectureValidator gate (H-plan, 2026-05-13):
                    # validate strict RECOMMENDED subspace; auto-repair common
                    # LLM mistakes (deprecated values, missing routing); drop
                    # only if irreparably malformed.
                    from automem.search.validator import ArchitectureValidator
                    _validator = ArchitectureValidator(strict=True)
                    _vr = _validator.validate(entry["architecture"])
                    if not _vr.is_valid:
                        try:
                            entry["architecture"] = _validator.repair(entry["architecture"])
                            _vr2 = _validator.validate(entry["architecture"])
                            if not _vr2.is_valid:
                                logger.warning(
                                    "Architect entry %d (config_id=%s) failed validation "
                                    "even after repair; dropping. violations=%s",
                                    i, entry.get("candidate_id"), _vr2.violations,
                                )
                                continue
                            logger.info(
                                "Architect entry %d repaired into the canonical space.",
                                i,
                            )
                        except Exception as _e:
                            logger.warning(
                                "Architect entry %d repair raised %s; dropping.",
                                i, _e,
                            )
                            continue
                    valid.append(entry)
                if valid:
                    return valid
                logger.warning(
                    "Attempt %d: architect returned a list but no valid entries; retrying.",
                    attempt,
                )
            else:
                logger.warning(
                    "Attempt %d: LLM returned non-list JSON; retrying...", attempt
                )
        except Exception as e:
            logger.warning("Attempt %d: LLM call error: %s", attempt, e)
            time.sleep(5 * attempt)

    logger.error("Failed to get valid candidates from LLM after 3 attempts.")
    return []


def _validate_candidate(cand: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate a candidate through the canonical public architecture model."""
    arch = cand.get("architecture", {})
    if not isinstance(arch, dict):
        return False, "architecture must be a dict"
    from automem.search.validator import ArchitectureValidator

    report = ArchitectureValidator(strict=True).validate(arch)
    return report.is_valid, "; ".join(report.violations)


# ======================================================================
# Round / cumulative metrics tracking (added 2026-05-13)
# ======================================================================

def _parse_discovery_thresholds(s: str) -> List[float]:
    """Parse '0.55,0.60,0.65,0.70' → [0.55, 0.6, 0.65, 0.7]."""
    out = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(float(tok))
        except ValueError:
            logger.warning("Ignoring invalid discovery threshold: %r", tok)
    return sorted(out)


def _collect_round_task_census(
    round_dir: Path,
    baseline_stats: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Load per-task results from one candidate to feed the Observation Graph
    census (level/category distribution + per-level baseline accuracy).

    All candidates in a round run the SAME search batch, so the first
    candidate's ``tasks/*.json`` is representative of the round's task set.
    Returns (task_results, baseline_per_level). On any failure returns
    ([], {}) — the graph update degrades gracefully to candidate-edge-only.
    """
    # Pick the candidate dir with the MOST task JSONs — that is the candidate
    # that completed the full fixed batch. Picking the first candidate could
    # lock the census to a partial / crashed subprocess. Codex P2 fix.
    best_files: List[Path] = []
    for cand_dir in sorted(round_dir.glob("candidate_*")):
        tdir = cand_dir / "tasks"
        if not tdir.is_dir():
            continue
        files = [f for f in tdir.glob("*.json") if f.stem.isdigit()]
        if len(files) > len(best_files):
            best_files = files

    try:
        from automem.task_taxonomy import classify_gaia_task
    except Exception:
        classify_gaia_task = None

    task_results: List[Dict[str, Any]] = []
    for tf in best_files:
        try:
            d = json.loads(tf.read_text(encoding="utf-8"))
        except Exception:
            continue
        # GAIA result JSONs carry no `category` field — derive it from the
        # question / file_name the same way the rest of the pipeline does.
        category = d.get("category")
        if not category and classify_gaia_task is not None:
            try:
                category = classify_gaia_task({
                    "Question": d.get("question") or d.get("full_query") or "",
                    "file_name": d.get("file_name") or "",
                })
            except Exception:
                category = None
        _traj = d.get("agent_trajectory")
        task_results.append({
            "level": d.get("Level") or d.get("level"),
            "category": category or "unknown",
            "task_score": d.get("task_score"),
            "item_index": d.get("item_index"),
            # benchmark-agnostic complexity signal for the Observation Graph:
            # GAIA uses `level`; benchmarks without it fall back to step count.
            "n_steps": len(_traj) if isinstance(_traj, list) else None,
        })

    # Per-bucket baseline accuracy from baseline_stats.per_task_scores (1-based).
    # Key by the SAME complexity bucket the Observation Graph uses (via
    # task_complexity), so graph._update_task_patterns' baseline lookup matches.
    # Keying by raw level would leave baseline_acc null after the graph switched
    # to simple/medium/complex buckets. Codex P2 fix.
    from automem.task_complexity import task_complexity as _tc
    per_task = (baseline_stats or {}).get("per_task_scores") or {}
    lvl_total: Dict[str, int] = {}
    lvl_correct: Dict[str, int] = {}
    for t in task_results:
        _n = t.get("n_steps")
        lvl = _tc(explicit_level=t.get("level"),
                  trajectory=range(_n) if isinstance(_n, int) and _n > 0 else None)
        idx = t.get("item_index")
        if idx is None:
            continue
        bs = per_task.get(str(idx))
        if bs is None:
            continue
        lvl_total[lvl] = lvl_total.get(lvl, 0) + 1
        if float(bs) >= 1.0:
            lvl_correct[lvl] = lvl_correct.get(lvl, 0) + 1
    baseline_per_level = {
        lvl: lvl_correct.get(lvl, 0) / c
        for lvl, c in lvl_total.items() if c > 0
    }
    return task_results, baseline_per_level


def _compute_round_metrics(candidate_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Aggregate per-candidate metrics into round-level summary stats."""
    accs, fits = [], []
    for c in candidate_results:
        if c.get("skipped") or c.get("failed"):
            continue
        m = c.get("metrics") or {}
        if "accuracy" in m:
            try:
                accs.append(float(m["accuracy"]))
            except (TypeError, ValueError):
                pass
        if "fitness" in m:
            try:
                fits.append(float(m["fitness"]))
            except (TypeError, ValueError):
                pass
    if not accs:
        return {
            "round_best_accuracy": 0.0,
            "round_best_fitness": 0.0,
            "round_mean_accuracy": 0.0,
            "round_mean_fitness": 0.0,
            "round_top2_mean_accuracy": 0.0,
        }
    top2 = sorted(accs, reverse=True)[:2]
    return {
        "round_best_accuracy": max(accs),
        "round_best_fitness": max(fits) if fits else 0.0,
        "round_mean_accuracy": sum(accs) / len(accs),
        "round_mean_fitness": (sum(fits) / len(fits)) if fits else 0.0,
        "round_top2_mean_accuracy": sum(top2) / len(top2),
    }


def _update_cumulative_tracking(
    prev_state: Dict[str, Any],
    round_id: int,
    round_metrics: Dict[str, float],
    thresholds: List[float],
) -> Dict[str, Any]:
    """Update cumulative_tracking state with this round's results.

    Returns a fresh dict (suitable for both round_done and search_state).
    """
    prev = prev_state.get("cumulative_tracking") or {}
    cum_acc = max(float(prev.get("cumulative_max_accuracy", 0.0)),
                  float(round_metrics["round_best_accuracy"]))
    cum_fit = max(float(prev.get("cumulative_max_fitness", 0.0)),
                  float(round_metrics["round_best_fitness"]))

    rtd_prev = prev.get("rounds_to_discovery") or {}
    rtd: Dict[str, Optional[int]] = {}
    for thr in thresholds:
        key = f"{thr:.2f}"
        # If this threshold already has a recorded round, keep it (first
        # crossing wins). Otherwise check if we cross now.
        prior = rtd_prev.get(key)
        if prior is not None:
            rtd[key] = prior
        elif cum_acc >= thr:
            rtd[key] = round_id
        else:
            rtd[key] = None
    return {
        "cumulative_max_accuracy": cum_acc,
        "cumulative_max_fitness": cum_fit,
        "rounds_to_discovery": rtd,
    }


# ======================================================================
# Search round
# ======================================================================

def _candidate_checkpoint_digest(candidates: List[Dict[str, Any]]) -> str:
    import hashlib

    payload = json.dumps(
        candidates,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_bound_candidate_checkpoint(
    candidates_path: Path, manifest_path: Path
) -> List[Dict[str, Any]]:
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))
    if not isinstance(candidates, list) or not candidates or not all(
        isinstance(candidate, dict) for candidate in candidates
    ):
        raise ValueError("candidates.json must contain a non-empty list of objects")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != 1
        or manifest.get("sha256") != _candidate_checkpoint_digest(candidates)
    ):
        raise ValueError("candidate manifest digest mismatch")
    return candidates


def run_search_round(
    round_id: int,
    run_dir: Path,
    model,
    pareto: ParetoFront,
    split: DataSplitConfig,
    baseline_stats: Dict,
    last_round_details: Dict,
    cumulative_principles: str,
    args: argparse.Namespace,
    search_batch_indices: List[int],
    diagnosis_model=None,
    prev_cumulative_tracking: Optional[Dict[str, Any]] = None,
    exploration_hints: str = "",
    obs_graph_json: str = "",
) -> Tuple[ParetoFront, Dict]:
    """Execute one full search round.

    Returns (updated_pareto, round_summary).
    """
    round_dir = run_dir / f"round_{round_id}"
    round_dir.mkdir(parents=True, exist_ok=True)

    done_marker = round_dir / "round_done.json"
    if done_marker.exists() and args.resume:
        logger.info("Round %d already done; loading from checkpoint.", round_id)
        with open(done_marker, "r", encoding="utf-8") as f:
            round_summary = json.load(f)
        # Rebuild pareto from saved front
        pareto_path = run_dir / "pareto_front.json"
        if pareto_path.exists():
            pareto = ParetoFront.load(str(pareto_path))
        return pareto, round_summary

    # A stateful candidate cannot safely resume by running only missing tasks:
    # its store may already contain memory from the failed task or from later
    # tasks. Persist the exact round-start canonical state once, then use this
    # immutable snapshot for every candidate launch and any clean full retry.
    round_start_dir = round_dir / "round_start_state"
    round_start_path = canonical_pool_path(round_start_dir)
    existing_candidate_dirs = list(round_dir.glob("candidate_*"))
    if round_start_path.exists():
        round_start_state = _load_canonical_state(round_start_dir)
    else:
        if existing_candidate_dirs:
            raise RuntimeError(
                f"Round {round_id} has candidate state but no round-start canonical "
                "snapshot; refusing an inexact resume"
            )
        round_start_state = _load_canonical_state(run_dir)
        _save_canonical_state(round_start_dir, round_start_state)
    pool_units = list(round_start_state["units"])
    logger.info("Round %d: canonical pool has %d units", round_id, len(pool_units))

    # Fixed batch shared across all rounds for cross-round comparability.
    batch_indices = list(search_batch_indices)

    # Resume: if candidates.json from a previous (partial) round exists, reuse
    # it instead of asking the search LLM for fresh candidates. This preserves
    # any per-task results that already landed under those exact architectures
    # (the eval subprocess auto-skips existing task json files), so partial
    # round work survives operator-initiated restarts cleanly.
    candidates_json_path = round_dir / "candidates.json"
    candidates_manifest_path = round_dir / "candidates_manifest.json"
    candidates_raw = None
    candidates_reused = False
    if candidates_json_path.exists() and args.resume:
        try:
            candidates_raw = _load_bound_candidate_checkpoint(
                candidates_json_path, candidates_manifest_path
            )
            candidates_reused = True
            logger.info(
                "Round %d: reused %d candidates from existing candidates.json "
                "(resume mode preserves partial task work).",
                round_id, len(candidates_raw),
            )
        except Exception as e:
            raise RuntimeError(
                f"Round {round_id} candidate checkpoint is unreadable or altered; "
                "refusing to attach existing task results to regenerated architectures"
            ) from e

    # Dry-run is an offline deterministic smoke path and never calls an LLM.
    if candidates_raw is not None:
        pass
    elif args.dry_run:
        candidates_raw = _fallback_candidates(round_id, args.num_candidates)
    elif getattr(args, "random_search", False):
        rng = random.Random(args.random_search_seed + round_id)
        candidates_raw = _sample_random_candidates(
            round_id=round_id, n=args.num_candidates, rng=rng,
        )
        logger.info("Round %d: sampled %d random candidates (seed=%d)",
                    round_id, len(candidates_raw),
                    args.random_search_seed + round_id)
    else:
        # R1 + R2+ both use the LLM proposer. R1 has no champion / no
        # ledger yet; the prompt template handles the round_id == 1 branch
        # by asking for {{ num_candidates }} candidates that span the
        # search space (baseline / explore_retrieval / explore_extract / ...).
        # The previous Method-4 hardcoded R1 seeds (seed_minimal /
        # seed_balanced / seed_graph) anchored the entire search to the
        # seed_balanced family — removed 2026-05-12 to recover diversity.
        # _round1_hardcoded_seeds() is kept below for reference / rollback.
        candidates_raw = generate_candidates(
            model=model,
            round_id=round_id,
            max_rounds=args.max_rounds,
            num_candidates=args.num_candidates,
            pareto=pareto,
            pool_units=pool_units,
            baseline_stats=baseline_stats,
            last_round_details=last_round_details,
            cumulative_principles=cumulative_principles,
            prompt_path=(getattr(args, "search_prompt", None) or str(SEARCH_PROMPT)),
            exploration_hints=exploration_hints,
            obs_graph_json=obs_graph_json,
            benchmark_name=getattr(args, "benchmark", "GAIA"),
        )

    # Codex CR2-11: when architect LLM produces nothing (transient outage,
    # JSON parse failure, etc.) we MUST still proceed with deterministic
    # fallback candidates rather than silently skipping the round. The old
    # `return pareto, {}` here was unreachable in dry_run but burned real
    # rounds in non-dry_run mode.
    if not candidates_raw:
        logger.warning(
            "Architect LLM returned no candidates for round %d; "
            "falling back to deterministic candidates.", round_id,
        )
        candidates_raw = _fallback_candidates(round_id, args.num_candidates)
    # ----------------------------------------------------------------------
    # Champion injection (elitism). For round_id >= 2 with a non-empty Pareto
    # front, prepend the current Pareto best as candidate 0 (unchanged
    # architecture, re-evaluated on this round's batch). The LLM-proposed
    # candidates are renumbered to id 1..K-1 and truncated to fit
    # num_candidates - 1 slots. Per-round best becomes monotone non-decreasing
    # under fixed batch (modulo agent stochasticity on the same architecture).
    # Note: at R2 the champion is whichever R1 LLM-proposed candidate first
    # entered the Pareto front. Since R1 now spans the search space rather
    # than using fixed seeds (Method 4 was removed 2026-05-12), the R2
    # champion family is no longer pre-determined.
    # ----------------------------------------------------------------------
    # Pull prior champion accuracy from last_round_details if available so the
    # IncumbentGate can compare same-arch acc across rounds (Pareto best).
    # Codex review fix (2026-05-13): the previous code looked for a top-level
    # "champion" key that nothing writes. The actual location is inside
    # last_round_details["candidates"][i] whose diversity_role == "champion".
    _prev_champ_acc = None
    try:
        if isinstance(last_round_details, dict):
            cands = last_round_details.get("candidates") or []
            for c in cands:
                if not isinstance(c, dict):
                    continue
                if (c.get("diversity_role") or "").lower() == "champion":
                    metrics = c.get("metrics") or {}
                    _prev_champ_acc = float(metrics.get("accuracy") or 0.0) or None
                    break
    except Exception:
        _prev_champ_acc = None
    if candidates_reused:
        # Resume re-injection guard (2026-06-11): candidates.json is saved
        # AFTER champion injection, so a reused list already contains this
        # round's champion at id 0. Re-injecting here would prepend a second
        # champion, shift every reloaded candidate down one slot, and silently
        # drop the last one — the existing per-candidate task files would then
        # belong to DIFFERENT architectures than the slots they sit in
        # (mid-round resume corruption; bug observed live on
        # gaia_evo_v2_260611 round 7, 2026-06-11, and latent in all earlier
        # mid-round resumes with elitism enabled).
        champion = None
        logger.info(
            "Round %d: candidates reused from checkpoint; skipping champion "
            "re-injection (reloaded list is already injection-complete).",
            round_id,
        )
    elif getattr(args, "disable_elitism", False):
        champion = None
        if round_id > 1:
            logger.info(
                "Round %d: --disable_elitism set; skipping champion injection. "
                "All %d slots are freshly sampled / proposed.",
                round_id, args.num_candidates,
            )
    else:
        # Protocol-v2 A2: under paired acceptance the champion lineage is
        # governed by champion_state.json (sign-test-gated succession), not
        # by pareto.best() — a max over noisy draws can flip leadership on
        # noise alone, which is exactly what the paired gate prevents.
        _champ_override = None
        if getattr(getattr(args, "_protocol", None), "acceptance", "threshold") == "paired":
            from automem.search.protocol import load_champion_state
            _champ_override = load_champion_state(run_dir)
        champion = _make_champion_candidate(
            pareto, round_id,
            prev_champion_acc=_prev_champ_acc,
            batch_size=getattr(args, "batch_size", 50),
            champion_override=_champ_override,
        ) if round_id > 1 else None
    if champion is not None:
        kept = list(candidates_raw[: max(0, args.num_candidates - 1)])
        for i, c in enumerate(kept):
            # Force candidate_id to 1..len(kept) so champion (id=0) does not collide.
            try:
                c["candidate_id"] = i + 1
            except (TypeError, AttributeError):
                pass
        candidates_raw = [champion] + kept
        best = pareto.best()
        logger.info(
            "Round %d: injected champion (config=%s, prior_acc=%.3f, "
            "prior_fit=%.3f) as candidate_0; %d LLM candidates renumbered to "
            "id 1..%d.",
            round_id, best.config_id, best.accuracy, best.fitness,
            len(kept), len(kept),
        )

    # Candidate IDs are filesystem identities, not proposer-controlled data.
    # Always derive them from the final ordered list so duplicate, negative or
    # missing model-supplied IDs cannot make subprocesses share a directory.
    normalized_candidates: List[Dict[str, Any]] = []
    for candidate in candidates_raw:
        if not isinstance(candidate, dict):
            logger.warning("Dropping non-object candidate before evaluation: %r", candidate)
            continue
        normalized = dict(candidate)
        normalized["candidate_id"] = len(normalized_candidates)
        normalized_candidates.append(normalized)
        if len(normalized_candidates) >= args.num_candidates:
            break
    candidates_raw = normalized_candidates

    # Bind filesystem candidate slots to this exact architecture list. Both
    # files are atomic; a corrupt/missing manifest makes resume fail closed.
    _atomic_write_json(candidates_json_path, candidates_raw)
    _atomic_write_json(
        candidates_manifest_path,
        {
            "schema_version": 1,
            "sha256": _candidate_checkpoint_digest(candidates_raw),
        },
    )

    # Evaluate candidates in parallel (same pool for all, fair comparison)
    # Phase 1: Validate, compile, import pool, launch eval subprocesses
    eval_jobs = []  # list of (cand, config_id, arch, cand_dir, tasks_dir, proc)
    round_results = []
    for cand in candidates_raw[:args.num_candidates]:
        cand_id = int(cand.get("candidate_id", len(eval_jobs) + len(round_results)))
        config_id = f"r{round_id}_c{cand_id}"
        arch = cand.get("architecture", {})

        # Validate
        valid, reason = _validate_candidate(cand)
        if not valid:
            logger.warning("Candidate %s invalid: %s; skipping.", config_id, reason)
            round_results.append({
                "config_id": config_id, "skipped": True, "reason": reason, "architecture": arch,
            })
            continue

        cand_dir = round_dir / f"candidate_{cand_id}"
        reuse_completed_outputs = False
        if cand_dir.exists() and not args.dry_run:
            existing_tasks_dir = cand_dir / "tasks"
            present, duplicates, invalid_files = _scan_task_result_indices(
                existing_tasks_dir,
                getattr(args, "_task_dataset_sha256", None),
            )
            expected = set(batch_indices)
            reuse_completed_outputs = (
                present == expected and not duplicates and not invalid_files
            )
            if reuse_completed_outputs:
                logger.info(
                    "%s: reusing an exact completed task set from the bound "
                    "candidate checkpoint.",
                    config_id,
                )
            elif any(cand_dir.iterdir()):
                # Partial candidate storage is causally unsafe: it may contain
                # memory written after a task whose JSON is missing. Rebuild
                # the whole candidate from the immutable round-start snapshot.
                import shutil as _shutil

                logger.warning(
                    "%s: discarding partial candidate state and rerunning the "
                    "full batch from the round-start canonical snapshot.",
                    config_id,
                )
                _shutil.rmtree(cand_dir)
        cand_dir.mkdir(parents=True, exist_ok=True)
        storage_dir = str(cand_dir / "storage")
        tasks_dir = cand_dir / "tasks"
        tasks_dir.mkdir(exist_ok=True)

        # Compile
        runtime_config = _compile_architecture(arch, storage_dir)
        if runtime_config is None:
            logger.warning("Compilation failed for %s; skipping.", config_id)
            round_results.append({
                "config_id": config_id, "skipped": True,
                "reason": "compilation_failed", "architecture": arch,
            })
            continue

        # Import canonical pool into compiled store directories.
        # Skipped under --no_canonical_import: each candidate starts empty,
        # only accumulating from its own 50-task subprocess.
        if reuse_completed_outputs:
            logger.info("%s: completed checkpoint skips storage re-import.", config_id)
        elif args.dry_run:
            logger.info("%s: dry-run skips canonical storage import.", config_id)
        elif getattr(args, "no_canonical_import", False):
            logger.info(
                "Round %d %s: --no_canonical_import set; starting with EMPTY store.",
                round_id, config_id,
            )
        else:
            import_canonical_to_storage(round_start_dir, runtime_config)

        # Launch eval subprocess (non-blocking)
        proc = None
        if not args.dry_run and not reuse_completed_outputs:
            proc = _start_eval_subprocess(tasks_dir=tasks_dir, task_indices=batch_indices,
                                          runtime_config=runtime_config, args=args)

        eval_jobs.append((cand, config_id, arch, cand_dir, tasks_dir, proc))

    # Phase 2: Wait for all eval subprocesses to complete; track failures.
    # Codex CR2-3: a non-zero exit signifies partial output that must NOT
    # enter the Pareto front, since the candidate may have completed only
    # the easy tasks before crashing and would otherwise look artificially
    # strong to the architect.
    failed_config_ids: set = set()
    failure_reasons: Dict[str, str] = {}
    for cand, config_id, arch, cand_dir, tasks_dir, proc in eval_jobs:
        if proc is not None:
            proc.wait()
            if hasattr(proc, "_log_file"):
                proc._log_file.close()
            if proc.returncode != 0:
                logger.error(
                    "Eval subprocess failed for %s (exit %d) — candidate will "
                    "be marked failed and excluded from Pareto + canonical sync.",
                    config_id, proc.returncode,
                )
                failed_config_ids.add(config_id)
                failure_reasons[config_id] = "subprocess_nonzero_exit"
            else:
                logger.info("Eval subprocess done for %s.", config_id)

    # ----------------------------------------------------------------------
    # FB1 + FB2: post-eval batch consistency check + self-heal. Quota cleanup
    # or partial-failure resume can leave a candidate evaluated on a slightly
    # different task subset than search_batch (run7 R1: candidates ran 45-48
    # of the expected 50, drifted by ~3-7 indices). _ensure_batch_complete
    # auto-retries the eval subprocess for missing indices ONLY (existing
    # task json files are skipped by the eval subprocess), and marks the
    # candidate failed if gaps remain after retry — biased subsets must not
    # enter the Pareto front.
    # ----------------------------------------------------------------------
    drift_report: Dict[str, Dict[str, List[int]]] = {}
    if not args.dry_run:
        for cand, config_id, arch, cand_dir, tasks_dir, proc in eval_jobs:
            if config_id in failed_config_ids:
                continue
            complete, missing, extras = _ensure_batch_complete(
                tasks_dir=tasks_dir,
                expected_indices=batch_indices,
                arch=arch,
                cand_dir=cand_dir,
                args=args,
                config_id=config_id,
                canonical_source_dir=round_start_dir,
            )
            drift_report[config_id] = {"missing": missing, "extras": extras}
            if not complete:
                logger.error(
                    "%s: task outputs do not exactly match the search batch "
                    "after retry (missing=%s extras=%s). Marking failed to "
                    "prevent biased metrics from polluting Pareto.",
                    config_id, missing, extras,
                )
                failed_config_ids.add(config_id)
                failure_reasons[config_id] = "non_exact_task_results"

    # Phase 3: Compute metrics, attribution, diagnosis (sequential — cheap)
    for cand, config_id, arch, cand_dir, tasks_dir, proc in eval_jobs:
        if config_id in failed_config_ids:
            # Record a failed entry but skip Pareto / attribution work so
            # partial-task fitness cannot pollute downstream feedback.
            round_results.append({
                "config_id": config_id,
                "hypothesis": cand.get("hypothesis", ""),
                "diversity_role": cand.get("diversity_role", ""),
                "rationale": cand.get("rationale", ""),
                "architecture": arch,
                "metrics": {},
                "attribution_summary": {},
                "added_to_front": False,
                "failed": True,
                "failure_reason": failure_reasons.get(
                    config_id, "candidate_evaluation_failed"
                ),
            })
            continue
        # Compute fitness (passes pool_units to enable Conditional Lift / MOR)
        # E7 fix (2026-05-16): wrap in try/except FitnessComputationError so a
        # mid-run evaluation failure routes the candidate to failed_config_ids
        # instead of polluting the Pareto front with a zeroed fitness.
        if args.dry_run:
            fitness_score, raw_metrics = _synthetic_candidate_metrics(
                arch,
                round_id=round_id,
                candidate_id=int(cand.get("candidate_id", 0)),
                total_tasks=len(batch_indices),
                baseline_accuracy=baseline_stats.get("baseline_accuracy", 0.0),
            )
        else:
            try:
                fitness_score, raw_metrics = compute_candidate_fitness(
                    tasks_dir=tasks_dir,
                    baseline_accuracy=baseline_stats.get("baseline_accuracy", 0.0),
                    baseline_per_task=baseline_stats.get("per_task_scores"),
                    round_id=round_id,
                    architecture=arch,
                    token_cap_per_task=args.token_cap_per_task,
                    latency_cap_per_task=args.latency_cap_per_task,
                    pool_units=pool_units,
                )
            except FitnessComputationError as fe:
                logger.warning(
                    "Fitness computation failed for %s: %s; routing to failed_config_ids.",
                    config_id, fe,
                )
                failed_config_ids.add(config_id)
                failure_reasons[config_id] = "fitness_computation_failed"
                round_results.append({
                    "config_id": config_id,
                    "added_to_front": False,
                    "failed": True,
                    "failure_reason": f"fitness_error_{fe.reason}",
                })
                continue

        # Post-hoc attribution audit (against round-start pool, fair comparison)
        task_results = load_task_results(str(tasks_dir))
        attributions, audit_summary = run_posthoc_audit(
            task_results=task_results,
            canonical_units=pool_units,
            top_k=5,
            max_steps=int(args.max_steps),
        )

        # H plan (2026-04-28): LLM-based deep sub-classification of the
        # REASONING_ERROR bucket. Only invoked when diagnosis_model is set
        # and not in dry_run, since it costs ~6-10 extra LLM calls per
        # candidate. Output is recorded inside attribution.json so the
        # next architect round sees the fine-grained breakdown.
        reasoning_subclasses = {}
        if diagnosis_model and not args.dry_run:
            try:
                from automem.search.attribution import llm_subclassify_reasoning_errors
                reasoning_subclasses = llm_subclassify_reasoning_errors(
                    task_results=task_results,
                    rule_attributions=attributions,
                    model=diagnosis_model,
                )
                if reasoning_subclasses:
                    logger.info(
                        "H plan: %s sub-classified %d reasoning errors for %s",
                        args.diagnosis_model, len(reasoning_subclasses), config_id,
                    )
            except Exception as e:
                logger.warning(
                    "llm_subclassify_reasoning_errors failed for %s: %s",
                    config_id, e,
                )

        # Smart-8 fix (2026-05-16): per-task memory compliance — for failed
        # tasks, ask LLM whether agent followed the retrieved units'
        # instructions across 6 dimensions. Helps distinguish "memory wrong"
        # from "agent ignored memory".
        memory_compliance_data = None
        if diagnosis_model and not args.dry_run:
            try:
                from automem.search.memory_compliance import (
                    compute_per_task_compliance, aggregate_candidate_compliance,
                )
                per_task_compliance = []
                # Sample up to 10 failed tasks (keep LLM cost bounded; tasks
                # are picked by lowest task_score = worst failures first).
                failed_tasks = sorted(
                    [t for t in task_results if t.get("task_score", 0.0) < 1.0],
                    key=lambda t: t.get("task_score", 0.0),
                )[:10]
                for t in failed_tasks:
                    rmt = (t.get("retrieved_memory_text")
                           or t.get("retrieved_memory_context") or "")
                    if not rmt:
                        continue
                    # Split units crudely on the standard "[TYPE]" header.
                    # G8 fix (codex review, 2026-05-16): filter out the
                    # leading preamble chunk (text BEFORE the first [TYPE]
                    # header). Without this filter, prose preambles become
                    # phantom "units" sent to extract_instructions, burning
                    # LLM calls on non-instruction text.
                    import re as _re
                    units = _re.split(r"(?=\[[A-Z_]+\])", str(rmt))
                    units = [u.strip() for u in units if u.strip()]
                    units = [u for u in units if _re.match(r"^\[[A-Z_]+\]", u)]
                    if not units:
                        continue
                    comp = compute_per_task_compliance(
                        t, units, diagnosis_model, max_units=3,
                    )
                    per_task_compliance.append(comp)
                memory_compliance_data = aggregate_candidate_compliance(
                    per_task_compliance,
                )
                logger.info(
                    "Memory compliance for %s: avg_score=%.3f over %d tasks. %s",
                    config_id,
                    memory_compliance_data.get("avg_followed_score", 0.0),
                    memory_compliance_data.get("n_tasks_with_compliance_data", 0),
                    memory_compliance_data.get("interpretation", ""),
                )
            except Exception as e:
                logger.warning("Memory compliance computation failed for %s: %s",
                               config_id, e)

        # Smart-10 (2026-05-17): reclassify REASONING_ERROR tasks whose LLM
        # subclass maps to an A3 top category — single source of truth in
        # breakdown counts.
        # I2 fix (codex review, 2026-05-17): MUST run BEFORE build_layer_diagnosis
        # so the evidence_by_category that the layer LLM sees (built internally
        # via _select_evidence_tasks) is consistent with what we later pass to
        # the synthesizer. Otherwise the two _select_evidence_tasks calls
        # produce divergent category buckets and the synthesizer sees
        # contradictory signals.
        try:
            from automem.search.attribution import reclassify_reasoning_from_subclasses
            reclassify_reasoning_from_subclasses(
                attributions, reasoning_subclasses, audit_summary,
            )
        except Exception as e:
            logger.warning("[Smart-10] reclassification failed for %s: %s", config_id, e)

        # B1 fix (2026-05-16): run LLM 4-layer diagnosis BEFORE saving so
        # save_attribution_report can persist layer_diag to disk. Previously
        # save happened first and layer_diag was lost on disk (only kept
        # in-memory via ParetoEntry.attribution_summary).
        layer_diag = {}
        if diagnosis_model and not args.dry_run:
            try:
                from automem.search.attribution import build_layer_diagnosis
                layer_diag = build_layer_diagnosis(
                    model=diagnosis_model,
                    task_results=task_results,
                    summary=audit_summary,
                    architecture=arch,
                    # Smart-1 fix (2026-05-16): pass per-task attributions so the
                    # diagnosis prompt can surface 3 worst-task evidence per
                    # category. Without this the LLM only sees aggregate counts.
                    attributions=attributions,
                    # Smart-2 fix (2026-05-16): pass run_dir + round_id so the
                    # diagnosis prompt can read past round_done.json files and
                    # surface metric trajectory (hit_rate trend, breakdown
                    # shift, pool size plateau, etc.).
                    run_dir=run_dir,
                    round_id=round_id,
                )
                # F8 fix (codex review, 2026-05-16): call_llm_json now returns
                # a stub dict {"_parse_failed": True, ...} on retry exhaustion.
                # Drop it rather than persisting the stub as a real diagnosis.
                if layer_diag.get("_parse_failed"):
                    logger.warning(
                        "Layer diagnosis stub returned for %s: %s; not persisting.",
                        config_id, layer_diag.get("_last_err"),
                    )
                    layer_diag = {}
                else:
                    logger.info("Layer diagnosis for %s: %s",
                                config_id, layer_diag.get("overall", ""))
            except Exception as e:
                logger.warning("Layer diagnosis failed for %s: %s", config_id, e)

        # Smart-1/12 (2026-05-17): also build evidence_by_category once here so
        # we can both persist it (S12) and pass it to the synthesizer (S9a).
        evidence_by_category_for_save: Dict[str, Any] = {}
        try:
            from automem.search.attribution import _select_evidence_tasks
            evidence_by_category_for_save = _select_evidence_tasks(
                attributions, task_results, top_k_per_category=3,
            )
        except Exception as e:
            logger.warning("[Smart-12] evidence_by_category extraction failed: %s", e)

        # Smart-9a (2026-05-17): synthesize all 6 diagnostic signals into a
        # single natural-language verdict via gpt-5.5. This is what the next
        # round's proposer reads as the PRIMARY signal (§2A).
        synthesized_verdict_data: Dict[str, Any] = {}
        if diagnosis_model and not args.dry_run and layer_diag:
            try:
                from automem.search.attribution import build_synthesized_verdict
                from automem.search.attribution import _gather_historical_metrics
                _hist = (_gather_historical_metrics(run_dir, round_id, max_lookback=4)
                         if round_id > 1 else [])
                synthesized_verdict_data = build_synthesized_verdict(
                    model=diagnosis_model,
                    rule_diagnosis=audit_summary.to_dict().get("rule_diagnosis", ""),
                    layer_diagnosis=layer_diag,
                    memory_compliance=memory_compliance_data,
                    breakdown=audit_summary.to_dict().get("breakdown", {}),
                    evidence_by_category=evidence_by_category_for_save,
                    historical_metrics=_hist,
                )
                if synthesized_verdict_data.get("_parse_failed"):
                    err_msg = synthesized_verdict_data.get("_last_err") or "unknown error"
                    logger.warning(
                        "[Smart-9a] synthesize_verdict stub for %s: %s",
                        config_id, err_msg,
                    )
                    # I4 fix (codex review, 2026-05-17): emit a minimal
                    # placeholder verdict so the proposer's §2E.1 doesn't
                    # silently lose the candidate. The placeholder marks
                    # confidence=low and points readers to §2E.3.
                    synthesized_verdict_data = {
                        "primary_signal": "(synthesizer failed — see layer_diagnosis in §2E.3 supporting detail)",
                        "confidence": "low",
                        "recommended_action": "Inspect §2E.3 supporting detail manually for this candidate.",
                        "memory_bypass": False,
                        "out_of_scope_for_memory": False,
                        "evidence_task_ids": [],
                        "cross_source_agreement": f"synthesizer_failed: {err_msg[:80]}",
                        "reasoning": "Synthesizer LLM did not return a valid verdict; raw signals available in §2E.3.",
                    }
                else:
                    logger.info(
                        "[Smart-9a] %s: %s (confidence=%s, memory_bypass=%s, oos=%s)",
                        config_id,
                        (synthesized_verdict_data.get("primary_signal") or "")[:120],
                        synthesized_verdict_data.get("confidence"),
                        synthesized_verdict_data.get("memory_bypass"),
                        synthesized_verdict_data.get("out_of_scope_for_memory"),
                    )
            except Exception as e:
                logger.warning("[Smart-9a] synthesize_verdict raised for %s: %s",
                               config_id, e)

        save_attribution_report(
            attributions, audit_summary,
            str(cand_dir / "attribution.json"),
            reasoning_subclasses=reasoning_subclasses,
            layer_diag=layer_diag,
            memory_compliance=memory_compliance_data,
            evidence_by_category=evidence_by_category_for_save,
            synthesized_verdict=synthesized_verdict_data,
        )

        attr_dict = audit_summary.to_dict()
        if layer_diag:
            attr_dict["layer_diagnosis"] = layer_diag
        # H plan: surface sub-class breakdown to round_summary so the
        # next architect round sees the fine-grained reasoning_error split.
        # Codex R3-7: also surface (n_sampled, n_total) so the architect
        # knows whether the breakdown is exhaustive or estimated.
        if reasoning_subclasses:
            sub_counts: Dict[str, int] = {}
            meta = reasoning_subclasses.get("__meta__", {}) or {}
            for tid, info in reasoning_subclasses.items():
                if tid == "__meta__":
                    continue
                key = info.get("subclass") or "true_reasoning_error"
                sub_counts[key] = sub_counts.get(key, 0) + 1
            attr_dict["reasoning_error_subclasses"] = sub_counts
            if meta.get("is_sampled") == "true":
                attr_dict["reasoning_error_subclasses_meta"] = {
                    "n_sampled": int(meta.get("n_sampled", 0)),
                    "n_total_reasoning": int(meta.get("n_total_reasoning", 0)),
                    "is_sampled": True,
                }

        # Build Pareto entry
        entry = ParetoEntry(
            config_id=config_id,
            architecture=arch,
            round_id=round_id,
            accuracy=raw_metrics.get("accuracy", 0.0),
            memory_lift=raw_metrics.get("memory_lift", 0.0),
            hit_rate=raw_metrics.get("hit_rate", 0.0),
            token_eff=raw_metrics.get("token_eff", 0.0),
            fitness=fitness_score,
            raw_metrics=raw_metrics,
            attribution_summary=attr_dict,
        )
        # Champion candidates are re-evaluations of an architecture already
        # on the Pareto front — do NOT add a duplicate entry; just record the
        # current-round metrics for round_summary / variance diagnostics.
        if cand.get("diversity_role") == "champion":
            if pareto.measurement_mode == "pooled":
                # Protocol-v2 A1: the champion re-eval is exactly the repeated
                # measurement pooling exists for — fold it into the champion's
                # running mean instead of discarding it (legacy kept the front
                # entry at its original, max-selected and therefore upward-
                # biased draw; that is the winner's-curse mechanism behind the
                # flat best-so-far curves).
                added_to_front = pareto.add(entry)
                logger.info(
                    "%s [champion re-eval, POOLED] | this draw acc=%.3f "
                    "fit=%.3f -> pooled %s",
                    config_id, raw_metrics.get("accuracy", 0.0), fitness_score,
                    pareto.best().summary_str() if pareto.best() else "?",
                )
            else:
                added_to_front = False
                logger.info(
                    "%s [champion re-eval] | acc=%.3f lift=%+.3f fit=%.3f "
                    "(prior fitness was %.3f); not added to front.",
                    config_id, raw_metrics.get("accuracy", 0.0),
                    raw_metrics.get("memory_lift", 0.0), fitness_score,
                    pareto.best().fitness if pareto.best() else 0.0,
                )
        else:
            added_to_front = pareto.add(entry)
            logger.info("%s %s | %s", config_id, "(FRONT)" if added_to_front else "(dominated)", entry.summary_str())

        result_entry = {
            "config_id": config_id,
            "hypothesis": cand.get("hypothesis", ""),
            "diversity_role": cand.get("diversity_role", ""),
            "rationale": cand.get("rationale", ""),
            "architecture": arch,
            "metrics": raw_metrics,
            "attribution_summary": attr_dict,
            "added_to_front": added_to_front,
        }
        round_results.append(result_entry)

        # Save candidate result
        with open(cand_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result_entry, f, indent=2, ensure_ascii=False)

    # Sync all SUCCESSFUL candidates' memories to canonical pool. Codex CR2-3:
    # do not absorb partial output from failed subprocesses — those memories
    # may be biased toward the easier tasks the candidate completed.
    # Codex Round-3 R3-2: pass the round-start canonical snapshot so
    # sync_tasks_to_canonical can compute deltas instead of taking max().
    round_start_pool_by_id: Dict[str, Dict[str, Any]] = {
        u.get("id"): u for u in pool_units if u.get("id")
    }
    # Protocol-v2 C7: winner-only canonical merge. Unconditional absorption
    # let every candidate — including clearly harmful ones — write into the
    # shared pool, which both poisons later rounds (xBench validation lift
    # -3.3pp) and makes the evaluation environment drift round-over-round.
    # Under "winner", only the round's best successful candidate merges.
    _merge_gate = getattr(getattr(args, "_protocol", None), "canonical_merge", "all")
    _winner_cfg_id: Optional[str] = None
    if _merge_gate == "winner":
        _ok_rows = [r for r in round_results
                    if r.get("config_id") not in failed_config_ids and r.get("metrics")]
        if _ok_rows:
            _winner_cfg_id = max(
                _ok_rows, key=lambda r: float(r.get("metrics", {}).get("fitness", 0.0))
            )["config_id"]
        logger.info("Canonical merge gate: winner-only (winner=%s)", _winner_cfg_id)
    for cand in candidates_raw[:args.num_candidates]:
        cand_id = int(cand.get("candidate_id", 0))
        cfg_id = f"r{round_id}_c{cand_id}"
        if cfg_id in failed_config_ids:
            logger.info("Skipping canonical sync for failed candidate %s", cfg_id)
            continue
        if _winner_cfg_id is not None and cfg_id != _winner_cfg_id:
            logger.info("Skipping canonical sync for %s (merge gate: winner=%s)",
                        cfg_id, _winner_cfg_id)
            continue
        cand_dir = round_dir / f"candidate_{cand_id}"
        if cand_dir.exists():
            sync_tasks_to_canonical(
                run_dir,
                cand_dir,
                round_start_pool_by_id,
                merge_id=f"round:{round_id}:candidate:{cfg_id}",
            )

    # Codex Q4-1 fix (2026-04-28): after merging candidate signals into
    # canonical, run the mandatory periodic ops on the canonical pool
    # itself. R3-1 (candidate-local deactivation) was safe only on the
    # premise that *something* actually prunes canonical; without this,
    # high-leakage tips and toxic units survive and re-poison every round.
    _run_canonical_periodic_ops(run_dir, round_id=round_id)

    # Save Pareto front checkpoint
    pareto.save(str(run_dir / "pareto_front.json"))

    # Protocol-v2 A2: paired-acceptance champion succession. The challenger
    # only takes over when it beats THIS round's champion measurement on a
    # per-task sign test (same fold, same tasks => paired). Replaces the
    # legacy behavior where pareto.best() — a max over noisy draws — was
    # the de-facto champion.
    champion_decision: Optional[Dict[str, Any]] = None
    if getattr(getattr(args, "_protocol", None), "acceptance", "threshold") == "paired":
        try:
            from automem.search import protocol as _p2
            per_task_by_config: Dict[str, Dict[str, float]] = {}
            for r in round_results:
                cfg = r.get("config_id", "")
                cand_tasks_dir = round_dir / f"candidate_{cfg.split('_c')[-1]}" / "tasks"
                if cand_tasks_dir.exists():
                    per_task_by_config[cfg] = _p2.per_task_scores_from_results(
                        load_task_results(str(cand_tasks_dir))
                    )
            champion_decision = _p2.update_champion_after_round(
                run_dir, round_id, round_results, per_task_by_config,
                alpha=getattr(getattr(args, "_protocol", None), "accept_alpha", 0.10),
            )
            logger.info("[paired_acceptance] %s",
                        json.dumps(champion_decision, ensure_ascii=False))
        except Exception as _e:
            logger.warning("[paired_acceptance] update failed: %s", _e)

    pool = load_canonical_pool(run_dir)

    # Round-level aggregate metrics + cumulative tracking (2026-05-13).
    # These let downstream analysis distinguish "best in this round" vs
    # "best ever" and answer "how many rounds to reach acc X" without
    # post-processing the per-candidate JSON files.
    round_metrics = _compute_round_metrics(round_results)
    thresholds = _parse_discovery_thresholds(getattr(args, "discovery_thresholds", "0.55,0.60,0.65,0.70"))
    cumulative_tracking = _update_cumulative_tracking(
        prev_state={"cumulative_tracking": prev_cumulative_tracking} if prev_cumulative_tracking else {},
        round_id=round_id,
        round_metrics=round_metrics,
        thresholds=thresholds,
    )

    round_summary = {
        "round_id": round_id,
        "batch_size": len(batch_indices),
        "expected_batch_indices": list(batch_indices),
        "num_candidates_evaluated": len([r for r in round_results if not r.get("skipped")]),
        "pareto_front_size": pareto.size(),
        "best_fitness": pareto.best().fitness if pareto.best() else 0.0,
        "pool_size": len(pool),
        "candidate_results": round_results,
        # FB2 audit trail: per-candidate task-set drift vs search_batch
        # (post self-heal). Empty arrays = perfectly aligned. Used by
        # downstream analysis to flag rounds where signal may be biased.
        "task_subset_drift": drift_report,
        # Round-level aggregates + cumulative cross-round state (2026-05-13).
        **round_metrics,
        "cumulative_tracking": cumulative_tracking,
        # Protocol-v2 A2: paired champion-succession decision (None under
        # legacy threshold acceptance).
        "champion_decision": champion_decision,
    }

    # NOTE: done_marker write is deferred until AFTER validation_checkpoint
    # (Codex CR2-12). The original code wrote round_done.json HERE, before
    # validation, so a crash mid-validation would leave the round flagged
    # done and resume would skip the missing checkpoint forever.

    logger.info(
        "Round %d evaluation done (pre-validation): front_size=%d, "
        "best_fitness=%.4f, pool_size=%d",
        round_id, pareto.size(),
        round_summary["best_fitness"], len(pool),
    )

    # ---- Validation checkpoint (fixed set, cross-round comparable) ----
    run_val_ckpt = (
        args.val_every > 0
        and round_id % args.val_every == 0
        and not args.dry_run
    )
    if run_val_ckpt:
        best_entry = pareto.best()
        val_ckpt_dir = round_dir / "validation_checkpoint"
        val_ckpt_marker = val_ckpt_dir / "checkpoint_done.json"
        if best_entry and not val_ckpt_marker.exists():
            val_ckpt_dir.mkdir(parents=True, exist_ok=True)
            val_tasks_dir = val_ckpt_dir / "tasks"
            val_tasks_dir.mkdir(parents=True, exist_ok=True)
            val_storage = str(val_ckpt_dir / "storage")

            val_config = _compile_architecture(best_entry.architecture, val_storage)
            if val_config:
                if not getattr(args, "no_canonical_import", False):
                    import_canonical_to_storage(run_dir, val_config)
                # Codex Q14-6 fix (2026-04-28): only write the
                # checkpoint marker if the validation subprocess
                # SUCCEEDED and produced all requested task results.
                # Previously the marker was written unconditionally,
                # so a partial validation got persisted as "complete"
                # and a later --resume skipped re-running it. The
                # round summary then reported partial/zero metrics
                # as authoritative validation.
                val_ok = _run_eval_subprocess(
                    tasks_dir=val_tasks_dir,
                    task_indices=list(split.validation_indices),
                    runtime_config=val_config,
                    args=args,
                )
                exact_error = None
                if val_ok:
                    try:
                        _require_exact_task_results(
                            val_tasks_dir,
                            list(split.validation_indices),
                            f"Validation checkpoint round {round_id}",
                            getattr(args, "_task_dataset_sha256", None),
                        )
                    except RuntimeError as exc:
                        exact_error = str(exc)
                        val_ok = False
                if not val_ok:
                    logger.warning(
                        "Validation subprocess R%d failed or returned a "
                        "non-exact task set (%s). Skipping checkpoint "
                        "marker; will retry next eligible round (Codex Q14-6 fix).",
                        round_id, exact_error or "subprocess failure",
                    )
                else:
                    # Validation MOR / Conditional Lift must reflect the pool
                    # actually used during validation — that is, the canonical
                    # pool AFTER candidates synced (see `import_canonical_to_storage`
                    # above), not the round-start `pool_units` snapshot. (Codex CR3)
                    validation_pool = load_canonical_pool(run_dir)
                    # E7 fix: skip checkpoint if fitness computation fails
                    # (e.g. validation subprocess produced no tasks).
                    try:
                        val_fitness, val_metrics = compute_candidate_fitness(
                            tasks_dir=val_tasks_dir,
                            baseline_accuracy=baseline_stats.get("baseline_accuracy", 0.0),
                            baseline_per_task=baseline_stats.get("per_task_scores"),
                            architecture=best_entry.architecture,
                            token_cap_per_task=args.token_cap_per_task,
                            latency_cap_per_task=args.latency_cap_per_task,
                            pool_units=validation_pool,
                        )
                    except FitnessComputationError as fe:
                        logger.warning(
                            "Validation checkpoint fitness failed: %s; skipping checkpoint.",
                            fe,
                        )
                        val_fitness, val_metrics = 0.0, {}
                    # Per-task pass/fail summary (so user can analyze
                    # which specific tasks failed without grepping the
                    # per-task JSONs). Includes GAIA category tag.
                    per_task_summary = _build_per_task_pass_fail_summary(
                        val_tasks_dir,
                        baseline_per_task=baseline_stats.get("per_task_scores"),
                    )
                    ckpt_result = {
                        "round_id": round_id,
                        "best_config_id": best_entry.config_id,
                        "architecture": best_entry.architecture,
                        "validation_metrics": val_metrics,
                        "validation_fitness": val_fitness,
                        "n_passed": sum(1 for r in per_task_summary if r["passed"]),
                        "n_failed": sum(1 for r in per_task_summary if not r["passed"]),
                        "per_task_results": per_task_summary,
                    }
                    with open(val_ckpt_marker, "w", encoding="utf-8") as f:
                        json.dump(ckpt_result, f, indent=2, ensure_ascii=False)
                    round_summary["validation_checkpoint"] = ckpt_result
                    logger.info(
                        "Validation checkpoint R%d: acc=%.4f lift=%.4f fitness=%.4f",
                        round_id,
                        val_metrics.get("accuracy", 0),
                        val_metrics.get("memory_lift", 0),
                        val_fitness,
                    )

    # Smart-4 fix (2026-05-16): differential diagnosis between best and worst
    # candidate. The LLM explains the gap and recommends a direction for the
    # next round's proposer.
    if (diagnosis_model is not None and not args.dry_run
            and len(round_summary.get("candidate_results", []) or []) >= 2):
        try:
            from automem.search.attribution import build_differential_diagnosis
            scored = [
                c for c in round_summary["candidate_results"]
                if not c.get("skipped") and not c.get("failed")
                and (c.get("metrics") or {}).get("accuracy") is not None
            ]
            if len(scored) >= 2:
                scored_sorted = sorted(
                    scored, key=lambda c: c["metrics"].get("accuracy", 0.0)
                )
                worst = scored_sorted[0]
                best = scored_sorted[-1]
                # G3 fix (codex review, 2026-05-16): skip when accuracy gap is
                # too small to be informative — a differential LLM call on two
                # near-tied candidates produces hallucinated reasoning.
                _acc_gap = (
                    (best.get("metrics") or {}).get("accuracy", 0.0)
                    - (worst.get("metrics") or {}).get("accuracy", 0.0)
                )
                if best is not worst and _acc_gap >= 0.02:
                    diff_diag = build_differential_diagnosis(
                        model=diagnosis_model,
                        best_candidate=best,
                        worst_candidate=worst,
                        round_id=round_id,
                    )
                    if isinstance(diff_diag, dict) and not diff_diag.get("_parse_failed"):
                        round_summary["differential_diagnosis"] = diff_diag
                        logger.info(
                            "Differential diagnosis R%d: %s | recommend: %s",
                            round_id,
                            diff_diag.get("key_change", ""),
                            diff_diag.get("recommendation_for_next_round", ""),
                        )
                    else:
                        logger.warning(
                            "Differential diagnosis R%d failed: %s",
                            round_id,
                            (diff_diag or {}).get("_last_err"),
                        )
        except Exception as e:
            logger.warning("Differential diagnosis R%d raised: %s", round_id, e)

    # Smart-9b (2026-05-17): round-level LLM synthesizer. Reads all per-candidate
    # synthesized_verdicts + differential_diagnosis + Pareto state → produces a
    # round-level natural-language verdict for the NEXT round's proposer.
    if diagnosis_model is not None and not args.dry_run:
        try:
            from automem.search.diagnosis_synthesizer import synthesize_round_verdict
            # Collect candidate verdicts from this round's round_summary.
            cand_verdicts: List[Dict[str, Any]] = []
            for c in (round_summary.get("candidate_results") or []):
                if c.get("skipped") or c.get("failed"):
                    continue
                attr_sum = c.get("attribution_summary") or {}
                sv = attr_sum.get("synthesized_verdict") or {}
                if sv:
                    cand_verdicts.append({
                        "config_id": c.get("config_id"),
                        "diversity_role": c.get("diversity_role"),
                        "architecture": c.get("architecture", {}),
                        "accuracy": (c.get("metrics") or {}).get("accuracy"),
                        "synthesized_verdict": sv,
                    })
            if cand_verdicts:
                pareto_status = {
                    "front_size": round_summary.get("pareto_front_size"),
                    "best_fitness": round_summary.get("best_fitness"),
                    "round_best_acc": round_summary.get("round_best_accuracy"),
                    "round_mean_acc": round_summary.get("round_mean_accuracy"),
                }
                cum_tracking = round_summary.get("cumulative_tracking", {})
                round_level_verdict = synthesize_round_verdict(
                    model=diagnosis_model,
                    round_id=round_id,
                    candidate_verdicts=cand_verdicts,
                    differential_diagnosis=round_summary.get("differential_diagnosis"),
                    pareto_status=pareto_status,
                    cumulative_tracking=cum_tracking,
                )
                if (isinstance(round_level_verdict, dict)
                        and not round_level_verdict.get("_parse_failed")):
                    round_summary["round_level_verdict"] = round_level_verdict
                    logger.info(
                        "[Smart-9b] R%d round_verdict: %s | next_focus: %s | mode=%s",
                        round_id,
                        (round_level_verdict.get("round_verdict") or "")[:120],
                        (round_level_verdict.get("next_round_focus") or "")[:120],
                        round_level_verdict.get("exploration_vs_exploit"),
                    )
                else:
                    logger.warning(
                        "[Smart-9b] R%d synthesize_round_verdict stub: %s",
                        round_id,
                        (round_level_verdict or {}).get("_last_err"),
                    )
        except Exception as e:
            logger.warning("[Smart-9b] R%d raised: %s", round_id, e)

    # Codex CR2-12: write done_marker AFTER validation_checkpoint so a crash
    # mid-validation does not leave the round flagged done with missing
    # validation data.
    _atomic_write_json(done_marker, round_summary)
    logger.info("Round %d FULLY done (incl. validation if applicable).", round_id)

    return pareto, round_summary


def _sample_random_candidates(
    round_id: int, n: int, rng: random.Random,
) -> List[Dict[str, Any]]:
    """Random candidates from the compatible public subset-Encode space.

    Baseline for method C (Random-Search) in the comparison experiment:
    isolates the contribution of LLM-driven search from the architecture space
    itself (i.e. how much value does the LLM+attribution feedback add over
    random sampling with the same budget?).
    """
    space = RECOMMENDED_ARCHITECTURE_SPACE
    extract_opts = space["extract_types"]
    storage_opts = space["storage_types"]
    retrieval_opts = space["retrieval_types"]
    mgmt_opts = space["management_types"]
    # Injection is no longer a search dimension (2026-05-12). Random sampling
    # produces 4-tuple architectures; runtime applies raw_topk uniformly.
    out: List[Dict[str, Any]] = []
    for cand_id in range(n):
        # Encode is a non-empty subset (uniform over sizes 1..5, then over
        # combinations of that size), routed to one common store.
        subset_size = rng.randint(1, len(extract_opts))
        encode_subset = sorted(
            rng.sample(extract_opts, subset_size), key=extract_opts.index
        )
        store = rng.choice(storage_opts)
        routing = {encode_type: store for encode_type in encode_subset}
        valid_ret = [r for r in get_valid_retrievals(routing) if r in retrieval_opts]
        retrieval = rng.choice(valid_ret)
        valid_mgmt = [
            m for m in get_valid_managements(routing, retrieval=retrieval)
            if m in mgmt_opts
        ]
        out.append({
            "candidate_id": cand_id,
            "hypothesis": "Random baseline candidate (no LLM guidance).",
            "architecture": {
                "extract_types": encode_subset,
                "storage_routing": routing,
                "retrieval": retrieval,
                "management": rng.choice(valid_mgmt),
            },
            "diversity_role": f"random_{cand_id}",
            "rationale": f"Uniform random sample from architecture space "
                         f"(round={round_id}).",
        })
    return out


def _round1_hardcoded_seeds() -> List[Dict[str, Any]]:
    """Method 4 calibration seeds for round 1.

    Three deterministic architectures spanning the search space's extremes
    (calibration probes — outcomes seed the §2F ledger before LLM proposing
    begins at R2):

      - seed_minimal:   simplest competent config — hybrid retrieval +
                        lightweight management; tip in json storage.
      - seed_balanced:  multi-Encode probe — tip+trajectory+workflow
                        extraction in hybrid storage with contrastive
                        retrieval (exercises the subset dimension of the
                        public space from round 1).
      - seed_graph:     graph-heavy config — graph storage + graph
                        retrieval + graph_consolidate management; outcome
                        depends on whether warmup grows the per-type pool
                        past graph's cold-start threshold (30).

    Round 1 does NOT invoke the LLM proposer. R2+ uses the LLM with these
    three reference points (and the ledger built from them) as context.
    """
    return [
        {
            "candidate_id": 0,
            "hypothesis": "Method-4 seed_minimal: simplest hybrid-retrieval baseline.",
            "diversity_role": "seed_minimal",
            "architecture": {
                "extract_types": ["tip"],
                "storage_routing": {"tip": "json"},
                "retrieval": "hybrid",
                "management": "lightweight",
            },
            "rationale": (
                "R1 calibration seed (Method 4). All json storage clears "
                "cold-start; hybrid retrieval covers both lexical and semantic "
                "matching as one of its paths. This is the simplest competent "
                "config in the v3 search space."
            ),
        },
        {
            "candidate_id": 1,
            "hypothesis": (
                "Method-4 seed_balanced: multi-Encode + contrastive retrieval probe."
            ),
            "diversity_role": "seed_balanced",
            "architecture": {
                "extract_types": ["tip", "trajectory", "workflow"],
                "storage_routing": {
                    "tip": "hybrid",
                    "trajectory": "hybrid",
                    "workflow": "hybrid",
                },
                "retrieval": "contrastive",
                "management": "lightweight",
            },
            "rationale": (
                "R1 calibration seed (Method 4). Tests a multi-Encode subset "
                "(tip+trajectory+workflow, the shape of previously discovered "
                "routes) with all types routed to one hybrid store, and "
                "contrastive retrieval on that embedding base. Calibrates how "
                "subset encoding trades off against the single-type seeds."
            ),
        },
        {
            "candidate_id": 2,
            "hypothesis": "Method-4 seed_graph: graph-heavy stress test.",
            "diversity_role": "seed_graph",
            "architecture": {
                "extract_types": ["insight"],
                "storage_routing": {"insight": "graph"},
                "retrieval": "graph",
                "management": "graph_consolidate",
            },
            "rationale": (
                "R1 calibration seed (Method 4). All graph storage + graph "
                "retrieval + graph_consolidate management. Graph storage has "
                "a cold-start threshold of 30 units, so this seed's outcome "
                "depends on whether warmup grows the per-type pool past that "
                "threshold. Outcome is empirical — measured at R1, not "
                "pre-judged."
            ),
        },
    ]


def _fallback_candidates(round_id: int, n: int) -> List[Dict[str, Any]]:
    """Generate simple fallback candidates when LLM fails.

    All architectures here use only options from RECOMMENDED_ARCHITECTURE_SPACE
    (4-tuple: extract_types / storage_routing / retrieval / management).
    Injection is fixed to raw_topk at the runtime layer, not selected here.
    """
    defaults = [
        {
            "candidate_id": 0, "hypothesis": "Fallback exploit candidate.",
            "diversity_role": "exploit",
            "architecture": {
                "extract_types": ["tip"],
                "storage_routing": {"tip": "json"},
                "retrieval": "hybrid", "management": "lightweight",
            },
            "rationale": "Fallback — LLM generation failed.",
        },
        {
            "candidate_id": 1, "hypothesis": "Fallback explore-retrieval candidate.",
            "diversity_role": "explore_retrieval",
            "architecture": {
                "extract_types": ["trajectory"],
                "storage_routing": {"trajectory": "json"},
                "retrieval": "cbr_rerank", "management": "lightweight",
            },
            "rationale": "Fallback — LLM generation failed.",
        },
        {
            "candidate_id": 2, "hypothesis": "Fallback explore-extract candidate.",
            "diversity_role": "explore_extract",
            "architecture": {
                "extract_types": ["tip", "workflow"],
                "storage_routing": {"tip": "vector", "workflow": "vector"},
                "retrieval": "hybrid", "management": "lightweight",
            },
            "rationale": "Fallback — LLM generation failed.",
        },
    ]
    import copy

    candidates = []
    for candidate_id in range(n):
        candidate = copy.deepcopy(defaults[candidate_id % len(defaults)])
        candidate["candidate_id"] = candidate_id
        if candidate_id >= len(defaults):
            candidate["diversity_role"] += f"_{candidate_id}"
        candidates.append(candidate)
    return candidates


# ======================================================================
# Champion injection (elitism) — bounds per-round best from below so the
# trajectory cannot regress from round to round under eval noise.
# ======================================================================

def _make_champion_candidate(
    pareto: ParetoFront,
    round_id: int,
    prev_champion_acc: Optional[float] = None,
    batch_size: int = 50,
    champion_override: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return a 'champion' candidate dict that re-evaluates the current Pareto
    best architecture unchanged. Used in round_id >= 2 to guarantee the per-
    round best is non-decreasing (modulo evaluation noise on the SAME arch).

    Returns None if the Pareto front is empty (e.g., round 1 before any
    candidate has been scored).

    IncumbentGate (H-plan, 2026-05-13): when `prev_champion_acc` is provided,
    consult the gate to check whether the new Pareto best (challenger) is
    *statistically* better than the previous champion. The current Pareto
    `best()` is always returned for elitism purposes (we still want to
    re-evaluate it), but a `champion_gate` field is added to the candidate
    dict noting whether the leadership change cleared the noise gate. This
    is a soft signal — downstream consumers (ledger, summary) can use it
    to flag noise-driven Pareto flips without breaking the round.
    """
    # Protocol-v2 A2: champion lineage override from champion_state.json.
    # Succession was already gated by the per-task sign test at the end of
    # the previous round, so the soft IncumbentGate check is skipped.
    if champion_override is not None and champion_override.get("architecture"):
        return {
            "candidate_id": 0,
            "hypothesis": (
                "Champion re-evaluation: rerun the paired-acceptance champion "
                "on this round's task fold. Provides elitism plus a fresh "
                "measurement for the champion's pooled estimate."
            ),
            "diversity_role": "champion",
            "architecture": dict(champion_override["architecture"]),
            "champion_gate": {
                "applied": False,
                "mode": "paired_acceptance",
                "lineage_config_id": champion_override.get("config_id"),
                "since_round": champion_override.get("since_round"),
            },
            "rationale": (
                f"Auto-injected by elitism schedule for round {round_id}. "
                f"Architecture follows champion_state.json "
                f"(config_id={champion_override.get('config_id')}, champion "
                f"since round {champion_override.get('since_round')}; "
                f"succession is gated by a per-task paired sign test)."
            ),
        }

    best = pareto.best()
    if best is None:
        return None

    gate_info: Dict[str, Any] = {"applied": False}
    if prev_champion_acc is not None and round_id >= 3:
        # Skip on R2: no prior champion to compare against (R1 had no champion).
        try:
            from automem.search.incumbent_gate import IncumbentGate
            gate = IncumbentGate(alpha=0.10, min_lift=0.02)
            decision = gate.is_challenger_significant(
                incumbent_acc=float(prev_champion_acc),
                challenger_acc=float(best.accuracy),
                n_tasks=int(batch_size),
            )
            gate_info = {
                "applied": True,
                "prev_champion_acc": float(prev_champion_acc),
                "current_pareto_best_acc": float(best.accuracy),
                "promote_passed": bool(decision.promote),
                "delta_acc": float(decision.delta_acc),
                "p_value": float(decision.p_value) if decision.p_value is not None else None,
                "reason": decision.reason,
            }
            if not decision.promote and decision.delta_acc > 0:
                logger.warning(
                    "[incumbent_gate] R%d champion change is noise-suspect: "
                    "%s",
                    round_id, decision.reason,
                )
        except Exception as _e:
            logger.warning("[incumbent_gate] check raised %s — skipping.", _e)

    return {
        "candidate_id": 0,
        "hypothesis": (
            "Champion re-evaluation: bound per-round best from below by "
            "rerunning the current Pareto best on this round's task batch. "
            "Provides elitism + variance reading on the leading architecture."
        ),
        "diversity_role": "champion",
        "architecture": dict(best.architecture),
        "champion_gate": gate_info,
        "rationale": (
            f"Auto-injected by elitism schedule for round {round_id}. "
            f"Architecture mirrors the current Pareto best "
            f"(prior config_id={best.config_id}, "
            f"prior_accuracy={best.accuracy:.3f}, "
            f"prior_fitness={best.fitness:.3f}). No mutation; sole purpose is "
            f"to bound the per-round-best trajectory from below."
        ),
    }


# ======================================================================
# Batch-consistency self-healing (FB1 + FB2)
# ======================================================================

def _ensure_batch_complete(
    tasks_dir: Path,
    expected_indices: List[int],
    arch: Dict[str, Any],
    cand_dir: Path,
    args: argparse.Namespace,
    config_id: str,
    canonical_source_dir: Path,
    max_retries: int = 1,
) -> Tuple[bool, List[int], List[int]]:
    """Verify exact outputs, rebuilding a partial stateful candidate if needed.

    Missing-only retries are invalid for an evolving memory store: a missing
    task may already have written memory, and later tasks may also have run.
    Every retry therefore removes the candidate and replays the full batch from
    the immutable round-start canonical snapshot.

    Returns (complete, missing_remaining, extras):
      - complete: True iff the directory exactly represents the expected set
      - missing_remaining: indices still missing after retry attempts
      - extras: indices present in tasks_dir that are NOT in expected
    """
    expected_set = set(expected_indices)

    def _scan_present() -> Tuple[set[int], List[int], List[str]]:
        return _scan_task_result_indices(
            tasks_dir, getattr(args, "_task_dataset_sha256", None)
        )

    for attempt in range(max_retries + 1):
        present, duplicates, invalid_files = _scan_present()
        missing = sorted(expected_set - present)
        extras = sorted(present - expected_set)
        if not missing and not extras and not duplicates and not invalid_files:
            return True, [], []
        if attempt >= max_retries:
            return False, missing, extras
        logger.warning(
            "%s: non-exact candidate outputs (missing=%s extras=%s duplicate=%s "
            "invalid=%s). Clean full-batch retry %d/%d.",
            config_id,
            missing,
            extras,
            duplicates,
            invalid_files,
            attempt + 1,
            max_retries,
        )
        import shutil as _shutil

        _shutil.rmtree(cand_dir)
        tasks_dir.mkdir(parents=True, exist_ok=True)
        retry_runtime = _compile_architecture(arch, str(cand_dir / "storage"))
        if retry_runtime is None:
            logger.error(
                "%s: cannot recompile architecture for clean retry.",
                config_id,
            )
            return False, missing, extras
        if not getattr(args, "no_canonical_import", False):
            import_canonical_to_storage(canonical_source_dir, retry_runtime)
        retry_proc = _start_eval_subprocess(
            tasks_dir=tasks_dir,
            task_indices=list(expected_indices),
            runtime_config=retry_runtime,
            args=args,
        )
        if retry_proc is None:
            return False, missing, extras
        retry_proc.wait()
        if hasattr(retry_proc, "_log_file"):
            retry_proc._log_file.close()
        if retry_proc.returncode != 0:
            logger.error(
                "%s: retry eval subprocess exit=%d; gap may persist.",
                config_id, retry_proc.returncode,
            )
            # Fall through to one final exact scan.

    # Final post-loop scan
    present, duplicates, invalid_files = _scan_present()
    missing = sorted(expected_set - present)
    extras = sorted(present - expected_set)
    return (
        not missing and not extras and not duplicates and not invalid_files,
        missing,
        extras,
    )


# ======================================================================
# Context builder for next round
# ======================================================================

def build_last_round_context(
    round_summary: Dict[str, Any],
    pareto: ParetoFront,
) -> Dict[str, Any]:
    """Build last_round_details dict for LLM context."""
    if not round_summary:
        return {"note": "No previous round."}

    return {
        "round_id": round_summary.get("round_id", -1),
        "batch_size": round_summary.get("batch_size", 0),
        "pareto_front_size_after": round_summary.get("pareto_front_size", 0),
        "best_fitness_after": round_summary.get("best_fitness", 0.0),
        # Smart-4 fix (2026-05-16): surface differential diagnosis so the
        # proposer can act on the best-vs-worst gap explanation.
        "differential_diagnosis": round_summary.get("differential_diagnosis", {}),
        # Smart-9b (2026-05-17): round-level synthesized verdict. The proposer
        # reads this FIRST as the strategic directive for the next round.
        "round_level_verdict": round_summary.get("round_level_verdict", {}),
        "candidates": [
            {
                "config_id": r.get("config_id"),
                "hypothesis": r.get("hypothesis", ""),
                "diversity_role": r.get("diversity_role", ""),
                "architecture": r.get("architecture", {}),
                "metrics": r.get("metrics", {}),
                # E11 fix (2026-05-16): prefer new field name; fall back to alias for old runs.
                # Smart-9a (2026-05-17): synthesized natural-language verdict.
                # This is the PRIMARY signal the proposer should read first.
                "synthesized_verdict": r.get("attribution_summary", {}).get("synthesized_verdict", {}),
                "attribution_diagnosis": (
                    r.get("attribution_summary", {}).get("rule_diagnosis", "")
                    or r.get("attribution_summary", {}).get("diagnosis", "")
                ),
                "layer_diagnosis": r.get("attribution_summary", {}).get("layer_diagnosis", {}),
                "breakdown": r.get("attribution_summary", {}).get("breakdown", {}),
                # G7 fix (codex review, 2026-05-16): surface memory_compliance
                # so the proposer can see "agent ignores extracted instructions"
                # patterns that would otherwise be invisible.
                "memory_compliance": r.get("attribution_summary", {}).get("memory_compliance", {}),
                "added_to_front": r.get("added_to_front", False),
            }
            for r in round_summary.get("candidate_results", [])
            if not r.get("skipped")
        ],
    }


# ──────────────────────────────────────────────────────────────────────────
# Removed `update_cumulative_principles` (Codex M2 fix, 2026-05-09).
# The freeform-string accumulator was replaced by the structured Experience
# Ledger (`automem.search.experience_ledger.ExperienceLedger`). The ledger
# is updated per round by the diagnosis LLM (gpt-5.5) and rendered into the
# proposer prompt's §2F via `ledger.render_for_prompt()`.
# ──────────────────────────────────────────────────────────────────────────


# ======================================================================
# Final validation
# ======================================================================

def run_final_runoff(
    run_dir: Path,
    pareto: ParetoFront,
    union_indices: List[int],
    baseline_stats: Dict,
    args: argparse.Namespace,
    k: int,
) -> Optional[Dict[str, Any]]:
    """Protocol-v2 M3: runoff among the top-K distinct architectures.

    The single pareto.best() point is selected by max over noisy 30-50 task
    draws, so handing it straight to final validation reports a winner's-
    curse-inflated architecture. The runoff re-evaluates the top-K distinct
    architectures (paired-acceptance champion always included) on the FULL
    fold union in one shot and picks the winner by runoff accuracy
    (fitness as tiebreaker). Results land in final_runoff/.
    """
    from automem.search import protocol as _p2

    champion_state = _p2.load_champion_state(run_dir)
    contenders = _p2.select_runoff_contenders(pareto, champion_state, k)
    if not contenders:
        raise RuntimeError("Final runoff requires at least one contender")
    if not union_indices:
        raise RuntimeError("Final runoff requires a non-empty evaluation fold union")
    if len(contenders) == 1:
        logger.info("Final runoff: single contender %s; selecting it directly.",
                    contenders[0]["config_id"])

    runoff_dir = run_dir / "final_runoff"
    runoff_dir.mkdir(parents=True, exist_ok=True)
    runoff_start_dir = runoff_dir / "runoff_start_state"
    runoff_snapshot_digest: Optional[str] = None
    if not args.dry_run:
        runoff_start_path = canonical_pool_path(runoff_start_dir)
        runoff_manifest_path = runoff_start_dir / "snapshot_manifest.json"
        existing_contenders = list(runoff_dir.glob("contender_*"))
        if runoff_start_path.exists() and runoff_manifest_path.exists():
            _, runoff_snapshot_digest = _load_bound_canonical_snapshot(
                runoff_start_dir
            )
        else:
            if existing_contenders:
                raise RuntimeError(
                    "Final runoff has contender state but no immutable canonical "
                    "snapshot; refusing an inexact resume"
                )
            if runoff_start_dir.exists():
                import shutil as _shutil

                _shutil.rmtree(runoff_start_dir)
            runoff_snapshot_digest = _save_bound_canonical_snapshot(
                run_dir, runoff_start_dir
            )
    results: List[Dict[str, Any]] = []

    for i, cont in enumerate(contenders):
        tag = f"contender_{i}_{cont['config_id']}"
        cont_dir = runoff_dir / tag
        tasks_dir = cont_dir / "tasks"
        reuse_completed = False
        if not args.dry_run:
            reuse_completed = _reuse_or_reset_stateful_stage(
                tasks_dir=tasks_dir,
                expected_indices=list(union_indices),
                reset_paths=[cont_dir],
                stage=f"Final runoff {tag}",
                expected_dataset_sha256=getattr(
                    args, "_task_dataset_sha256", None
                ),
            )
        tasks_dir.mkdir(parents=True, exist_ok=True)
        runtime_config = _compile_architecture(cont["architecture"], str(cont_dir / "storage"))
        if runtime_config is None:
            raise RuntimeError(f"Final runoff failed to compile {tag}")
        if (
            not args.dry_run
            and not reuse_completed
            and not getattr(args, "no_canonical_import", False)
        ):
            import_canonical_to_storage(runoff_start_dir, runtime_config)
        if not args.dry_run and not reuse_completed:
            ok = _run_eval_subprocess(
                tasks_dir=tasks_dir,
                task_indices=list(union_indices),
                runtime_config=runtime_config,
                args=args,
            )
            if not ok:
                raise RuntimeError(f"Final runoff subprocess failed for {tag}")
            _require_exact_task_results(
                tasks_dir,
                list(union_indices),
                f"Final runoff {tag}",
                getattr(args, "_task_dataset_sha256", None),
            )
        elif not args.dry_run:
            _require_exact_task_results(
                tasks_dir,
                list(union_indices),
                f"Final runoff {tag}",
                getattr(args, "_task_dataset_sha256", None),
            )
        if not args.dry_run:
            try:
                fitness_score, raw_metrics = compute_candidate_fitness(
                    tasks_dir=tasks_dir,
                    baseline_accuracy=baseline_stats.get("baseline_accuracy", 0.0),
                    baseline_per_task=baseline_stats.get("per_task_scores"),
                    architecture=cont["architecture"],
                    token_cap_per_task=args.token_cap_per_task,
                    latency_cap_per_task=args.latency_cap_per_task,
                    pool_units=load_canonical_pool(run_dir),
                )
            except FitnessComputationError as fe:
                raise RuntimeError(f"Final runoff fitness failed for {tag}: {fe}") from fe
        else:
            fitness_score, raw_metrics = _synthetic_candidate_metrics(
                cont["architecture"],
                round_id=args.max_rounds + 1,
                candidate_id=i,
                total_tasks=len(union_indices),
                baseline_accuracy=baseline_stats.get("baseline_accuracy", 0.0),
            )
        results.append({
            "config_id": cont["config_id"],
            "source": cont.get("source"),
            "architecture": cont["architecture"],
            "runoff_metrics": raw_metrics,
            "runoff_fitness": fitness_score,
        })
        logger.info("Final runoff: %s -> acc=%.3f fit=%.3f",
                    tag, raw_metrics.get("accuracy", 0.0), fitness_score)

    winner = max(results, key=lambda r: (
        float(r["runoff_metrics"].get("accuracy", 0.0)),
        float(r["runoff_fitness"]),
    ))
    payload = {
        "n_contenders": len(results),
        "union_size": len(union_indices),
        "winner_config_id": winner["config_id"],
        "canonical_snapshot": (
            "runoff_start_state/canonical/pool.json" if not args.dry_run else None
        ),
        "canonical_snapshot_sha256": runoff_snapshot_digest,
        "contenders": results,
    }
    with open(runoff_dir / "runoff_result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Final runoff winner: %s (acc=%.3f over %d tasks)",
                winner["config_id"],
                winner["runoff_metrics"].get("accuracy", 0.0), len(union_indices))
    return winner


def run_protocol_runoff(
    run_dir: Path,
    pareto: ParetoFront,
    union_indices: List[int],
    baseline_stats: Dict[str, Any],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Execute mandatory M3 selection from the immutable public protocol."""
    runoff_k = args._protocol.final_runoff
    if runoff_k <= 0:
        raise RuntimeError("fixed evaluation protocol must enable final runoff")
    return run_final_runoff(
        run_dir,
        pareto,
        union_indices,
        baseline_stats,
        args,
        k=runoff_k,
    )


def run_final_validation(
    run_dir: Path,
    pareto: ParetoFront,
    split: DataSplitConfig,
    baseline_stats: Dict,
    args: argparse.Namespace,
    best_override: Optional[Dict[str, Any]] = None,
) -> None:
    """Evaluate the runoff winner once on the held-out final-test split."""
    best = pareto.best()
    if best_override is not None and best_override.get("architecture"):
        # Protocol-v2 M3: final selection decided by the runoff, not by the
        # (max-biased) single front leader.
        from types import SimpleNamespace
        best = SimpleNamespace(
            config_id=str(best_override.get("config_id", "runoff_winner")),
            architecture=dict(best_override["architecture"]),
            raw_metrics=dict(best_override.get("runoff_metrics", {})),
        )
    if best is None:
        raise RuntimeError("Pareto front is empty; cannot run final validation")

    final_indices = list(split.final_test_indices)
    if not final_indices:
        raise RuntimeError("Final-test split is empty; cannot run final validation")

    val_dir = run_dir / "final_validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    baseline_tasks_dir = val_dir / "baseline_tasks"
    baseline_tasks_dir.mkdir(parents=True, exist_ok=True)
    storage_path = val_dir / "storage"
    storage_dir = str(storage_path)
    tasks_dir = val_dir / "tasks"
    reuse_memory_results = False
    if not args.dry_run:
        reuse_memory_results = _reuse_or_reset_stateful_stage(
            tasks_dir=tasks_dir,
            expected_indices=final_indices,
            reset_paths=[tasks_dir, storage_path],
            stage="Final-test memory candidate",
            expected_dataset_sha256=getattr(args, "_task_dataset_sha256", None),
        )
    tasks_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Final validation on %d held-out tasks: best config=%s, arch=%s",
        len(final_indices),
        best.config_id,
        best.architecture,
    )

    # Evaluate the no-memory control on the final-test subset only now, after
    # architecture selection, so held-out outcomes cannot influence search.
    if args.dry_run:
        final_baseline_results = [
            {
                "task_id": f"dry-{index}",
                "item_index": index + 1,
                "task_score": float((index * 17 + 3) % 5 != 0),
            }
            for index in final_indices
        ]
    else:
        ok = _run_eval_subprocess(
            tasks_dir=baseline_tasks_dir,
            task_indices=final_indices,
            runtime_config=None,
            args=args,
            extra_flags=[],
        )
        if not ok:
            raise RuntimeError("Final-test no-memory baseline subprocess failed")
        _require_exact_task_results(
            baseline_tasks_dir,
            final_indices,
            "Final-test no-memory baseline",
            getattr(args, "_task_dataset_sha256", None),
        )
        final_baseline_results = load_task_results(str(baseline_tasks_dir))

    final_baseline_scores: Dict[str, float] = {}
    for row in final_baseline_results:
        score = float(row.get("task_score", 0.0) or 0.0)
        final_baseline_scores[str(row["item_index"])] = score
    final_baseline_accuracy = (
        sum(float(row.get("task_score", 0.0) or 0.0) for row in final_baseline_results)
        / len(final_baseline_results)
    )

    runtime_config = _compile_architecture(best.architecture, storage_dir)
    if runtime_config is None:
        raise RuntimeError("Failed to compile best architecture for final validation")

    if (
        not args.dry_run
        and not reuse_memory_results
        and not getattr(args, "no_canonical_import", False)
    ):
        import_canonical_to_storage(run_dir, runtime_config)

    if not args.dry_run and not reuse_memory_results:
        ok = _run_eval_subprocess(
            tasks_dir=tasks_dir,
            task_indices=final_indices,
            runtime_config=runtime_config,
            args=args,
        )
        if not ok:
            raise RuntimeError("Final-test memory candidate subprocess failed")
        _require_exact_task_results(
            tasks_dir,
            final_indices,
            "Final-test memory candidate",
            getattr(args, "_task_dataset_sha256", None),
        )
    elif not args.dry_run:
        _require_exact_task_results(
            tasks_dir,
            final_indices,
            "Final-test memory candidate",
            getattr(args, "_task_dataset_sha256", None),
        )

    final_pool = load_canonical_pool(run_dir)
    if args.dry_run:
        fitness_score, raw_metrics = _synthetic_candidate_metrics(
            best.architecture,
            round_id=args.max_rounds + 2,
            candidate_id=0,
            total_tasks=len(final_indices),
            baseline_accuracy=final_baseline_accuracy,
        )
        synthetic_passes = min(
            len(final_indices),
            max(0, int(round(raw_metrics["accuracy"] * len(final_indices)))),
        )
        raw_metrics["accuracy"] = synthetic_passes / len(final_indices)
        raw_metrics["memory_lift"] = (
            raw_metrics["accuracy"] - final_baseline_accuracy
        )
        if "hit_rate" in raw_metrics and "token_eff" in raw_metrics:
            fitness_score = round(
                0.70 * raw_metrics["accuracy"]
                + 0.20 * raw_metrics["hit_rate"]
                + 0.10 * raw_metrics["token_eff"],
                4,
            )
            raw_metrics["fitness"] = fitness_score
    else:
        try:
            fitness_score, raw_metrics = compute_candidate_fitness(
                tasks_dir=tasks_dir,
                baseline_accuracy=final_baseline_accuracy,
                baseline_per_task=final_baseline_scores,
                architecture=best.architecture,
                token_cap_per_task=args.token_cap_per_task,
                latency_cap_per_task=args.latency_cap_per_task,
                pool_units=final_pool,
            )
        except FitnessComputationError as fe:
            raise RuntimeError(
                f"Final validation fitness computation failed: {fe}"
            ) from fe

    # Per-task pass/fail summary so the user can analyze which specific
    # tasks failed in the final-validation run.
    if args.dry_run:
        per_task_summary = [
            {
                "task_id": f"dry-{index}",
                "item_index": index + 1,
                "level": None,
                "category": "unknown",
                "score": 1.0 if position < synthetic_passes else 0.0,
                "passed": position < synthetic_passes,
                "baseline_score": final_baseline_scores.get(str(index + 1)),
                "lift": None,
                "status": "synthetic",
                "question": "",
                "golden_answer": "",
                "agent_answer": "",
            }
            for position, index in enumerate(final_indices)
        ]
        for row in per_task_summary:
            if row["baseline_score"] is not None:
                row["lift"] = round(row["score"] - row["baseline_score"], 4)
    else:
        per_task_summary = _build_per_task_pass_fail_summary(
            tasks_dir,
            baseline_per_task=final_baseline_scores,
        )

    result = {
        "best_config_id": best.config_id,
        "architecture": best.architecture,
        "evaluation_split": "final_test",
        "final_test_indices": final_indices,
        "search_metrics": best.raw_metrics,
        "final_test_metrics": raw_metrics,
        "final_test_fitness": fitness_score,
        "baseline_accuracy": final_baseline_accuracy,
        "improvement_over_baseline": round(
            raw_metrics.get("accuracy", 0.0) - final_baseline_accuracy, 4
        ),
        "n_passed": sum(1 for r in per_task_summary if r["passed"]),
        "n_failed": sum(1 for r in per_task_summary if not r["passed"]),
        "per_task_results": per_task_summary,
    }

    with open(val_dir / "validation_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info(
        "Final validation: acc=%.4f (baseline=%.4f, lift=%.4f)",
        raw_metrics.get("accuracy", 0.0),
        final_baseline_accuracy,
        result["improvement_over_baseline"],
    )


# ======================================================================
# Main
# ======================================================================

def main() -> None:
    args = parse_args()
    _apply_benchmark_split_defaults(args)
    _validate_search_args(args)
    if not args.dry_run and args.concurrency != 1:
        raise ValueError(
            "Architecture search requires --concurrency 1 because candidate "
            "tasks share an evolving memory pool; concurrent completion order "
            "changes the evaluated system. Run standalone no-memory benchmarks "
            "separately if parallel throughput is required."
        )

    # ------------------------------------------------------------------
    # Resolve the single fixed public evaluation protocol.
    # ------------------------------------------------------------------
    from automem.search.protocol import ProtocolConfig
    _proto = ProtocolConfig.resolve(args)
    args.val_every = _proto.val_every
    args._protocol = _proto

    run_dir = setup_run_dir(args)
    logger.info(
        "[evaluation_protocol] %s",
        json.dumps(_proto.to_dict(), ensure_ascii=False),
    )

    # Cross-platform kernel-backed run lock. Keep the FileLock object alive
    # for the process lifetime; release is also automatic on process exit.
    from filelock import FileLock, Timeout as FileLockTimeout

    active_lock = run_dir / ".run_active.lock"
    run_lock = FileLock(str(active_lock))
    try:
        run_lock.acquire(timeout=0)
    except FileLockTimeout:
        logger.error(
            "Another AutoMem process holds the run-dir lock at %s. "
            "Refusing to start a second instance.",
            active_lock,
        )
        sys.exit(2)
    import atexit as _atexit
    def _release_lock():
        try:
            run_lock.release()
        except Exception:
            pass
    _atexit.register(_release_lock)

    logger.info("=== AutoMem Run: %s ===", args.run_name)
    logger.info("  max_rounds=%d, num_candidates=%d, batch_size=%d",
                args.max_rounds, args.num_candidates, args.batch_size)

    # Load tasks and create splits
    tasks = load_tasks(args.infile)
    args._loaded_tasks = tasks
    args._task_dataset_sha256 = dataset_file_sha256(args.infile)
    split = create_or_load_splits(run_dir, tasks, args)
    args._resolved_split = split.to_dict()
    if len(split.optimization_indices) < _proto.fold_rotation:
        raise ValueError(
            "The optimization split must contain at least "
            f"{_proto.fold_rotation} tasks for the fixed fold-rotation protocol"
        )

    # Model initialization is deferred until search_state is loaded. A completed
    # run may be resumed solely to append held-out final validation and should
    # not require proposer/diagnosis endpoints that will never be called.
    model = None
    diagnosis_model = None

    # Codex Q7-A1 fix (2026-04-28): detect protocol mismatch BEFORE
    # baseline/warmup. The previous Q6-A4 fix only invalidated Pareto
    # and search_state — but baseline_done.json, warmup state, and
    # round_done.json markers all guard themselves with `args.resume`,
    # so a resume after prompt changes would silently REUSE the old
    # baseline accuracy (computed under old prompts) and old per-round
    # candidate evals. Now: compute protocol up-front; on mismatch,
    # force args.resume=False for this entire run so every checkpoint
    # is recomputed under the new protocol.
    #
    # Codex Q8 audit (2026-04-28):
    #   • Q8-A2: missing protocol_path on a resume that already has
    #     baseline/warmup/search checkpoints means the run was created
    #     before eval_protocol.json existed. Treat it as mismatch and
    #     do the full cleanup, otherwise old prompt outputs are
    #     trusted as the new protocol.
    #   • Q8-A1: GAIA eval runner (run_flash_searcher_mm_gaia.py:822)
    #     skips per-task JSON files if they already exist. Deleting
    #     done-markers is not enough — must also wipe the tasks/
    #     directories so reruns regenerate per-task results under the
    #     new protocol.
    #   • Q8-A3: per-round folders are at `run_dir / round_<id>` — the
    #     glob `rounds/round_*/round_done.json` matched nothing. Use
    #     the correct path layout.
    protocol_path = run_dir / "eval_protocol.json"
    current_protocol = _compute_eval_protocol_signature(
        eval_model=(getattr(args, "model", "") or "").strip(),
        protocol=getattr(args, "_protocol", None),
        args=args,
    )

    def _has_resume_artifacts(rd: Path) -> bool:
        """Q8-A2 + Q9-A1: check if any resume checkpoint or partial
        task output is already on disk. The GAIA eval runner skips
        existing per-task JSONs, so a legacy run that crashed BEFORE
        writing baseline_done.json but AFTER writing some
        baseline/tasks/*.json still poisons reruns. Treat any of these
        as "this run has protocol-dependent state".
        """
        # 1. Done markers (high-confidence signal).
        for m in (
            rd / "baseline" / "baseline_done.json",
            rd / "warmup" / "warmup_done.json",
            rd / "search_state.json",
            rd / "pareto_front.json",
        ):
            if m.exists():
                return True
        for _ in rd.glob("round_*/round_done.json"):
            return True
        # 2. Partial per-task outputs (Q9-A1 fix). Check for any *.json
        # under baseline/tasks, warmup/tasks, candidate tasks, or
        # validation tasks. These are protocol-dependent.
        for partial_glob in (
            "baseline/tasks/*.json",
            "warmup/tasks/*.json",
            "round_*/candidate_*/tasks/*.json",
            "validation_checkpoints/*/tasks/*.json",
            "final_validation/tasks/*.json",
        ):
            for _ in rd.glob(partial_glob):
                return True
        # 3. Persisted memory artifacts (canonical pool, storage dirs)
        # also encode protocol-specific extraction prompts.
        # Codex Q10-A3 fix (2026-04-28): the previous list `vector.faiss`
        # was wrong — JsonStorage writes `memory_db.json`, HybridStorage
        # writes `faiss.index`, GraphStorage writes `graph.json`,
        # VectorStorage writes `metadata.json`. Without these, a legacy
        # run with only vector-store artifacts skipped mismatch
        # detection.
        if (rd / "canonical" / "pool.json").exists():
            return True
        for storage_file in (
            "memory_db.json",
            "faiss.index",
            "graph.json",
            "metadata.json",
        ):
            for _ in rd.glob(f"**/{storage_file}"):
                return True
        return False

    saved_protocol = None
    if protocol_path.exists():
        try:
            with open(protocol_path) as _f:
                saved_protocol = json.load(_f)
        except Exception:
            saved_protocol = None
    protocol_mismatch = False
    if args.resume:
        # Q8-A2: missing protocol_path AND existing resume artifacts =
        # mismatch (legacy run from before signatures were tracked).
        # The current_protocol value will be installed once cleanup is done.
        if saved_protocol is None and _has_resume_artifacts(run_dir):
            logger.warning(
                "Resume requested but eval_protocol.json is missing while "
                "baseline / warmup / round checkpoints exist. Treating as "
                "protocol mismatch (Codex Q8-A2 fix)."
            )
            protocol_mismatch = True
        elif saved_protocol is not None and (
            (saved_protocol or {}).get("digest") != current_protocol.get("digest")
        ):
            # Compare the DIGEST only (2026-07-11). Every material protocol
            # input (prompts, models, runtime policy, fixed protocol) is folded
            # into the digest; the signature's other keys are
            # descriptive. The old full-dict inequality meant that ADDING any
            # informational key (e.g. runtime_info) — or a descriptive field
            # drifting while the digest matched — wiped every pre-existing
            # run's rounds on resume even though the protocol was identical.
            logger.warning(
                "Eval protocol changed since the run was started "
                "(prompt/model/op digest mismatch). Forcing resume=False. "
                "Saved=%s current=%s (Codex Q7-A1 fix).",
                (saved_protocol or {}).get("digest"),
                current_protocol.get("digest"),
            )
            protocol_mismatch = True

    if protocol_mismatch:
        # Codex Q9-A4 + Q12-A1 fix (2026-04-28): protocol cleanup is
        # safe-by-construction now that main() acquired an exclusive
        # cross-platform FileLock on `.run_active.lock` before this point.
        # No other AutoMem process can be writing into this run_dir.
        # The previous PID-file probe had a TOCTOU race.
        logger.info(
            "Protocol mismatch: running cleanup under exclusive run-dir "
            "lock (no concurrent writers possible)."
        )
        args.resume = False
        # Delete stale done markers AND the corresponding tasks/ dirs.
        # GAIA eval runner skips existing task json files unconditionally
        # (Q8-A1), so leaving tasks/ in place would silently reuse stale
        # per-task outputs under the new protocol.
        # Codex Q9-A2 fix (2026-04-28): also clean canonical/pool.json
        # and final_validation/. Otherwise run_warmup delta-merges new
        # extractions into the stale canonical pool, and final
        # validation rerun reuses old per-task outputs.
        import shutil as _shutil
        cleanup_targets = [
            run_dir / "baseline",
            run_dir / "warmup",
            run_dir / "search_state.json",
            run_dir / "pareto_front.json",
            run_dir / "data_split.json",
            run_dir / "search_batch.json",
            run_dir / "search_folds.json",
            run_dir / "validation_checkpoints",
            run_dir / "canonical",  # Q9-A2: pool.json is protocol-dependent
            run_dir / "final_validation",  # Q9-A2: stale per-task outputs
            run_dir / "final_runoff",
            run_dir / "champion_state.json",
            run_dir / "ledger",
            # Observation graph is protocol-dependent too: a graph built under
            # the old prompt/model must not feed the Proposer after a reset
            # (would contaminate the on/off ablation). Codex P2 fix 2026-05-20.
            run_dir / "observation_graph.json",
        ]
        for tgt in cleanup_targets:
            if tgt.is_dir() and not tgt.is_symlink():
                _shutil.rmtree(tgt)
                logger.warning("  wiped stale dir %s", tgt.name)
            elif tgt.exists() or tgt.is_symlink():
                tgt.unlink()
                logger.warning("  removed stale %s", tgt.name)
        # Q8-A3: per-round folders are `run_dir / round_<id>` (not
        # `rounds/round_*`). Match the correct layout. Wipe whole
        # folders so per-task outputs under them are regenerated.
        for round_dir in run_dir.glob("round_*"):
            if not round_dir.is_dir():
                continue
            _shutil.rmtree(round_dir)
            logger.warning("  wiped stale %s", round_dir.name)
        # Split and batch files are protocol inputs. Recreate them under the
        # new settings before recording the replacement signature.
        split = create_or_load_splits(run_dir, tasks, args)
        args._resolved_split = split.to_dict()
        if len(split.optimization_indices) < _proto.fold_rotation:
            raise ValueError(
                "The optimization split must contain at least "
                f"{_proto.fold_rotation} tasks for the fixed fold-rotation protocol"
            )
        current_protocol = _compute_eval_protocol_signature(
            eval_model=(getattr(args, "model", "") or "").strip(),
            protocol=getattr(args, "_protocol", None),
            args=args,
        )
    # Persist current protocol signature ASAP (so subsequent runs know
    # which protocol generated baseline / warmup data).
    args._baseline_protocol_digest = current_protocol.get("baseline_digest")
    _atomic_write_json(protocol_path, current_protocol)

    # Phase 0: No-memory baseline
    baseline_stats = run_baseline(run_dir, split, args)

    # Phase 1: Warmup (seed canonical pool)
    run_warmup(run_dir, split, args)

    # Search batch: legacy = one fixed batch reused across rounds;
    # protocol-v2 M2 = stratified folds rotated per round (within-round
    # comparisons stay paired; the champion's pooled estimate spans folds).
    _proto = getattr(args, "_protocol", None)
    search_folds: Optional[List[List[int]]] = None
    if _proto is not None and _proto.fold_rotation > 1:
        search_folds = load_or_create_search_folds(
            run_dir, split, args, n_folds=_proto.fold_rotation, tasks=tasks,
        )
        search_batch_indices = sorted({i for f in search_folds for i in f})
        logger.info("Fold rotation ACTIVE: %d folds, union=%d tasks",
                    len(search_folds), len(search_batch_indices))
    else:
        # Fixed search batch (sampled once, reused across rounds)
        search_batch_indices = load_or_create_search_batch(run_dir, split, args, tasks=tasks)

    # Load/init Pareto front. With Q7-A1 forcing resume=False on
    # mismatch, this branch only succeeds for genuine same-protocol
    # resumes.
    pareto_path = run_dir / "pareto_front.json"
    if pareto_path.exists() and args.resume:
        pareto = ParetoFront.load(str(pareto_path))
        logger.info("Loaded Pareto front: %d entries, %d on front",
                    len(pareto.all_evaluated()), pareto.size())
    else:
        pareto = ParetoFront()
    # Protocol-v2 A1: pooled measurement mode (running mean per arch instead
    # of keeping the max-fitness draw). Set on create AND on load so resumed
    # legacy fronts adopt the run's configured mode.
    if _proto is not None and _proto.champion_scoring == "pooled":
        pareto.measurement_mode = "pooled"

    # Load search state for resume.
    state_path = run_dir / "search_state.json"
    state = {}
    if state_path.exists() and args.resume:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

    start_round = state.get("next_round", 1)
    last_round_summary = state.get("last_round_summary", {})
    if start_round <= args.max_rounds:
        model, diagnosis_model = _initialize_search_models(args)
    else:
        logger.info(
            "All search rounds are complete; skipping proposer and diagnosis "
            "model initialization."
        )

    # Initialise the Experience Ledger. On resume, it auto-loads from
    # {run_dir}/ledger/ledger.json. Disabled mode (--no_ledger) makes
    # render_for_prompt() return a placeholder and update_with_round() a no-op
    # — used for ablation comparison vs ledger-on runs.
    from automem.search.experience_ledger import ExperienceLedger
    ledger = ExperienceLedger(run_dir, no_ledger=args.no_ledger)

    # Smart-5 fix (2026-05-16): diversity-aware proposer. Track per-dimension
    # exploration coverage; surface under-tested options to the proposer LLM
    # so later rounds don't keep proposing minor variants of the Pareto sweet
    # spot. On resume, replay all evaluated architectures from the Pareto
    # history so the tracker reflects state at this point.
    from automem.search.exploration_tracker import ExplorationTracker
    from automem.architecture_space import RECOMMENDED_ARCHITECTURE_SPACE
    explorer = ExplorationTracker(RECOMMENDED_ARCHITECTURE_SPACE)

    # Observation Graph (optional, --obs_graph_enabled). Rule-based structural
    # experience map fed to the Proposer as extra context. Persisted to
    # observation_graph.json so it survives --resume. When disabled, obs_graph
    # stays None and the Proposer prompt is byte-for-byte the baseline.
    obs_graph = None
    if getattr(args, "obs_graph_enabled", False):
        from automem.observation.graph import ObservationGraph
        obs_graph = ObservationGraph(run_dir / "observation_graph.json")
        logger.info("Observation Graph ENABLED (nodes=%d, edges=%d at start)",
                    len(obs_graph.nodes), len(obs_graph.edges))
    try:
        for entry in pareto.all_evaluated():
            arch = getattr(entry, "architecture", None) or entry.get("architecture")
            if arch:
                explorer.update(arch)
    except Exception as _e:
        logger.warning("ExplorationTracker replay failed: %s", _e)
    ledger_prompt_path = str(SEARCH_PROMPT.parent / "ledger_update.txt")
    if not Path(ledger_prompt_path).exists():
        logger.warning(
            "[ledger] prompt template not found at %s — ledger update will fail "
            "until you create it. Search continues; principles stay empty.",
            ledger_prompt_path,
        )

    # Phase 2: Search loop
    for round_id in range(start_round, args.max_rounds + 1):
        logger.info("=== Search Round %d / %d ===", round_id, args.max_rounds)

        last_round_details = build_last_round_context(last_round_summary, pareto)

        # Render the ledger fresh each round — its content reflects the
        # latest update_with_round() output, so resume + cross-round eviction
        # are reflected automatically.
        cumulative_principles = ledger.render_for_prompt()

        # Pass forward the cumulative tracking from prior rounds so this
        # round's run_search_round can extend it (max accumulator + threshold
        # bookkeeping). On resume, state["cumulative_tracking"] survives.
        prev_cumulative_tracking = state.get("cumulative_tracking") if isinstance(state, dict) else None

        # Smart-5 fix (2026-05-16): render exploration hints for this round's
        # proposer prompt. Empty on R1 (no history).
        exploration_hints = explorer.render_hints(min_count=2, max_per_dim=5)

        # Observation Graph context for the Proposer (empty until it has data).
        obs_graph_json = ""
        if obs_graph is not None and not obs_graph.is_empty():
            try:
                obs_graph_json = obs_graph.to_proposer_json()
            except Exception as _e:
                logger.warning("ObservationGraph serialization failed: %s", _e)

        # Protocol-v2 M2: this round's fold (legacy: the fixed batch).
        round_batch_indices = search_batch_indices
        if search_folds:
            from automem.search.protocol import fold_for_round
            round_batch_indices = fold_for_round(search_folds, round_id)
            logger.info(
                "Round %d evaluates fold %d/%d (%d tasks).",
                round_id, (round_id - 1) % len(search_folds) + 1,
                len(search_folds), len(round_batch_indices),
            )

        pareto, round_summary = run_search_round(
            round_id=round_id,
            run_dir=run_dir,
            model=model,
            pareto=pareto,
            split=split,
            baseline_stats=baseline_stats,
            last_round_details=last_round_details,
            cumulative_principles=cumulative_principles,
            args=args,
            search_batch_indices=round_batch_indices,
            diagnosis_model=diagnosis_model,
            prev_cumulative_tracking=prev_cumulative_tracking,
            exploration_hints=exploration_hints,
            obs_graph_json=obs_graph_json,
        )

        # Observation Graph update (rule-based, no LLM). Fold this round's
        # candidate architectures + per-level metrics into the graph so the
        # NEXT round's Proposer sees an enriched structural map. Non-fatal.
        if obs_graph is not None:
            try:
                graph_candidates = [
                    {"architecture": c.get("architecture"),
                     "metrics": c.get("metrics")}
                    for c in (round_summary.get("candidate_results") or [])
                    if c.get("metrics") and not c.get("failed") and not c.get("skipped")
                ]
                if graph_candidates:
                    # Census the task batch ONCE — the search batch is fixed
                    # across rounds, so re-censusing would inflate n_tasks /
                    # category counts. We census until categories are actually
                    # populated (stub nodes from candidate edges have none).
                    need_census = not any(
                        n.kind == "task_pattern" and n.attrs.get("categories")
                        for n in obs_graph.nodes.values()
                    )
                    task_census, baseline_per_level = ([], {})
                    if need_census:
                        task_census, baseline_per_level = _collect_round_task_census(
                            run_dir / f"round_{round_id}", baseline_stats,
                        )
                    obs_graph.update_from_round(
                        round_id, graph_candidates,
                        task_results=(task_census or None),
                        baseline_per_level=(baseline_per_level or None),
                    )
                    logger.info(
                        "ObservationGraph updated after round %d "
                        "(nodes=%d, edges=%d, censused=%s).",
                        round_id, len(obs_graph.nodes), len(obs_graph.edges),
                        bool(task_census),
                    )
            except Exception as _e:
                logger.warning("ObservationGraph update failed (non-fatal): %s", _e)

        # Smart-5 fix (2026-05-16): update tracker with this round's
        # candidates so the NEXT round's hints reflect what was just tried.
        try:
            for c in (round_summary.get("candidate_results") or []):
                a = c.get("architecture")
                if a:
                    explorer.update(a)
        except Exception as _e:
            logger.warning("ExplorationTracker.update failed: %s", _e)

        # Ledger update: gather this round's per-candidate attribution.json files
        # and let the diagnosis model produce a structured delta.
        if not args.no_ledger and diagnosis_model is not None and not args.dry_run:
            attributions: List[Dict[str, Any]] = []
            for cand_dir in (run_dir / f"round_{round_id}").glob("candidate_*"):
                attr_file = cand_dir / "attribution.json"
                if attr_file.exists():
                    try:
                        a = json.loads(attr_file.read_text(encoding="utf-8"))
                        a["config_id"] = f"r{round_id}_c{cand_dir.name.split('_')[-1]}"
                        attributions.append(a)
                    except Exception as _e:
                        logger.warning("[ledger] could not load %s: %s",
                                       attr_file, _e)
            try:
                ledger.update_with_round(
                    round_id=round_id,
                    round_summary=round_summary,
                    attributions=attributions,
                    diagnosis_model=diagnosis_model,
                    prompt_template_path=ledger_prompt_path,
                )
            except Exception as _e:
                logger.warning("[ledger] update_with_round raised: %s — "
                               "ledger left unchanged this round.", _e)

        last_round_summary = round_summary

        # Save search state. We no longer persist a freeform
        # `cumulative_principles` string — the ledger is the source of
        # truth and lives in run_dir/ledger/ledger.json.
        # cumulative_tracking persists so resumed runs can extend the
        # rounds_to_discovery map without losing earlier crossings.
        state = {
            "next_round": round_id + 1,
            "last_round_summary": last_round_summary,
            "ledger_disabled": args.no_ledger,
            "cumulative_tracking": round_summary.get("cumulative_tracking", {}),
        }
        _atomic_write_json(state_path, state)

        # Log Pareto front after each round
        best = pareto.best()
        if best:
            logger.info("Best on Pareto front: %s", best.summary_str())

    # Fixed protocol M3 always runs, independently of the optional held-out
    # validation report.
    _runoff_winner = run_protocol_runoff(
        run_dir, pareto, search_batch_indices, baseline_stats, args
    )
    if args.final_validation:
        run_final_validation(run_dir, pareto, split, baseline_stats, args,
                             best_override=_runoff_winner)

    # Final summary
    best = pareto.best()
    logger.info("=== AutoMem Search Complete ===")
    logger.info("Run dir: %s", run_dir)
    logger.info("Total evaluated: %d", len(pareto.all_evaluated()))
    logger.info("Pareto front size: %d", pareto.size())
    if _runoff_winner is not None:
        logger.info(
            "Best architecture (M3 runoff): [%s] acc=%.3f fit=%.4f | %s",
            _runoff_winner.get("config_id", "runoff_winner"),
            float((_runoff_winner.get("runoff_metrics") or {}).get("accuracy", 0.0)),
            float(_runoff_winner.get("runoff_fitness", 0.0)),
            _runoff_winner.get("architecture", {}),
        )
    elif best:
        logger.info("Best architecture: %s", best.summary_str())


if __name__ == "__main__":
    main()
