#!/usr/bin/env python
"""AppWorld benchmark runner for AutoMem.

Evaluates a memory-equipped AppWorld agent on tasks exported by
`automem.benchmarks.appworld.prepare_data`. Scoring is AppWorld's own
deterministic state/answer assertions (`world.evaluate()`); no judge LLM is
used. The CLI contract mirrors the WebWalkerQA runner so the architecture
search engine can launch it unchanged.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import threading

from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from automem.config import get_memory_config, load_runtime_config
from automem.endpoints import resolve_openai_endpoint
from automem.evaluation.io import read_jsonl, write_jsonl
from automem.evaluation.utils import (
    TaskTimer,
    TokenCounter,
    create_run_directory,
    dataset_file_sha256,
    enrich_result_with_metrics,
    capture_memory_metrics,
    generate_unified_report,
    load_completed_task_results,
    require_complete_task_run,
    save_task_result,
    task_identity_digest,
)
from automem.memory_types import MemoryType, TrajectoryData, get_provider_class
from flashoagents.models import OpenAIServerModel

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

load_dotenv(override=False)


def _validate_appworld_items(data):
    """Enforce the dataset contract before any model is initialized."""

    required_fields = ("task_id", "instruction")
    for row_number, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"AppWorld row {row_number} must be a JSON object")
        invalid = [
            field
            for field in required_fields
            if not isinstance(item.get(field), str) or not item[field].strip()
        ]
        if invalid:
            raise ValueError(
                f"AppWorld row {row_number} has missing or empty required "
                f"field(s): {', '.join(invalid)}"
            )


def parse_task_indices(indices_str):
    """Parse index string like "5", "1-10" or "1,3,5-8,10" into a 1-based index set."""

    if not indices_str:
        return None

    indices = set()
    for part in indices_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-")
            start, end = int(start.strip()), int(end.strip())
            if start > end:
                raise ValueError(f"Invalid range: {part} (start > end)")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return indices


def load_memory_provider(memory_type_str, model=None, runtime_config_path=None):
    """Load and initialize memory provider from type string"""

    if not memory_type_str:
        return None
    try:
        memory_type = MemoryType(memory_type_str)
    except ValueError:
        logger.error(f"Invalid memory type: {memory_type_str}")
        return None
    try:
        provider_class = get_provider_class(memory_type)
        config = get_memory_config(memory_type, runtime_config_path)
        if model is not None:
            config["model"] = model
        provider = provider_class(config=config)
        if not provider.initialize():
            logger.error(f"Failed to initialize memory provider: {memory_type_str}")
            return None
        logger.info(f"Memory provider loaded: {memory_type_str}")
        return provider
    except Exception as e:
        logger.error(f"Failed to load memory provider {memory_type_str}: {e}")
        import traceback

        traceback.print_exc()
        return None


def _evaluation_summary(tracker) -> dict:
    """Reduce a TestTracker to a compact, JSON-safe summary."""

    try:
        stats = tracker.to_dict(stats_only=True)
    except Exception:
        stats = {}
    summary = {
        "success": bool(tracker.success),
    }
    for key in ("num_tests", "num_passes", "num_failures", "pass_percentage"):
        if key in stats:
            summary[key] = stats[key]
    return summary


def _experiment_name(prefix: str, item_index, task_id: str) -> str:
    safe_task_id = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task_id))
    return f"{prefix}_idx{item_index}_{safe_task_id}"


def process_item(
    item,
    model_config,
    summary_interval,
    prompts_type,
    max_steps,
    memory_type_str=None,
    item_index=None,
    enable_memory_evolution=True,
    judge_model=None,  # unused: AppWorld scoring is deterministic
    shared_memory_provider=None,
    extract_plan=None,
    runtime_config_path=None,
    experiment_prefix="automem",
):
    """Process a single AppWorld task with timing and metrics tracking."""

    _validate_appworld_items([item])

    from appworld import AppWorld

    task_id = item["task_id"]
    instruction = item["instruction"]
    split = item.get("split", "")
    difficulty = item.get("difficulty")

    task_model = OpenAIServerModel(**model_config)
    task_model.reset_total_counts()

    memory_provider = shared_memory_provider
    if memory_provider is not None:
        try:
            memory_provider.reset_experiment_metrics()
        except Exception:
            pass
        try:
            memory_provider.model = task_model
            if getattr(memory_provider, "manager", None) is not None:
                memory_provider.manager.llm_client = task_model
        except Exception:
            pass
    elif memory_type_str:
        memory_provider = load_memory_provider(
            memory_type_str, task_model, runtime_config_path
        )

    timer = TaskTimer()
    timer.start()

    world = None
    try:
        world = AppWorld(
            task_id,
            experiment_name=_experiment_name(experiment_prefix, item_index, task_id),
        )

        from automem.benchmarks.appworld.agent import AppWorldAgent

        agent = AppWorldAgent(
            task_model,
            world,
            summary_interval=summary_interval,
            prompts_type=prompts_type,
            max_steps=max_steps,
            memory_provider=memory_provider,
        )

        result = agent(instruction)
        if not isinstance(result, dict) or result.get("error"):
            raise RuntimeError(
                f"Task agent failed before producing an answer: {result!r}"
            )

        try:
            agent_messages = agent.agent_fn.write_memory_to_messages(
                include_system_prompt=False
            )
        except Exception:
            agent_messages = []

        trajectory = result.get("agent_trajectory", [])

        try:
            tracker = world.evaluate()
            evaluation = _evaluation_summary(tracker)
        except Exception as e:
            logger.warning(f"Evaluation failed for {task_id}: {e}")
            evaluation = {"success": False, "error": str(e)}

        is_correct = bool(evaluation.get("success"))

        if memory_provider and enable_memory_evolution:
            try:
                trajectory_data = TrajectoryData(
                    query=instruction,
                    trajectory=trajectory,
                    result=result.get("agent_result"),
                    metadata={
                        "item_index": item_index,
                        "status": "success",
                        "is_correct": is_correct,
                        "split": split,
                        "difficulty": difficulty,
                        "task_id": task_id,
                        "full_query": instruction,
                    },
                )
                success, msg = memory_provider.take_in_memory(
                    trajectory_data, extract_plan=extract_plan
                )
                if success:
                    logger.debug(f"Memory ingested: {msg}")
                else:
                    logger.warning(f"Memory ingestion failed: {msg}")
            except Exception as e:
                logger.warning(f"take_in_memory failed: {e}")

        token_counter = TokenCounter.from_model(task_model)

        task_result = {
            "agent_result": result.get("agent_result"),
            "judgement": "correct" if is_correct else "incorrect",
            "judge_unjudged": False,
            "judge_fallback": None,
            "task_score": 1.0 if is_correct else 0.0,
            "success": is_correct,
            "item_index": item_index,
            "task_identity": item.get("_task_identity"),
            "task_id": task_id,
            "question": instruction,
            "enhanced_question": instruction,
            "golden_answer": None,
            "split": split,
            "difficulty": difficulty,
            "evaluation": evaluation,
            "status": "success",
            "agent_trajectory": trajectory,
            "agent_messages": agent_messages,
            "memory_metrics": capture_memory_metrics(memory_provider),
        }

        timer.stop()
        return enrich_result_with_metrics(task_result, timer, token_counter)

    except Exception as e:
        import traceback

        error_msg = traceback.format_exc()
        logger.error(
            f"Exception occurred while processing task {task_id}: {error_msg}"
        )
        task_result = {
            "agent_result": None,
            "judgement": None,
            "judge_unjudged": False,
            "judge_fallback": None,
            "task_score": 0.0,
            "success": False,
            "status": "error",
            "error": str(e),
            "error_traceback": error_msg,
            "item_index": item_index,
            "task_identity": item.get("_task_identity"),
            "task_id": task_id,
            "question": instruction,
            "enhanced_question": instruction,
            "golden_answer": None,
            "split": split,
            "difficulty": difficulty,
            "agent_trajectory": [],
            "agent_messages": [],
            "memory_metrics": capture_memory_metrics(memory_provider),
        }
        timer.stop()
        token_counter = TokenCounter.from_model(task_model)
        return enrich_result_with_metrics(task_result, timer, token_counter)
    finally:
        if world is not None:
            try:
                world.save_logs()
            except Exception as e:
                logger.debug(f"save_logs failed for {task_id}: {e}")
            try:
                world.close()
            except Exception as e:
                logger.debug(f"world.close failed for {task_id}: {e}")


def main(args):
    infile = Path(args.infile).expanduser()
    if not infile.is_file():
        raise FileNotFoundError(f"AppWorld input file not found: {infile}")
    args.infile = str(infile)
    outfile = Path(args.outfile).expanduser()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    args.outfile = str(outfile)

    runtime_config_path = getattr(args, "runtime_config_json", None)
    runtime_config = (
        load_runtime_config(runtime_config_path) if runtime_config_path else None
    )
    extract_plan = runtime_config["extract_plan"] if runtime_config else None

    random.seed(args.seed)
    dataset_sha256 = dataset_file_sha256(args.infile)

    raw = read_jsonl(args.infile)
    data = []
    for idx, it in enumerate(raw):
        if isinstance(it, dict):
            it = dict(it)
            it["_global_index"] = idx + 1
            it["_task_identity"] = task_identity_digest(dataset_sha256, idx + 1)
        data.append(it)

    _validate_appworld_items(data)
    logger.info(f"Loaded {len(data)} items from {args.infile}")

    custom_role_conversions = {"tool-call": "assistant", "tool-response": "user"}
    task_api_key, task_api_base = resolve_openai_endpoint()
    model_config = {
        "model_id": args.model or os.environ.get("DEFAULT_MODEL", "gpt-5"),
        "custom_role_conversions": custom_role_conversions,
        "max_completion_tokens": args.token_budget,
        "api_key": task_api_key,
        "api_base": task_api_base,
    }
    model = OpenAIServerModel(**model_config)

    if args.difficulty:
        try:
            difficulty_filter = int(args.difficulty)
        except ValueError:
            raise ValueError(f"--difficulty must be an integer (1/2/3), got {args.difficulty!r}")
        before = len(data)
        data = [it for it in data if it.get("difficulty") == difficulty_filter]
        logger.info(
            f"Difficulty filter applied: difficulty={difficulty_filter}, kept {len(data)}/{before}"
        )

    if args.task_indices:
        try:
            selected_indices = parse_task_indices(args.task_indices)
            data = [
                data[i - 1] for i in sorted(selected_indices) if 0 < i <= len(data)
            ]
            logger.info(f"Selected {len(data)} tasks from indices: {args.task_indices}")
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid --task_indices: {args.task_indices!r}") from e
        if not data:
            raise ValueError(
                f"--task_indices selected no AppWorld tasks: {args.task_indices!r}"
            )
    elif args.sample_num is not None:
        data = data[: args.sample_num]
        logger.info(f"Limited to first {args.sample_num} tasks")

    data_to_run = data
    logger.info(f"Total data to process: {len(data_to_run)}")

    memory_name = ""
    if args.memory_provider:
        try:
            memory_name = MemoryType(args.memory_provider).value + "_"
        except ValueError:
            pass

    if args.direct_output_dir:
        run_dir = args.direct_output_dir
        os.makedirs(run_dir, exist_ok=True)
        logger.info(f"Using direct output directory: {run_dir}")
    else:
        out_dir = os.path.dirname(args.outfile) or "."
        base_name = os.path.splitext(os.path.basename(args.outfile))[0]
        run_dir = create_run_directory(out_dir, base_name, memory_name)
        logger.info(f"Run directory created: {run_dir}")

    results = []
    file_lock = threading.Lock()
    effective_concurrency = args.concurrency
    shared_memory_provider = None
    if args.shared_memory_provider and args.memory_provider:
        if effective_concurrency != 1:
            logger.warning(
                "--shared_memory_provider with --concurrency %d (>1): shared pool is "
                "accessed concurrently; results may be mildly nondeterministic.",
                effective_concurrency,
            )
        shared_memory_provider = load_memory_provider(
            args.memory_provider, model, runtime_config_path
        )
        if shared_memory_provider is None:
            raise RuntimeError("Failed to initialize shared memory provider")

    def safe_write(result):
        with file_lock:
            idx = result.get("item_index")
            filename = f"{idx}.json" if idx is not None else None
            save_task_result(result, run_dir, filename)

    if args.memory_provider:
        if shared_memory_provider is not None:
            logger.info(
                f"Memory provider enabled: {args.memory_provider} (shared provider, concurrency={effective_concurrency})"
            )
        else:
            logger.info(
                f"Memory provider enabled: {args.memory_provider} (each thread creates independent instance)"
            )

    # Skip-on-exist: resumed runs continue from where they stopped.
    _n_before = len(data_to_run)
    _completed_identities = {
        row["item_index"]: row["task_identity"]
        for row in load_completed_task_results(run_dir)
    }
    data_to_run = [
        it
        for it in data_to_run
        if not (
            isinstance(it, dict)
            and it.get("_global_index") is not None
            and _completed_identities.get(it["_global_index"])
            == it.get("_task_identity")
        )
    ]
    if len(data_to_run) < _n_before:
        logger.info(
            f"Skip-on-exist: {_n_before - len(data_to_run)} already-done task(s) in "
            f"{run_dir}; running {len(data_to_run)} remaining."
        )

    future_errors = []
    summary_interval = random.randint(
        args.summary_interval - 1, args.summary_interval + 1
    )

    # AppWorld's execute()/evaluate() drive an IPython REPL guarded by signal
    # timeouts, which only work in the main interpreter thread. Running them in
    # a ThreadPoolExecutor worker both fails every execution AND leaks the
    # safety guard's patched builtins (its disable() sits on the success path).
    # So tasks run sequentially in the main thread; parallel evaluation would
    # require `appworld serve` remote-environment mode.
    if effective_concurrency > 1:
        logger.warning(
            "--concurrency %d requested but AppWorld requires main-thread "
            "execution; forcing sequential (use `appworld serve` for parallel).",
            effective_concurrency,
        )
        effective_concurrency = 1

    from tqdm import tqdm as _tqdm

    for item in _tqdm(data_to_run, desc="Processing AppWorld"):
        try:
            result = process_item(
                item,
                model_config,
                summary_interval,
                args.prompts_type,
                args.max_steps,
                args.memory_provider,
                (item.get("_global_index") if isinstance(item, dict) else None),
                args.enable_memory_evolution,
                args.judge_model,
                shared_memory_provider,
                extract_plan,
                runtime_config_path,
                args.experiment_name,
            )
            if result:
                results.append(result)
                safe_write(result)

                metrics = result.get("metrics", {})
                if result.get("status") == "success":
                    logger.info(
                        f"Task done [{len(results)}/{len(data_to_run)}]: {result['task_id']} "
                        f"| pass={result.get('success')} | Time: {metrics.get('elapsed_time', 0):.1f}s "
                        f"| Tokens: {metrics.get('total_tokens', 0)}"
                    )
                elif result.get("status") == "error":
                    logger.warning(
                        f"Task error [{len(results)}/{len(data_to_run)}]: {result['task_id']} "
                        f"| Error: {str(result.get('error', 'Unknown'))[:200]}"
                    )
        except Exception as exc:
            import traceback

            logger.error(f"Task processing raised: {traceback.format_exc()}")
            future_errors.append(str(exc))

    logger.info(f"Processing completed. Total results: {len(results)}")
    require_complete_task_run(
        "AppWorld", results, len(data_to_run), future_errors
    )
    all_results = [
        row
        for row in load_completed_task_results(run_dir)
        if row["task_identity"]
        == task_identity_digest(dataset_sha256, row["item_index"])
    ]

    write_jsonl(args.outfile, all_results)
    logger.info(f"Results saved to {args.outfile}")

    report_path = os.path.join(run_dir, "report.txt")
    generate_unified_report(
        all_results,
        report_path,
        dataset_name="AppWorld",
        has_levels=True,
        level_key="difficulty",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AppWorld evaluation with AutoMem memory")

    parser.add_argument("--infile", type=str, required=True,
                        help="AppWorld JSONL input path (from prepare_data)")
    parser.add_argument("--outfile", type=str,
                        default="runs/benchmarks/appworld/results.jsonl",
                        help="Output path for results")
    parser.add_argument("--model", type=str, default=None, help="Task model id override")
    parser.add_argument("--sample_num", type=int, default=None,
                        help="Number of samples to process")
    parser.add_argument("--task_indices", type=str, default=None,
                        help='Task indices to run, e.g. "5" or "1-10" or "1,3,5-10"')
    parser.add_argument("--summary_interval", type=int, default=8,
                        help="Summary interval for agent")
    parser.add_argument("--prompts_type", type=str, default="appworld",
                        help="Prompt templates to use (default: appworld)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max_steps", type=int, default=40,
                        help="Maximum number of steps for agent")
    parser.add_argument("--token_budget", type=int, default=32768,
                        help="Model max completion tokens")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="Unused: AppWorld scoring is deterministic")
    parser.add_argument("--memory_provider", type=str,
                        choices=[MemoryType.MODULAR.value], default=None,
                        help="Enable the AutoMem modular provider")
    parser.add_argument("--enable_memory_evolution", action="store_true", default=True,
                        help="Enable memory system evolution (take_in_memory). Default: True")
    parser.add_argument("--disable_memory_evolution", dest="enable_memory_evolution",
                        action="store_false",
                        help="Disable memory system evolution (skip take_in_memory)")
    parser.add_argument("--difficulty", type=str, default=None,
                        help="Filter tasks by AppWorld difficulty (1/2/3)")
    parser.add_argument("--direct_output_dir", type=str, default=None,
                        help="Direct output directory (skips timestamped nesting)")
    parser.add_argument("--runtime_config_json", type=str, default=None,
                        help="Structured RuntimeConfig JSON emitted by AutoMem search")
    parser.add_argument("--shared_memory_provider", action="store_true",
                        help="Reuse one memory provider across the full run")
    parser.add_argument("--experiment_name", type=str, default="automem",
                        help="Prefix for AppWorld experiment output directories")

    args = parser.parse_args()

    main(args)
