#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The OPPO Inc. PersonalAI team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Unified utilities for evaluation tasks including logging, reporting, and statistics.
"""

import os
import json
import hashlib
import math
import re
import tempfile
import time
import logging
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime

logger = logging.getLogger(__name__)


def dataset_file_sha256(path: str | os.PathLike[str]) -> str:
    """Hash the exact benchmark file used to assign global task indices."""

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def task_identity_digest(dataset_sha256: str, item_index: int) -> str:
    """Bind one result to an exact dataset file and one-based row index."""

    if not re.fullmatch(r"[0-9a-f]{64}", str(dataset_sha256 or "")):
        raise ValueError("dataset_sha256 must be a lowercase SHA-256 digest")
    if type(item_index) is not int or item_index < 1:
        raise ValueError("item_index must be a positive exact integer")
    material = f"automem-task-v1\0{dataset_sha256}\0{item_index}".encode("ascii")
    return hashlib.sha256(material).hexdigest()


def task_result_validation_error(result: Any) -> Optional[str]:
    """Return why a task result is unsafe to score, or ``None`` when valid."""

    if not isinstance(result, dict):
        return "result is not a JSON object"
    status = result.get("status")
    if status != "success":
        return f"task runner did not report explicit success: {status!r}"
    if result.get("judge_unjudged") is not False:
        return "task result must explicitly confirm a usable judge verdict"

    item_index = result.get("item_index")
    if type(item_index) is not int or item_index < 1:
        return "item_index must be a positive exact integer"
    task_identity = result.get("task_identity")
    if not isinstance(task_identity, str) or not re.fullmatch(
        r"[0-9a-f]{64}", task_identity
    ):
        return "task_identity must be a lowercase SHA-256 digest"

    task_score = result.get("task_score")
    if (
        isinstance(task_score, bool)
        or not isinstance(task_score, (int, float))
        or not math.isfinite(float(task_score))
    ):
        return "task_score must be a finite number"
    if not 0.0 <= float(task_score) <= 1.0:
        return "task_score must be between 0 and 1"
    return None


def _task_result_filename_matches(filename: str, payload: Dict[str, Any]) -> bool:
    """Bind a task checkpoint to its one-based internal item index."""

    return filename == f"{payload['item_index']}.json"


def completed_task_result_stems(run_dir: str) -> set[str]:
    """Return filenames whose JSON payload is complete enough to resume-skip."""

    completed: set[str] = set()
    directory = os.path.realpath(os.path.abspath(run_dir))
    if not os.path.isdir(directory):
        return completed
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(directory, filename)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            task_result_validation_error(payload) is None
            and _task_result_filename_matches(filename, payload)
        ):
            completed.add(filename[:-5])
    return completed


def load_completed_task_results(run_dir: str) -> List[Dict[str, Any]]:
    """Load validated per-task checkpoints for deterministic resume reports."""

    results: List[Dict[str, Any]] = []
    directory = os.path.realpath(os.path.abspath(run_dir))
    if not os.path.isdir(directory):
        return results
    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(directory, filename)
        if os.path.islink(path) or not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as stream:
                payload = json.load(stream)
        except (OSError, json.JSONDecodeError):
            continue
        if (
            task_result_validation_error(payload) is None
            and _task_result_filename_matches(filename, payload)
        ):
            results.append(payload)
    return sorted(results, key=lambda result: result["item_index"])


def require_complete_task_run(
    dataset_name: str,
    results: List[Dict[str, Any]],
    expected_count: int,
    future_errors: List[str],
) -> None:
    """Make runner infrastructure and judge failures visible to the caller."""

    invalid_results = [
        reason
        for result in results
        if (reason := task_result_validation_error(result)) is not None
    ]
    valid_indices = [
        result["item_index"]
        for result in results
        if task_result_validation_error(result) is None
    ]
    duplicate_indices = len(valid_indices) - len(set(valid_indices))
    if (
        future_errors
        or invalid_results
        or duplicate_indices
        or len(results) != expected_count
    ):
        raise RuntimeError(
            f"{dataset_name} run incomplete: "
            f"future_errors={len(future_errors)}, "
            f"invalid_results={len(invalid_results)}, "
            f"duplicate_indices={duplicate_indices}, "
            f"completed={len(results)}/{expected_count}"
        )


# AppWorld freezes the world clock by patching time.time/time.monotonic
# while a world is open. Capture the real functions at import time so
# duration measurement is unaffected by the patch.
_REAL_MONOTONIC = time.monotonic
_REAL_TIME = time.time


class TaskTimer:
    """Timer for tracking task execution time"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start the timer"""
        self.start_time = _REAL_MONOTONIC()
    
    def stop(self):
        """Stop the timer and return elapsed time"""
        self.end_time = _REAL_MONOTONIC()
        return self.elapsed()
    
    def elapsed(self):
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time


class TokenCounter:
    """Counter for tracking token usage and API calls"""
    
    def __init__(self):
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.api_calls = 0
    
    def add(self, prompt_tokens: int = 0, completion_tokens: int = 0, api_calls: int = 1):
        """Add token usage"""
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += (prompt_tokens + completion_tokens)
        self.api_calls += api_calls
    
    def to_dict(self):
        """Convert to dictionary"""
        return {
            "total_tokens": self.total_tokens,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "api_calls": self.api_calls
        }
    
    @staticmethod
    def from_trajectory(trajectory: List[Dict]) -> 'TokenCounter':
        """Extract token usage from agent trajectory"""
        counter = TokenCounter()
        for step in trajectory:
            if isinstance(step, dict):
                # Extract from step metadata if available
                usage = step.get("usage", {})
                if usage:
                    counter.add(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        api_calls=1
                    )
        return counter
    
    @staticmethod
    def from_model(model) -> 'TokenCounter':
        """Extract token usage from model's total counts"""
        counter = TokenCounter()
        if hasattr(model, 'get_total_counts'):
            counts = model.get_total_counts()
            counter.total_tokens = counts.get("total_tokens", 0)
            counter.prompt_tokens = counts.get("total_input_tokens", 0)
            counter.completion_tokens = counts.get("total_output_tokens", 0)
            counter.api_calls = counts.get("total_api_calls", 0)
        return counter


def save_task_result(
    result: Dict[str, Any],
    run_dir: str,
    filename: Optional[str] = None
) -> str:
    """
    Save a single task result to JSON file.
    
    Args:
        result: Task result dictionary
        run_dir: Directory to save the file
        filename: Optional custom filename, otherwise uses item_index or task_id
    
    Returns:
        Path to saved file
    """
    os.makedirs(run_dir, exist_ok=True)
    output_dir = os.path.realpath(os.path.abspath(run_dir))
    
    if filename is None:
        # Determine filename from result
        idx = result.get("item_index")
        if idx is None:
            idx = result.get("task_id")
        if idx is not None:
            filename = f"{idx}.json"
        else:
            import uuid
            filename = f"{uuid.uuid4().hex}.json"
    
    filename = str(filename)
    if (
        not filename
        or filename in {".", ".."}
        or os.path.isabs(filename)
        or "/" in filename
        or "\\" in filename
        or not re.fullmatch(r"[A-Za-z0-9._-]+\.json", filename)
    ):
        raise ValueError(f"Unsafe task-result filename: {filename!r}")

    filepath = os.path.join(output_dir, filename)
    fd, temp_path = tempfile.mkstemp(
        dir=output_dir, prefix=f".{filename}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        # Atomic replacement also replaces a pre-existing symlink itself
        # instead of following it to a file outside output_dir.
        os.replace(temp_path, filepath)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise

    return filepath


def generate_unified_report(
    results: List[Dict[str, Any]],
    output_path: str,
    dataset_name: str = "Evaluation",
    has_levels: bool = True,
    level_key: str = "level"
) -> Dict[str, Any]:
    """
    Generate unified evaluation report with statistics.
    
    Args:
        results: List of task results
        output_path: Path to save the report
        dataset_name: Name of the dataset for report title
        has_levels: Whether the dataset has difficulty levels
        level_key: Key name for level/difficulty field
    
    Returns:
        Statistics dictionary
    """
    if not results:
        logger.warning("No results to generate report")
        return {}
    
    total = len(results)
    successful = sum(1 for r in results if r.get("status") == "success")
    errors = sum(1 for r in results if r.get("status") == "error")
    
    correct = sum(1 for r in results if str(r.get("judgement") or "").strip().lower() == "correct")
    incorrect = sum(1 for r in results if str(r.get("judgement") or "").strip().lower() == "incorrect")
    
    # Aggregate token usage and timing
    total_tokens = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_api_calls = 0
    total_time = 0.0
    
    for r in results:
        metrics = r.get("metrics", {})
        total_tokens += metrics.get("total_tokens", 0)
        total_prompt_tokens += metrics.get("prompt_tokens", 0)
        total_completion_tokens += metrics.get("completion_tokens", 0)
        total_api_calls += metrics.get("api_calls", 0)
        total_time += metrics.get("elapsed_time", 0)
    
    # Statistics by level/difficulty
    by_level = defaultdict(lambda: {"total": 0, "correct": 0})
    if has_levels:
        for r in results:
            level = r.get(level_key, "unknown")
            by_level[level]["total"] += 1
            if str(r.get("judgement") or "").strip().lower() == "correct":
                by_level[level]["correct"] += 1
    
    # Generate report
    stats = {
        "dataset": dataset_name,
        "total_tasks": total,
        "successful": successful,
        "errors": errors,
        "correct": correct,
        "incorrect": incorrect,
        "accuracy": correct / total if total > 0 else 0,
        "total_tokens": total_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_api_calls": total_api_calls,
        "total_time": total_time,
        "avg_tokens_per_task": total_tokens / total if total > 0 else 0,
        "avg_prompt_tokens_per_task": total_prompt_tokens / total if total > 0 else 0,
        "avg_completion_tokens_per_task": total_completion_tokens / total if total > 0 else 0,
        "avg_time_per_task": total_time / total if total > 0 else 0,
    }
    
    if has_levels:
        stats["by_level"] = dict(by_level)
    
    # Write report file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(f"{dataset_name} Evaluation Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total Tasks: {total}\n")
        f.write(f"Successful: {successful} ({successful/total*100:.1f}%)\n")
        f.write(f"Errors: {errors} ({errors/total*100:.1f}%)\n\n")
        
        f.write(f"Correct: {correct}\n")
        f.write(f"Incorrect: {incorrect}\n")
        f.write(f"Accuracy: {correct/total*100:.2f}%\n\n")
        
        f.write("-" * 80 + "\n")
        f.write("Resource Usage\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Tokens: {total_tokens:,}\n")
        f.write(f"  - Prompt Tokens: {total_prompt_tokens:,}\n")
        f.write(f"  - Completion Tokens: {total_completion_tokens:,}\n")
        f.write(f"Total API Calls: {total_api_calls}\n")
        f.write(f"Total Time: {total_time:.2f}s ({total_time/60:.2f}m)\n\n")
        f.write("Average Per Task:\n")
        f.write(f"  - Tokens: {stats['avg_tokens_per_task']:.1f}\n")
        f.write(f"  - Prompt Tokens: {stats['avg_prompt_tokens_per_task']:.1f}\n")
        f.write(f"  - Completion Tokens: {stats['avg_completion_tokens_per_task']:.1f}\n")
        f.write(f"  - Time: {stats['avg_time_per_task']:.2f}s\n\n")
        
        if has_levels:
            f.write("-" * 80 + "\n")
            f.write(f"By {level_key.capitalize()}\n")
            f.write("-" * 80 + "\n")
            for level in sorted(by_level.keys()):
                level_stats = by_level[level]
                acc = level_stats["correct"] / level_stats["total"] * 100 if level_stats["total"] > 0 else 0
                f.write(f"  {level}: {level_stats['correct']}/{level_stats['total']} ({acc:.1f}%)\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    logger.info(f"Report saved to {output_path}")
    
    # Print summary to console
    print("\n" + "=" * 80)
    print(f"{dataset_name} Evaluation Summary")
    print("=" * 80)
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
    print(f"Tokens: {total_tokens:,} (Prompt: {total_prompt_tokens:,} | Completion: {total_completion_tokens:,})")
    print(f"API Calls: {total_api_calls} | Time: {total_time/60:.1f}m")
    print("=" * 80 + "\n")
    
    return stats


def enrich_result_with_metrics(
    result: Dict[str, Any],
    timer: TaskTimer,
    token_counter: Optional[TokenCounter] = None,
    trajectory: Optional[List[Dict]] = None
) -> Dict[str, Any]:
    """
    Enrich result dictionary with timing and token metrics.
    
    Args:
        result: Original result dictionary
        timer: TaskTimer instance
        token_counter: Optional TokenCounter instance
        trajectory: Optional trajectory for extracting token usage
    
    Returns:
        Enriched result dictionary
    """
    elapsed_time = timer.elapsed()
    
    # Extract or use provided token counter
    if token_counter is None and trajectory is not None:
        token_counter = TokenCounter.from_trajectory(trajectory)
    
    metrics = {
        "elapsed_time": elapsed_time,
    }
    
    if token_counter:
        metrics.update(token_counter.to_dict())
    
    result["metrics"] = metrics
    return result


def capture_memory_metrics(memory_provider) -> Dict[str, Any]:
    """Safely retrieve provider-level experiment metrics for a single task."""
    if memory_provider is None:
        return {}

    getter = getattr(memory_provider, "get_experiment_metrics", None)
    if not callable(getter):
        return {}

    try:
        metrics = getter() or {}
        return metrics if isinstance(metrics, dict) else {}
    except Exception as e:
        logger.warning(f"Failed to capture memory metrics: {e}")
        return {}


def create_run_directory(
    base_dir: str,
    dataset_name: str,
    memory_name: str = "",
    use_timestamp: bool = True
) -> str:
    """
    Create run directory with optional timestamp.
    
    Args:
        base_dir: Base output directory
        dataset_name: Name of dataset/evaluation
        memory_name: Optional memory provider name prefix
        use_timestamp: If True, creates nested timestamped dir; if False, uses base_dir directly
    
    Returns:
        Path to created run directory
    """
    if not use_timestamp:
        os.makedirs(base_dir, exist_ok=True)
        return base_dir
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"{memory_name}{timestamp}" if memory_name else timestamp
    run_dir = os.path.join(base_dir, f"{dataset_name}_runs", run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir
