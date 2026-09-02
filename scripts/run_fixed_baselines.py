#!/usr/bin/env python
"""Fixed memory-system baselines on AppWorld, mapped into AutoMem's architecture space.

Following the paper's Table 1 ("existing systems are fixed points in a shared
architectural space"), each classic memory system is approximated as one fixed
architecture configuration evaluated under the same automem-runtime-v1 as the
searched candidates — so the comparison isolates the architecture, not the
execution policy.

Mapping (system -> approximation):
  mem0        tip / vector / hybrid / lightweight          (update+delete ~ lightweight)
  memorybank  trajectory / vector / hybrid / lightweight   (forget ~ time_decay)
  expel       tip+insight / hybrid / contrastive / json_full (update+merge ~ json_full)
  voyager     shortcut / vector / hybrid / tool_manager    (skill validate ~ tool_manager)
  awm         workflow / json / hybrid / json_full         (merge ~ cluster_merge)

Per baseline, two phases:
  learn: warmup+search tasks (70) with memory evolution ON  -> builds its own pool
  test:  held-out final_test tasks (40) with evolution OFF -> pure retrieval evaluation

Usage:
  python scripts/run_fixed_baselines.py --run_root runs/baselines \
      --split_config runs/search/appworld-full/data_split.json [--only mem0,expel]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BASELINES: dict[str, dict] = {
    "mem0": {
        "extract_types": ["tip"],
        "storage_routing": {"tip": "vector"},
        "retrieval": "hybrid",
        "management": "lightweight",
    },
    "memorybank": {
        "extract_types": ["trajectory"],
        "storage_routing": {"trajectory": "vector"},
        "retrieval": "hybrid",
        "management": "lightweight",
    },
    "expel": {
        "extract_types": ["tip", "insight"],
        "storage_routing": {"tip": "hybrid", "insight": "hybrid"},
        "retrieval": "contrastive",
        "management": "json_full",
    },
    "voyager": {
        "extract_types": ["shortcut"],
        "storage_routing": {"shortcut": "vector"},
        "retrieval": "hybrid",
        "management": "tool_manager",
    },
    "awm": {
        "extract_types": ["workflow"],
        "storage_routing": {"workflow": "json"},
        "retrieval": "hybrid",
        "management": "json_full",
    },
}


def compile_runtime_config(arch: dict, storage_dir: str, out_path: Path) -> None:
    from automem.architecture.compiler import ArchitectureCompiler
    from automem.architecture.models import ArchitectureSpec

    compiler = ArchitectureCompiler(base_storage_dir=storage_dir)
    spec = ArchitectureSpec.from_search_dict(arch)
    runtime_config = compiler.compile_spec(spec)
    out_path.write_text(
        json.dumps(runtime_config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def indices_str(indices: list[int]) -> str:
    # data_split indices are 0-based; runner expects 1-based
    return ",".join(str(i + 1) for i in sorted(indices))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=str, default="runs/baselines")
    parser.add_argument(
        "--split_config", type=str,
        default="runs/search/appworld-full/data_split.json",
    )
    parser.add_argument("--infile", type=str, default="data/appworld/appworld_train_dev.jsonl")
    parser.add_argument("--model", type=str, default="hy3")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--only", type=str, default=None, help="Comma-separated subset of baselines")
    parser.add_argument("--sequential", action="store_true", help="Run baselines one at a time")
    args = parser.parse_args()

    split = json.loads(Path(args.split_config).read_text(encoding="utf-8"))
    learn_indices = split["profile_indices"] + split["optimization_indices"]
    test_indices = split["final_test_indices"]

    names = list(BASELINES)
    if args.only:
        names = [n.strip() for n in args.only.split(",") if n.strip() in BASELINES]

    run_root = Path(args.run_root)
    procs = []
    for name in names:
        arch = BASELINES[name]
        base_dir = run_root / name
        base_dir.mkdir(parents=True, exist_ok=True)
        # One shared storage dir per baseline: learn phase writes, test phase reads.
        cfg_path = base_dir / "runtime_config.json"
        compile_runtime_config(arch, str(base_dir / "storage"), cfg_path)

        def runner_cmd(tasks_dir: Path, indices: list[int], evolve: bool) -> list[str]:
            cmd = [
                sys.executable, "-m", "automem.benchmarks.appworld.runner",
                "--infile", args.infile,
                "--outfile", str(tasks_dir / "results.jsonl"),
                "--task_indices", indices_str(indices),
                "--max_steps", str(args.max_steps),
                "--model", args.model,
                "--memory_provider", "modular",
                "--shared_memory_provider",
                "--runtime_config_json", str(cfg_path),
                "--direct_output_dir", str(tasks_dir),
                "--experiment_name", f"baseline_{name}",
            ]
            if not evolve:
                cmd.append("--disable_memory_evolution")
            return cmd

        # Learn phase THEN test phase, sequentially within one baseline.
        script = " && ".join(
            " ".join(cmd)
            for cmd in (
                runner_cmd(base_dir / "learn", learn_indices, evolve=True),
                runner_cmd(base_dir / "test", test_indices, evolve=False),
            )
        )
        log_path = base_dir / "driver.log"
        print(f"[{name}] launching; log -> {log_path}")
        procs.append(
            (name, subprocess.Popen(["bash", "-c", script], stdout=open(log_path, "w"), stderr=subprocess.STDOUT))
        )
        if args.sequential:
            procs[-1][1].wait()

    if not args.sequential:
        failed = []
        for name, proc in procs:
            code = proc.wait()
            print(f"[{name}] exited with {code}")
            if code != 0:
                failed.append(name)
        if failed:
            raise SystemExit(f"baselines failed: {failed}")


if __name__ == "__main__":
    main()
