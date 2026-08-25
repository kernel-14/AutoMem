#!/usr/bin/env python
"""Parallel fixed-baseline runner: shards the learn/test phases for speed.

Same protocol as run_fixed_baselines.py (learn 70 with evolution ON ->
test 40 held-out with evolution OFF), but each phase is split into shards
running as separate processes (AppWorld requires per-process main-thread
REPL, so parallelism comes from processes, not threads).

Shard A of the learn phase inherits the already-accumulated storage; shard B
starts a fresh storage. After learning, all shard pools are merged (dedup by
unit id) into a canonical pool and re-imported into each test shard's storage,
so every test shard evaluates against the same merged memory pool.

Usage:
  python scripts/parallel_baseline.py --run_root runs/baselines \
      --split_config runs/search/appworld-full/data_split.json [--shards 2]
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

STORAGE_UNIT_FILES = {
    "json": ["store_json/memory_db.json"],
    "vector": ["store_vector/metadata.json"],
    "hybrid": ["store_hybrid/memory_db.json"],
}


def compile_runtime_config(arch: dict, storage_dir: str, out_path: Path):
    from automem.architecture.compiler import ArchitectureCompiler
    from automem.architecture.models import ArchitectureSpec

    compiler = ArchitectureCompiler(base_storage_dir=storage_dir)
    spec = ArchitectureSpec.from_search_dict(arch)
    runtime_config = compiler.compile_spec(spec)
    out_path.write_text(
        json.dumps(runtime_config.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return runtime_config


def indices_str(indices: list[int]) -> str:
    return ",".join(str(i + 1) for i in sorted(indices))  # 1-based for runner


def read_pool_units(storage_root: Path) -> list[dict]:
    """Collect every persisted unit dict under one storage root."""

    units: list[dict] = []
    for rel in STORAGE_UNIT_FILES.values():
        for name in rel:
            p = storage_root / name
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if isinstance(data, list):
                    units.extend(u for u in data if isinstance(u, dict) and u.get("id"))
    return units


_EMBED_MODEL = None


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer

        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        _EMBED_MODEL = SentenceTransformer(model_name)
    return _EMBED_MODEL


def merge_pools(storage_roots: list[Path], out_canonical: Path) -> int:
    by_id: dict[str, dict] = {}
    for root in storage_roots:
        for unit in read_pool_units(root):
            by_id.setdefault(str(unit["id"]), unit)
    units = list(by_id.values())
    # Vector stores never persist embeddings (they live only in the faiss
    # index), so re-embed here — otherwise VectorStorage.add silently skips
    # every unit on canonical import.
    model = None
    n_embedded = 0
    for unit in units:
        if not unit.get("embedding"):
            if model is None:
                model = _get_embed_model()
            unit["embedding"] = model.encode(
                unit.get("content", "") if isinstance(unit.get("content"), str)
                else json.dumps(unit.get("content", {}), ensure_ascii=False),
                convert_to_numpy=True,
            ).tolist()
            n_embedded += 1
    state = {
        "schema_version": 2,
        "units": units,
        "applied_merges": [],
        "periodic_rounds": [],
        "graph_edges": [],
    }
    out_canonical.parent.mkdir(parents=True, exist_ok=True)
    out_canonical.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    if n_embedded:
        print(f"    (re-embedded {n_embedded}/{len(units)} units)")
    return len(by_id)


def import_canonical(run_dir: Path, runtime_config) -> None:
    from automem.search.engine import import_canonical_to_storage

    import_canonical_to_storage(Path(run_dir), runtime_config)


def runner_cmd(cfg_path: Path, tasks_dir: Path, indices: list[int], args, evolve: bool) -> list[str]:
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
    ]
    if evolve:
        cmd.append("--enable_memory_evolution")
    else:
        cmd.append("--disable_memory_evolution")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_root", type=str, default="runs/baselines")
    parser.add_argument("--split_config", type=str,
                        default="runs/search/appworld-full/data_split.json")
    parser.add_argument("--infile", type=str, default="data/appworld/appworld_train_dev.jsonl")
    parser.add_argument("--model", type=str, default="hy3")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--shards", type=int, default=2)
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    split = json.loads(Path(args.split_config).read_text(encoding="utf-8"))
    learn_all = sorted(set(split["profile_indices"] + split["optimization_indices"]))
    test_all = sorted(set(split["final_test_indices"]))

    names = [n.strip() for n in (args.only or "").split(",") if n.strip()] or list(BASELINES)
    run_root = Path(args.run_root)

    # ---------- Phase 1: sharded learn ----------
    learn_procs = []
    for name in names:
        arch = BASELINES[name]
        base = run_root / name
        learn_dir = base / "learn"
        done = set()
        for pattern in ("*.json", "shard_*/*.json"):
            for f in learn_dir.glob(pattern):
                try:
                    done.add(int(json.loads(f.read_text(encoding="utf-8"))["item_index"]))
                except Exception:
                    continue
        remaining = [i for i in learn_all if (i + 1) not in done]
        if not remaining:
            print(f"[{name}] learn already complete")
            continue
        shards = [remaining[k::args.shards] for k in range(args.shards)]
        for k, shard in enumerate(shards):
            if not shard:
                continue
            # Shard 0 inherits the existing storage; others start fresh.
            if k == 0 and (base / "runtime_config.json").is_file():
                cfg_path = base / "runtime_config.json"
            else:
                compile_runtime_config(
                    arch, str(base / f"storage_learn_{k}"), base / f"runtime_config_learn_{k}.json"
                )
                cfg_path = base / f"runtime_config_learn_{k}.json"
            tasks_dir = learn_dir / f"shard_{k}"
            tasks_dir.mkdir(parents=True, exist_ok=True)
            log = open(base / f"learn_shard_{k}.log", "w")
            print(f"[{name}] learn shard {k}: {len(shard)} tasks")
            learn_procs.append(
                (name, k, subprocess.Popen(
                    runner_cmd(cfg_path, tasks_dir, shard, args, evolve=True),
                    stdout=log, stderr=subprocess.STDOUT,
                ))
            )
    for name, k, proc in learn_procs:
        code = proc.wait()
        print(f"[{name}] learn shard {k} exited {code}")
        if code != 0:
            raise SystemExit(f"[{name}] learn shard {k} failed")

    # ---------- Phase 2: merge pools -> canonical ----------
    for name in names:
        base = run_root / name
        roots = [base / "storage"] + [base / f"storage_learn_{k}" for k in range(args.shards)]
        n_units = merge_pools([r for r in roots if r.is_dir()], base / "canonical" / "pool.json")
        print(f"[{name}] merged pool: {n_units} units")

    # ---------- Phase 3: sharded test (evolution off, shared merged pool) ----------
    test_procs = []
    for name in names:
        arch = BASELINES[name]
        base = run_root / name
        canonical_dir = base / "canonical"
        test_shards = [test_all[k::args.shards] for k in range(args.shards)]
        for k, shard in enumerate(test_shards):
            if not shard:
                continue
            storage_dir = base / f"storage_test_{k}"
            cfg_path = base / f"runtime_config_test_{k}.json"
            rc = compile_runtime_config(arch, str(storage_dir), cfg_path)
            # import merged canonical units (base/canonical/pool.json) into this shard's fresh storage
            import_canonical(base, rc)
            tasks_dir = base / "test" / f"shard_{k}"
            tasks_dir.mkdir(parents=True, exist_ok=True)
            log = open(base / f"test_shard_{k}.log", "w")
            print(f"[{name}] test shard {k}: {len(shard)} tasks")
            test_procs.append(
                (name, k, subprocess.Popen(
                    runner_cmd(cfg_path, tasks_dir, shard, args, evolve=False),
                    stdout=log, stderr=subprocess.STDOUT,
                ))
            )
    for name, k, proc in test_procs:
        code = proc.wait()
        print(f"[{name}] test shard {k} exited {code}")
        if code != 0:
            raise SystemExit(f"[{name}] test shard {k} failed")

    # ---------- Summary ----------
    print("\n=== Held-out comparison (40 tasks) ===")
    print(f"{'system':<12} {'pass':>5} {'acc':>7}")
    for name in names:
        files = list((run_root / name / "test").glob("shard_*/*.json"))
        passed = 0
        total = 0
        for f in files:
            r = json.loads(f.read_text(encoding="utf-8"))
            passed += int(bool(r.get("success")))
            total += 1
        print(f"{name:<12} {passed:>5} {passed / max(total, 1):>6.1%}  ({total} tasks)")
    print("reference:   no-memory 8  20.0% | AutoMem(r2_c2) 9  22.5%")


if __name__ == "__main__":
    main()
