#!/usr/bin/env python3
"""

Usage
-----
    python main.py --settings config/example_settings.json \\
                   --n_qubits 8 --depth 4 --engine tn \\
                   [--n_runs 20] [--max_bond 32] [--run_id 0]

All CLI arguments override the corresponding values in the settings file,
so the SLURM launcher only needs to pass the parameters that vary across
array tasks (typically ``--n_qubits``, ``--depth``, ``--engine``).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("floquet.main")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    
    p = argparse.ArgumentParser(
        description="mssim: Multi System Simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--settings", required=True,
        help="Path to the JSON settings file.",
    )
    p.add_argument(
        "--n_qubits", type=int, default=None,
        help="Number of qubits (overrides settings).",
    )
    p.add_argument(
        "--depth", type=int, default=None,
        help="Circuit depth (overrides settings).",
    )
    p.add_argument(
        "--engine", type=str, default=None,
        help="Comma-separated engine keys, e.g. 'tn,sv' or 'all' (overrides settings).",
    )
    p.add_argument(
        "--n_runs", type=int, default=None,
        help="Number of random-parameter trials per engine (overrides settings).",
    )
    p.add_argument(
        "--max_bond", type=int, default=None,
        help="Maximum MPS bond dimension (overrides settings).",
    )
    p.add_argument(
        "--run_id", type=int, default=0,
        help="SLURM array task ID, embedded in metadata.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Output file path (overrides settings.output.filename).",
    )
    return p.parse_args(argv)


def load_settings(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)

def merge(settings: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides on top of the settings dict."""
    cfg = dict(settings)  # shallow copy for safety

    # Model overrides
    cfg.setdefault("model", {})
    if args.n_qubits is not None:
        cfg["model"]["n_qubits"] = args.n_qubits
    if args.depth is not None:
        cfg["model"]["depth"] = args.depth

    # Execution overrides
    cfg.setdefault("execution", {})
    if args.engine is not None:
        cfg["execution"]["engines"] = args.engine.split(",")
    if args.n_runs is not None:
        cfg["execution"]["n_runs"] = args.n_runs
    if args.max_bond is not None:
        cfg["execution"]["max_bond_dimension"] = args.max_bond

    # Output overrides
    cfg.setdefault("output", {})
    if args.output is not None:
        cfg["output"]["filename"] = args.output

    return cfg

def main(argv: list[str] | None = None) -> None:
    
    args = parse_args(argv)
    settings = merge(load_settings(args.settings), args)

    model_cfg = settings["model"]
    circuit_name: str = model_cfg["circuit"]
    n_qubits: int = model_cfg["n_qubits"]
    depth: int = model_cfg["depth"]
    observable: list[str] | None = model_cfg.get("observable", None)
    
    circuit_kwargs: dict = model_cfg.get("kwargs", {})
    circuit_kwargs["observable"] = observable

    from src.mssim.circuits.library import build_circuit
    from src.mssim.engines.registry import build_engines
    from src.mssim.executor import Executor


    logger.info("Building circuit '%s' — n_qubits=%d, depth=%d", circuit_name, n_qubits, depth)
    model = build_circuit(circuit_name, n_qubits=n_qubits, depth=depth, **circuit_kwargs)

    # ---- Build engines ---------------------------------------------------
    exec_cfg = settings["execution"]
    engine_keys: list[str] = exec_cfg.get("engines", ["all"])
    max_bond: int | None = exec_cfg.get("max_bond_dimension", None)
    n_runs: int = exec_cfg.get("n_runs", 1)

    logger.info("Building engines: %s (max_bond=%s)", engine_keys, max_bond)
    engines = build_engines(engine_keys, max_bond_dimension=max_bond)

    # ---- Output ----------------------------------------------------------
    out_cfg = settings["output"]
    output_file: str = out_cfg.get("filename", "results.jsonl")
    output_fmt: str = out_cfg.get("format", "jsonl")
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    # ---- Extra metadata: SLURM / host context ----------------------------
    extra_metadata = {
        "slurm_task_id": args.run_id,
        "hostname": socket.gethostname(),
        "settings_file": os.path.abspath(args.settings),
        "launch_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        # SLURM env vars (present only inside a job)
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }

    # ---- Run -------------------------------------------------------------
    executor = Executor(
        n_runs=n_runs,
        output_file=output_file,
        output_fmt=output_fmt,
        extra_metadata=extra_metadata,
        skip_on_error=True,
        verbose=True,
    )

    logger.info(
        "Starting benchmark: %d engine(s) × %d run(s) → %s",
        len(engines), n_runs, output_file,
    )
    batch_results = executor.run(model, engines)

    # ---- Print summary table to stdout -----------------------------------
    print("\n" + "=" * 72)
    print(f"{'BENCHMARK SUMMARY':^72}")
    print("=" * 72)
    for br in batch_results:
        s = br.summary()
        print(
            f"\n  Engine : {s['engine']}\n"
            f"  Circuit: {s['circuit']}  n_qubits={s['n_qubits']}  depth={s['depth']}\n"
            f"  Runs   : {s['n_runs']}\n"
            f"  ⟨O⟩    : {s['expval_mean']:.6f} ± {s['expval_std']:.6f}\n"
            f"  Time   : {s['elapsed_mean_s']:.4f} s ± {s['elapsed_std_s']:.4f} s  "
            f"(total {s['elapsed_total_s']:.2f} s)"
        )
        if "fidelity_mean" in s:
            print(f"  Fidelity: {s['fidelity_mean']:.6f} ± {s['fidelity_std']:.6f}")
    print("=" * 72 + "\n")

    logger.info("Done. Results written to '%s'.", output_file)


if __name__ == "__main__":
    main()
