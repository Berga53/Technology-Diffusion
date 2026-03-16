import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import gurobipy as gp
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS = ROOT / "experiments" / "technology diffusion"
RESULTS.mkdir(parents=True, exist_ok=True)

from technology_diffusion import (
    NS_technology_diffusion_binary_search,
    SD_start,
    betweenness,
    build_golberg_liu_ip,
    create_pa_graph,
    degree,
    degree_connected,
    degree_threshold,
    high_thetas_start,
    random_start,
    resume_print,
    suppress_print,
)

DEFAULT_C_LIST = [1, 5, 10, 20]
DEFAULT_N_LIST = [200, 400, 600, 1000, 2000]
DEFAULT_SEED_LIST = [1, 42, 53, 99, 101]


def _format_values_for_name(prefix: str, values: list[int]) -> str:
    if not values:
        return f"{prefix}-none"
    unique_values = sorted(set(values))
    if len(unique_values) == 1:
        return f"{prefix}-{unique_values[0]}"
    joined = "-".join(str(value) for value in unique_values)
    return f"{prefix}s-{joined}"


def _sanitize_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip()).strip("-_.")


def build_run_tag(args: argparse.Namespace) -> str:
    parts = [_format_values_for_name("seed", args.seed_list)]
    if args.run_label:
        sanitized = _sanitize_label(args.run_label)
        if sanitized:
            parts.append(sanitized)
    return "__".join(parts)


def add_tag_to_path(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.stem}__{tag}{path.suffix}")


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if not args.auto_name_outputs:
        return args.results_csv_path, args.gurobi_log_path, args.static_params_path

    run_tag = build_run_tag(args)
    return (
        add_tag_to_path(args.results_csv_path, run_tag),
        add_tag_to_path(args.gurobi_log_path, run_tag),
        add_tag_to_path(args.static_params_path, run_tag),
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the technology diffusion experiment.")
    parser.add_argument("--c-list", type=int, nargs="+", default=DEFAULT_C_LIST)
    parser.add_argument("--n-list", type=int, nargs="+", default=DEFAULT_N_LIST)
    parser.add_argument("--seed-list", type=int, nargs="+", default=DEFAULT_SEED_LIST)
    parser.add_argument("--init-nodes", type=int, default=5)
    parser.add_argument("--init-mode", choices=["complete", "tree"], default="complete")
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--xi", type=float, default=0.1)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--buffer-dim", type=int, default=5000)
    parser.add_argument("--min-conn", type=int, default=20)
    parser.add_argument("--mg-max-depth", type=int, default=5)
    parser.add_argument("--mg-memory-len", type=int, default=10)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--max-time-scale",
        type=float,
        default=180.0,
        help="Per-run time limit scale used as max_time_scale * (n_nodes // 100).",
    )
    parser.add_argument(
        "--skip-gurobi-from-n",
        type=int,
        default=1000,
        help="Skip Goldberg-Liu IP build/optimize for runs with n_nodes >= this threshold.",
    )
    parser.add_argument(
        "--results-csv-path",
        type=Path,
        default=RESULTS / "technology_diffusion_results.csv",
    )
    parser.add_argument(
        "--gurobi-log-path",
        type=Path,
        default=RESULTS / "gurobi.log",
    )
    parser.add_argument(
        "--static-params-path",
        type=Path,
        default=RESULTS / "technology_diffusion_static_params.json",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="",
        help="Optional label appended to output file names, useful for reruns of the same batch.",
    )
    parser.add_argument(
        "--auto-name-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append a tag derived from n/c/seed selections to output file names.",
    )
    parser.add_argument("--save-results-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-gurobi-log", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_strategy() -> list:
    return [
        degree_threshold,
        high_thetas_start,
        SD_start,
        degree_connected,
        degree,
        betweenness,
        random_start,
    ]


def configure_gurobi_env(save_gurobi_log: bool, gurobi_log_path: Path) -> gp.Env:
    gurobi_env = gp.Env(empty=True)
    gurobi_env.setParam("OutputFlag", 1 if save_gurobi_log else 0)
    gurobi_env.setParam("LogToConsole", 0)
    gurobi_env.setParam("LogFile", str(gurobi_log_path) if save_gurobi_log else "")
    gurobi_env.start()
    return gurobi_env


def main() -> None:
    args = parse_args()
    strategy = build_strategy()
    results_csv_path, gurobi_log_path, static_params_path = resolve_output_paths(args)

    combinations = list(itertools.product(args.n_list, args.c_list, args.seed_list))

    print(f"Total runs: {len(combinations)}")
    print(f"Results CSV: {results_csv_path}")
    print(f"Gurobi log: {gurobi_log_path}")
    print(f"Static params: {static_params_path}")
    results = []
    start_all = time.time()
    gurobi_env = configure_gurobi_env(args.save_gurobi_log, gurobi_log_path)

    try:
        for run_idx, (n_nodes, c, seed) in enumerate(combinations, start=1):
            max_time = args.max_time_scale * (n_nodes // 100)
            model = None
            
            print("\n" + "=" * 90)
            print(
                f"Run {run_idx}/{len(combinations)} | "
                f"N={n_nodes}, c={c}, seed={seed}, init_nodes={args.init_nodes}, init_mode={args.init_mode}"
            )

            g, thetas = create_pa_graph(
                n_nodes=n_nodes,
                c=c,
                seed=seed,
                init_nodes=args.init_nodes,
                init_mode=args.init_mode,
            )

            gl_skipped_for_size = n_nodes >= args.skip_gurobi_from_n
            if gl_skipped_for_size:
                print(
                    f"Skipping Goldberg-Liu IP for N={n_nodes} (threshold: {args.skip_gurobi_from_n})."
                )
                gl_build_failed = True
                gl_build_time = None
                gl_opt_time = None
                gl_runtime = None
                gl_k = None
                gl_history_json = json.dumps([])
                x = None
            else:
                print("Building Goldberg-Liu IP...", end="")
                start = time.time()
                model, x = build_golberg_liu_ip(g, thetas, max_time, env=gurobi_env)
                build_time = time.time() - start

                gl_history = [(n_nodes, 0.0)]

                def gl_callback(model: gp.Model, where: int) -> None:
                    if where == gp.GRB.Callback.MIPSOL:
                        obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                        runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
                        gl_history.append([int(round(obj)), round(runtime + build_time, 4)])

                if max_time - build_time < 0:
                    print(
                        f"\rGoldberg-Liu IP build failed: build time {round(build_time, 2)}s exceeds max time {max_time}s."
                        + " " * 20
                    )
                    gl_build_failed = True
                    gl_build_time = None
                    gl_opt_time = None
                    gl_runtime = None
                    gl_k = None
                    gl_history_json = json.dumps([])
                else:
                    print(f"\rGoldberg-Liu IP built in {round(build_time, 2)} seconds!" + " " * 20)
                    print(f"Solving Goldberg-Liu IP with time limit {round(max_time - build_time, 2)} seconds...", end="")
                    suppress_print()
                    model.setParam("TimeLimit", max_time - build_time)
                    resume_print()
                    model.optimize(gl_callback)
                    print(f"\rGoldberg-Liu IP solved in {round(model.Runtime, 2)} seconds!" + " " * 20)

                    gl_build_failed = False
                    gl_build_time = round(build_time, 4)
                    gl_opt_time = round(float(model.Runtime), 4)
                    gl_runtime = round(build_time + gl_opt_time, 4)
                    gl_k = int(round(model.objVal)) if model.SolCount > 0 else None
                    if model.SolCount > 0:
                        gl_history.append([gl_k, gl_runtime])
                        gl_history_json = json.dumps(gl_history)
                    else:
                        gl_history_json = json.dumps([])

            ns_k, final_x, ns_runtime, ns_history = NS_technology_diffusion_binary_search(
                g,
                thetas,
                strategy,
                args.delta,
                args.xi,
                args.d,
                args.min_conn,
                args.mg_max_depth,
                args.mg_memory_len,
                max_time,
                args.buffer_dim,
                args.verbose,
            )

            ns_history_json = json.dumps([tuple(event) for event in ns_history])
            gap = (ns_k - gl_k) if (ns_k is not None and gl_k is not None) else None

            results.append(
                [
                    n_nodes,
                    c,
                    seed,
                    g.number_of_nodes(),
                    g.number_of_edges(),
                    max_time,
                    gl_build_failed,
                    gl_build_time,
                    gl_opt_time,
                    gl_runtime,
                    gl_k,
                    ns_runtime,
                    ns_k,
                    gap,
                    gl_history_json,
                    ns_history_json,
                ]
            )

            if gl_build_failed:
                if gl_skipped_for_size:
                    print("Goldberg-Liu -> Skipped for size threshold.")
                else:
                    print("Goldberg-Liu -> Build failed.")
            else:
                print(f"Goldberg-Liu -> K={gl_k}, runtime={round(gl_runtime, 2)}s")
            print(f"NS Binary    -> K={ns_k}, runtime={round(float(ns_runtime), 2)}s, gap(NS-GL)={gap}")

            if x is not None:
                del x
            if model is not None:
                model.dispose()
                del model
            del final_x
    finally:
        gurobi_env.dispose()

    elapsed_all = time.time() - start_all
    columns = [
        "n_nodes",
        "c",
        "seed",
        "num_nodes",
        "num_edges",
        "max_time",
        "GL_build_failed",
        "GL_build_time_s",
        "GL_optimization_time_s",
        "GL_runtime_s",
        "GL_K",
        "NS_runtime_s",
        "NS_K",
        "NS_GL_gap",
        "GL_history_json",
        "NS_history_json",
    ]

    results_df = pd.DataFrame(results, columns=columns)
    results_df = results_df.sort_values(["n_nodes", "c", "seed"]).reset_index(drop=True)

    print("\n" + "=" * 90)
    print(f"Completed {len(results_df)} runs in {round(elapsed_all, 2)} seconds.")

    if args.save_results_csv:
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to: {results_csv_path}")

    static_params = {
        "init_nodes": args.init_nodes,
        "init_mode": args.init_mode,
        "delta": args.delta,
        "xi": args.xi,
        "d": args.d,
        "buffer_dim": args.buffer_dim,
        "min_conn": args.min_conn,
        "mg_max_depth": args.mg_max_depth,
        "mg_memory_len": args.mg_memory_len,
        "verbose": args.verbose,
        "max_time_scale": args.max_time_scale,
        "skip_gurobi_from_n": args.skip_gurobi_from_n,
        "strategy_names": [fn.__name__ for fn in strategy],
    }

    with static_params_path.open("w", encoding="utf-8") as file:
        json.dump(static_params, file, indent=2)
    print(f"Static parameters saved to: {static_params_path}")

    if args.save_gurobi_log:
        print(f"Gurobi log saved to: {gurobi_log_path}")


if __name__ == "__main__":
    main()