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
    build_exact_ip,
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
    parser = argparse.ArgumentParser(description="Run technology diffusion experiment comparing NS, Goldberg-Liu, and Exact IP.")
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
        "--skip-exact-from-n",
        type=int,
        default=20,
        help="Skip Exact IP build/optimize for runs with n_nodes >= this threshold.",
    )
    parser.add_argument(
        "--exact-time-horizon",
        type=int,
        default=None,
        help="Optional time horizon for Exact IP. Defaults to n_nodes.",
    )
    parser.add_argument(
        "--exact-use-simultaneous",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use simultaneous propagation constraints in Exact IP.",
    )
    parser.add_argument(
        "--results-csv-path",
        type=Path,
        default=RESULTS / "technology_diffusion_exact_results.csv",
    )
    parser.add_argument(
        "--gurobi-log-path",
        type=Path,
        default=RESULTS / "gurobi_exact.log",
    )
    parser.add_argument(
        "--static-params-path",
        type=Path,
        default=RESULTS / "technology_diffusion_exact_static_params.json",
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
            gl_model = None
            exact_model = None
            gl_x = None
            exact_x = None
            exact_y = None
            exact_n = None

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
            else:
                print("Building Goldberg-Liu IP...", end="")
                gl_build_start = time.time()
                gl_model, gl_x = build_golberg_liu_ip(g, thetas, max_time, env=gurobi_env)
                build_time = time.time() - gl_build_start

                gl_history = [(n_nodes, 0.0)]

                def gl_callback(model: gp.Model, where: int) -> None:
                    if where == gp.GRB.Callback.MIPSOL:
                        obj = model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
                        runtime = model.cbGet(gp.GRB.Callback.RUNTIME)
                        gl_history.append([int(round(obj)), round(runtime + build_time, 4)])

                if gl_model is None or max_time - build_time < 0:
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
                    print(
                        f"Solving Goldberg-Liu IP with time limit {round(max_time - build_time, 2)} seconds...",
                        end="",
                    )
                    suppress_print()
                    gl_model.setParam("TimeLimit", max_time - build_time)
                    resume_print()
                    gl_model.optimize(gl_callback)
                    print(f"\rGoldberg-Liu IP solved in {round(gl_model.Runtime, 2)} seconds!" + " " * 20)

                    gl_build_failed = False
                    gl_build_time = round(build_time, 4)
                    gl_opt_time = round(float(gl_model.Runtime), 4)
                    gl_runtime = round(build_time + gl_opt_time, 4)
                    gl_k = int(round(gl_model.objVal)) if gl_model.SolCount > 0 else None
                    if gl_model.SolCount > 0:
                        gl_history.append([gl_k, gl_runtime])
                        gl_history_json = json.dumps(gl_history)
                    else:
                        gl_history_json = json.dumps([])

            exact_skipped_for_size = n_nodes >= args.skip_exact_from_n
            if exact_skipped_for_size:
                print(f"Skipping Exact IP for N={n_nodes} (threshold: {args.skip_exact_from_n}).")
                exact_build_failed = True
                exact_build_time = None
                exact_opt_time = None
                exact_runtime = None
                exact_k = None
            else:
                print("Building Exact IP...", end="")
                exact_build_start = time.time()
                exact_model, exact_x, exact_y, exact_n = build_exact_ip(
                    g,
                    thetas,
                    k=n_nodes,
                    use_simultaneous=args.exact_use_simultaneous,
                    time_horizon=args.exact_time_horizon,
                    max_time=max_time,
                )
                exact_build_elapsed = time.time() - exact_build_start

                if exact_model is None or max_time - exact_build_elapsed < 0:
                    print(
                        f"\rExact IP build failed: build time {round(exact_build_elapsed, 2)}s exceeds max time {max_time}s."
                        + " " * 20
                    )
                    exact_build_failed = True
                    exact_build_time = None
                    exact_opt_time = None
                    exact_runtime = None
                    exact_k = None
                else:
                    print(f"\rExact IP built in {round(exact_build_elapsed, 2)} seconds!" + " " * 20)
                    print(
                        f"Solving Exact IP with time limit {round(max_time - exact_build_elapsed, 2)} seconds...",
                        end="",
                    )
                    exact_model.setParam("OutputFlag", 1 if args.save_gurobi_log else 0)
                    exact_model.setParam("LogToConsole", 0)
                    exact_model.setParam("LogFile", str(gurobi_log_path) if args.save_gurobi_log else "")
                    exact_model.setParam("TimeLimit", max_time - exact_build_elapsed)
                    exact_model.optimize()
                    print(f"\rExact IP solved in {round(exact_model.Runtime, 2)} seconds!" + " " * 20)

                    exact_build_failed = False
                    exact_build_time = round(exact_build_elapsed, 4)
                    exact_opt_time = round(float(exact_model.Runtime), 4)
                    exact_runtime = round(exact_build_elapsed + exact_opt_time, 4)
                    exact_k = int(round(exact_model.objVal)) if exact_model.SolCount > 0 else None

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
            ns_gl_gap = (ns_k - gl_k) if (ns_k is not None and gl_k is not None) else None
            ns_exact_gap = (ns_k - exact_k) if (ns_k is not None and exact_k is not None) else None
            gl_exact_gap = (gl_k - exact_k) if (gl_k is not None and exact_k is not None) else None

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
                    exact_build_failed,
                    exact_build_time,
                    exact_opt_time,
                    exact_runtime,
                    exact_k,
                    ns_runtime,
                    ns_k,
                    ns_gl_gap,
                    ns_exact_gap,
                    gl_exact_gap,
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
                print(f"Goldberg-Liu -> K={gl_k}, runtime={round(float(gl_runtime), 2)}s")

            if exact_build_failed:
                if exact_skipped_for_size:
                    print("Exact IP     -> Skipped for size threshold.")
                else:
                    print("Exact IP     -> Build failed.")
            else:
                print(f"Exact IP     -> K={exact_k}, runtime={round(float(exact_runtime), 2)}s")

            print(
                f"NS Binary    -> K={ns_k}, runtime={round(float(ns_runtime), 2)}s, "
                f"gap(NS-GL)={ns_gl_gap}, gap(NS-EX)={ns_exact_gap}"
            )

            if gl_x is not None:
                del gl_x
            if gl_model is not None:
                gl_model.dispose()
                del gl_model
            if exact_n is not None:
                del exact_n
            if exact_y is not None:
                del exact_y
            if exact_x is not None:
                del exact_x
            if exact_model is not None:
                exact_model.dispose()
                del exact_model
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
        "EX_build_failed",
        "EX_build_time_s",
        "EX_optimization_time_s",
        "EX_runtime_s",
        "EX_K",
        "NS_runtime_s",
        "NS_K",
        "NS_GL_gap",
        "NS_EX_gap",
        "GL_EX_gap",
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
        "skip_exact_from_n": args.skip_exact_from_n,
        "exact_time_horizon": args.exact_time_horizon,
        "exact_use_simultaneous": args.exact_use_simultaneous,
        "strategy_names": [fn.__name__ for fn in strategy],
    }

    with static_params_path.open("w", encoding="utf-8") as file:
        json.dump(static_params, file, indent=2)
    print(f"Static parameters saved to: {static_params_path}")

    if args.save_gurobi_log:
        print(f"Gurobi log saved to: {gurobi_log_path}")


if __name__ == "__main__":
    main()
