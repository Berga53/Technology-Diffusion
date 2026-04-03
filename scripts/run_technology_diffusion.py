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

RESULTS = ROOT / "results" / "technology diffusion"
RESULTS.mkdir(parents=True, exist_ok=True)

from technology_diffusion import (
    NaDS_technology_diffusion_binary_search,
    degree_discount,
    betweenness,
    build_golberg_liu_ip,
    create_pa_graph,
    degree,
    degree_connected,
    degree_threshold,
    high_thetas,
    random_start,
    technology_diffusion_heuristics,
)

DEFAULT_C_LIST = [1, 5, 10, 20]
DEFAULT_N_LIST = [200, 400, 600, 1000, 2000]
DEFAULT_SEED_LIST = [1, 42, 53, 99, 101]

ALGORITHM_LABEL_WIDTH = 34
K_VALUE_WIDTH = 3
TIME_VALUE_WIDTH = 7

NADS_STRATEGY_FUNCTIONS = {
    "high_thetas": high_thetas,
    "degree_threshold": degree_threshold,
    "random_start": random_start,
    "degree_discount": degree_discount,
    "degree_connected": degree_connected,
    "degree": degree,
    "betweenness": betweenness,
}

DEFAULT_NADS_STRATEGY = ["high_thetas", "degree_threshold", "random_start"]


def print_algorithm_line(algorithm: str, message: str) -> None:
    print(f"{algorithm:<{ALGORITHM_LABEL_WIDTH}} | {message}")


def format_k_time(k_value: int | None, total_time: float | None) -> str:
    k_text = "None" if k_value is None else str(k_value)
    time_text = "None" if total_time is None else f"{total_time:.2f}s"
    return f"k: {k_text:>{K_VALUE_WIDTH}} | time: {time_text:>{TIME_VALUE_WIDTH}}"


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
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic heuristics, Goldberg-Liu, and NaDS for the technology diffusion experiment."
        )
    )
    parser.add_argument("--c-list", type=int, nargs="+", default=DEFAULT_C_LIST)
    parser.add_argument("--n-list", type=int, nargs="+", default=DEFAULT_N_LIST)
    parser.add_argument("--seed-list", type=int, nargs="+", default=DEFAULT_SEED_LIST)
    parser.add_argument("--init-nodes", type=int, default=5)
    parser.add_argument("--init-mode", choices=["complete", "tree"], default="complete")
    parser.add_argument("--connected", type=int, default=1)
    parser.add_argument("--delta", type=float, default=0.5)
    parser.add_argument("--xi", type=float, default=0.1)
    parser.add_argument("--d", type=int, default=2)
    parser.add_argument("--buffer-dim", type=int, default=5000)
    parser.add_argument("--min-conn", type=int, default=20)
    parser.add_argument("--mg-max-depth", type=int, default=5)
    parser.add_argument("--mg-memory-len", type=int, default=10)
    parser.add_argument(
        "--nads-strategy",
        type=str,
        nargs="+",
        choices=sorted(NADS_STRATEGY_FUNCTIONS.keys()),
        default=DEFAULT_NADS_STRATEGY,
        help="Ordered list of strategy initializers to try in NaDS binary search. ",
    )
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
        help="Skip Goldberg-Liu optimize for runs with n_nodes >= this threshold.",
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
        help="Append a tag derived from seed selections to output file names.",
    )
    parser.add_argument("--save-results-csv", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-gurobi-log", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_nads_strategy(strategy_names: list[str]) -> list:
    return [NADS_STRATEGY_FUNCTIONS[name] for name in strategy_names]


def build_heuristics() -> list:
    return [
        degree_threshold,
        high_thetas,
        degree_discount,
        degree_connected,
        degree,
        betweenness,
    ]


def configure_gurobi_env(save_gurobi_log: bool, gurobi_log_path: Path) -> gp.Env:
    gurobi_env = gp.Env(empty=True)
    gurobi_env.setParam("OutputFlag", 1 if save_gurobi_log else 0)
    gurobi_env.setParam("LogToConsole", 0)
    gurobi_env.setParam("LogFile", str(gurobi_log_path) if save_gurobi_log else "")
    gurobi_env.start()
    return gurobi_env


def append_result(
    rows: list[list],
    algorithm: str,
    n_nodes: int,
    c: int,
    seed: int,
    total_time: float | None,
    k_value: int | None,
    history: list | None,
) -> None:
    rows.append(
        [
            algorithm,
            n_nodes,
            c,
            seed,
            total_time,
            k_value,
            json.dumps(history if history is not None else None),
        ]
    )


def run_goldberg_liu(
    g,
    thetas,
    n_nodes: int,
    c: int,
    seed: int,
    max_time: float,
    skip_gurobi_from_n: int,
    gurobi_env: gp.Env,
) -> tuple[float | None, int | None, list | None]:
    if n_nodes >= skip_gurobi_from_n:
        print(f"Skipping Goldberg-Liu IP for N={n_nodes} (threshold: {skip_gurobi_from_n}).")
        print_algorithm_line(
            "goldberg_liu",
            f"{format_k_time(None, None)} | skipped for N={n_nodes} (threshold={skip_gurobi_from_n})",
        )
        return None, None, None

    gl_start = time.time()
    print("Building Goldberg-Liu IP...", end="", flush=True)
    model, _ = build_golberg_liu_ip(g, thetas, max_time, env=gurobi_env)
    build_time = time.time() - gl_start

    if model is None:
        print(
            f"\rGoldberg-Liu IP build failed: time limit reached during model construction ({round(build_time, 2)}s)."
            + " " * 20
        )
        print_algorithm_line(
            "goldberg_liu",
            f"{format_k_time(None, None)} | build failed (time limit reached during model construction)",
        )
        return None, None, None

    print(f"\rGoldberg-Liu IP built in {round(build_time, 2)} seconds!" + " " * 20)

    gl_history = [[n_nodes, 0.0]]

    def gl_callback(callback_model: gp.Model, where: int) -> None:
        if where == gp.GRB.Callback.MIPSOL:
            obj = callback_model.cbGet(gp.GRB.Callback.MIPSOL_OBJ)
            runtime = callback_model.cbGet(gp.GRB.Callback.RUNTIME)
            gl_history.append([int(round(obj)), round(runtime + build_time, 4)])

    remaining = max_time - build_time
    if remaining <= 0:
        model.dispose()
        print(
            f"Goldberg-Liu IP build failed: build time {round(build_time, 2)}s exceeds max time {round(max_time, 2)}s."
        )
        print_algorithm_line(
            "goldberg_liu",
            f"{format_k_time(None, None)} | build time exceeded max time ({round(build_time, 2)}s > {round(max_time, 2)}s)",
        )
        return None, None, None

    model.setParam("TimeLimit", remaining)
    print(
        f"Solving Goldberg-Liu IP with time limit {round(remaining, 2)} seconds...",
        end="",
        flush=True,
    )
    model.optimize(gl_callback)
    print(f"\rGoldberg-Liu IP solved in {round(float(model.Runtime), 2)} seconds!" + " " * 20)

    if model.SolCount > 0:
        gl_k = int(round(model.objVal))
        total_time = round(float(time.time() - gl_start), 4)
        gl_history.append([gl_k, total_time])
        print_algorithm_line(
            "goldberg_liu",
            format_k_time(gl_k, total_time),
        )
    else:
        gl_k = None
        total_time = None
        gl_history = None
        print_algorithm_line(
            "goldberg_liu",
            f"{format_k_time(None, None)} | no feasible solution found",
        )

    model.dispose()
    return total_time, gl_k, gl_history


def main() -> None:
    args = parse_args()
    nads_strategy = build_nads_strategy(args.nads_strategy)
    heuristics = build_heuristics()
    results_csv_path, gurobi_log_path, static_params_path = resolve_output_paths(args)

    combinations = list(itertools.product(args.n_list, args.c_list, args.seed_list))

    print(f"Total graph runs: {len(combinations)}")
    print(f"Deterministic heuristics: {len(heuristics)}")
    print(f"Results CSV: {results_csv_path}")
    print(f"Gurobi log: {gurobi_log_path}")
    print(f"Static params: {static_params_path}")

    results = []
    start_all = time.time()
    gurobi_env = configure_gurobi_env(args.save_gurobi_log, gurobi_log_path)

    try:
        for run_idx, (n_nodes, c, seed) in enumerate(combinations, start=1):
            max_time = args.max_time_scale * (n_nodes // 100)

            print("\n" + "=" * 90)
            print(
                f"Run {run_idx}/{len(combinations)} | "
                f"N={n_nodes}, c={c}, seed={seed}, init_nodes={args.init_nodes}, init_mode={args.init_mode}"
            )

            if args.save_gurobi_log:
                with gurobi_log_path.open("a", encoding="utf-8") as log_f:
                    log_f.write(
                        f"\n{'='*60}\n"
                        f"Run {run_idx}/{len(combinations)} | "
                        f"n_nodes={n_nodes}, c={c}, seed={seed}, "
                        f"init_nodes={args.init_nodes}, init_mode={args.init_mode}\n"
                        f"{'='*60}\n"
                    )

            g, thetas = create_pa_graph(
                n_nodes=n_nodes,
                c=c,
                seed=seed,
                init_nodes=args.init_nodes,
                init_mode=args.init_mode,
            )

            for heuristic_fn in heuristics:
                h_name = heuristic_fn.__name__
                _, h_k, h_runtime, h_history = technology_diffusion_heuristics(
                    g,
                    n_nodes,
                    thetas=thetas,
                    connected=args.connected,
                    heuristic=heuristic_fn,
                )
                append_result(
                    results,
                    h_name,
                    n_nodes,
                    c,
                    seed,
                    round(float(h_runtime), 4),
                    int(h_k) if h_k is not None else None,
                    [list(event) for event in h_history] if h_history is not None else None,
                )
                print_algorithm_line(
                    f"heuristic: {h_name}",
                    format_k_time(int(h_k) if h_k is not None else None, float(h_runtime)),
                )

            gl_total_time, gl_k, gl_history = run_goldberg_liu(
                g,
                thetas,
                n_nodes,
                c,
                seed,
                max_time,
                args.skip_gurobi_from_n,
                gurobi_env,
            )
            append_result(
                results,
                "goldberg_liu",
                n_nodes,
                c,
                seed,
                gl_total_time,
                gl_k,
                gl_history,
            )

            nads_k, final_x, nads_runtime, nads_history = NaDS_technology_diffusion_binary_search(
                g,
                thetas,
                nads_strategy,
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
            append_result(
                results,
                "nads_binary_search",
                n_nodes,
                c,
                seed,
                round(float(nads_runtime), 4),
                int(nads_k) if nads_k is not None else None,
                [list(event) for event in nads_history] if nads_history is not None else None,
            )
            print_algorithm_line(
                "nads_binary_search",
                format_k_time(int(nads_k) if nads_k is not None else None, float(nads_runtime)),
            )

            if final_x is not None:
                del final_x

    finally:
        gurobi_env.dispose()

    elapsed_all = time.time() - start_all
    columns = [
        "algorithm",
        "n_nodes",
        "c",
        "seed",
        "total_time",
        "K",
        "history",
    ]

    results_df = pd.DataFrame(results, columns=columns)
    results_df = results_df.sort_values(["n_nodes", "c", "seed", "algorithm"]).reset_index(drop=True)

    print("\n" + "=" * 90)
    print(f"Completed {len(results_df)} algorithm evaluations in {round(elapsed_all, 2)} seconds.")

    if args.save_results_csv:
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to: {results_csv_path}")

    static_params = {
        "init_nodes": args.init_nodes,
        "init_mode": args.init_mode,
        "connected": args.connected,
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
        "heuristics": [fn.__name__ for fn in heuristics],
        "nads_strategy": [fn.__name__ for fn in nads_strategy],
        "result_columns": columns,
    }

    with static_params_path.open("w", encoding="utf-8") as file:
        json.dump(static_params, file, indent=2)
    print(f"Static parameters saved to: {static_params_path}")

    if args.save_gurobi_log:
        print(f"Gurobi log saved to: {gurobi_log_path}")


if __name__ == "__main__":
    main()
