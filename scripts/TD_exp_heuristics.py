import argparse
import itertools
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

RESULTS = ROOT / "experiments" / "technology diffusion"
RESULTS.mkdir(parents=True, exist_ok=True)

from technology_diffusion import (
    SD_start,
    betweenness,
    create_pa_graph,
    degree,
    degree_connected,
    degree_threshold,
    high_thetas_start,
    random_start,
    technology_diffusion_heuristics,
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


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if not args.auto_name_outputs:
        return args.results_csv_path, args.static_params_path

    run_tag = build_run_tag(args)
    return (
        add_tag_to_path(args.results_csv_path, run_tag),
        add_tag_to_path(args.static_params_path, run_tag),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the technology diffusion heuristics experiment.")
    parser.add_argument("--c-list", type=int, nargs="+", default=DEFAULT_C_LIST)
    parser.add_argument("--n-list", type=int, nargs="+", default=DEFAULT_N_LIST)
    parser.add_argument("--seed-list", type=int, nargs="+", default=DEFAULT_SEED_LIST)
    parser.add_argument("--init-nodes", type=int, default=5)
    parser.add_argument("--init-mode", choices=["complete", "tree"], default="complete")
    parser.add_argument("--connected", type=int, default=1)
    parser.add_argument(
        "--results-csv-path",
        type=Path,
        default=RESULTS / "technology_diffusion_heuristics_results.csv",
    )
    parser.add_argument(
        "--static-params-path",
        type=Path,
        default=RESULTS / "technology_diffusion_heuristics_static_params.json",
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
    return parser.parse_args()


def build_heuristics() -> list:
    return [
        degree_threshold,
        high_thetas_start,
        SD_start,
        degree_connected,
        degree,
        betweenness,
        random_start,
    ]


def main() -> None:
    args = parse_args()
    heuristics = build_heuristics()
    results_csv_path, static_params_path = resolve_output_paths(args)

    combinations = list(itertools.product(args.n_list, args.c_list, args.seed_list))

    print(f"Total graph runs: {len(combinations)}")
    print(f"Heuristics per graph: {len(heuristics)}")
    print(f"Total heuristic evaluations: {len(combinations) * len(heuristics)}")
    print(f"Results CSV: {results_csv_path}")
    print(f"Static params: {static_params_path}")

    results = []
    start_all = time.time()
    for run_idx, (n_nodes, c, seed) in enumerate(combinations, start=1):
        print("\n" + "=" * 90)
        print(
            f"Graph run {run_idx}/{len(combinations)} | "
            f"N={n_nodes}, c={c}, seed={seed}, init_nodes={args.init_nodes}, init_mode={args.init_mode}"
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
            h_start = time.time()
            _, h_k, _ = technology_diffusion_heuristics(
                g,
                n_nodes,
                thetas=thetas,
                connected=args.connected,
                heuristic=heuristic_fn,
            )
            h_runtime = round(time.time() - h_start, 4)

            results.append(
                [
                    n_nodes,
                    c,
                    seed,
                    g.number_of_nodes(),
                    g.number_of_edges(),
                    h_name,
                    h_runtime,
                    h_k,
                ]
            )

            print(f"{h_name:<14} -> K={h_k}, runtime={round(float(h_runtime), 2)}s")

    elapsed_all = time.time() - start_all
    columns = [
        "n_nodes",
        "c",
        "seed",
        "num_nodes",
        "num_edges",
        "heuristic_name",
        "heuristic_runtime_s",
        "heuristic_K",
    ]

    results_df = pd.DataFrame(results, columns=columns)
    results_df = results_df.sort_values(["n_nodes", "c", "seed", "heuristic_name"]).reset_index(drop=True)

    print("\n" + "=" * 90)
    print(f"Completed {len(results_df)} heuristic evaluations in {round(elapsed_all, 2)} seconds.")

    if args.save_results_csv:
        results_df.to_csv(results_csv_path, index=False)
        print(f"Results saved to: {results_csv_path}")

    static_params = {
        "init_nodes": args.init_nodes,
        "init_mode": args.init_mode,
        "connected": args.connected,
        "heuristic_names": [fn.__name__ for fn in heuristics],
    }

    with static_params_path.open("w", encoding="utf-8") as file:
        json.dump(static_params, file, indent=2)
    print(f"Static parameters saved to: {static_params_path}")


if __name__ == "__main__":
    main()
