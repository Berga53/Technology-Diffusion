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

RESULTS = ROOT / "results" / "technology diffusion"
RESULTS.mkdir(parents=True, exist_ok=True)

from technology_diffusion import create_pa_graph
from technology_diffusion.approx import approx

DEFAULT_C_LIST = [1, 5, 10, 20]
DEFAULT_N_LIST = [200, 400, 600, 1000, 2000]
DEFAULT_SEED_LIST = [1, 42, 53, 101]

ALGORITHM_LABEL_WIDTH = 34
K_VALUE_WIDTH = 3
TIME_VALUE_WIDTH = 7


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


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
	if not args.auto_name_outputs:
		return args.results_csv_path, args.static_params_path

	run_tag = build_run_tag(args)
	return (
		add_tag_to_path(args.results_csv_path, run_tag),
		add_tag_to_path(args.static_params_path, run_tag),
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Run the approximation algorithm for the technology diffusion experiment."
		)
	)
	parser.add_argument("--c-list", type=int, nargs="+", default=DEFAULT_C_LIST)
	parser.add_argument("--n-list", type=int, nargs="+", default=DEFAULT_N_LIST)
	parser.add_argument("--seed-list", type=int, nargs="+", default=DEFAULT_SEED_LIST)
	parser.add_argument("--init-nodes", type=int, default=5)
	parser.add_argument("--init-mode", choices=["complete", "tree"], default="complete")
	parser.add_argument(
		"--max-time",
		type=float,
		default=None,
		help="Maximum runtime (seconds) for a single approx solve. Default: no time limit.",
	)
	parser.add_argument(
		"--results-csv-path",
		type=Path,
		default=RESULTS / "technology_diffusion_results_approx.csv",
	)
	parser.add_argument(
		"--static-params-path",
		type=Path,
		default=RESULTS / "technology_diffusion_static_params_approx.json",
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
	return parser.parse_args()


def append_result(
	rows: list[list],
	algorithm: str,
	n_nodes: int,
	c: int,
	seed: int,
	total_time: float | None,
	k_value: int | None,
	selected_seed_nodes: list[int],
) -> None:
	rows.append(
		[
			algorithm,
			n_nodes,
			c,
			seed,
			total_time,
			k_value,
			json.dumps(selected_seed_nodes),
		]
	)


def main() -> None:
	args = parse_args()
	results_csv_path, static_params_path = resolve_output_paths(args)

	combinations = list(itertools.product(args.n_list, args.c_list, args.seed_list))

	print(f"Total graph runs: {len(combinations)}")
	print(f"Results CSV: {results_csv_path}")
	print(f"Static params: {static_params_path}")

	results = []
	start_all = time.time()

	for run_idx, (n_nodes, c, seed) in enumerate(combinations, start=1):
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

		approx_k, final_x, approx_runtime, _ = approx(g, thetas, max_time=args.max_time)

		selected_nodes = (
			[idx for idx, value in enumerate(final_x) if int(value) == 1]
			if final_x is not None
			else []
		)

		append_result(
			results,
			"approx",
			n_nodes,
			c,
			seed,
			approx_runtime,
			int(approx_k) if approx_k is not None else None,
			selected_nodes,
		)

		print_algorithm_line(
			"approx",
			format_k_time(int(approx_k) if approx_k is not None else None, approx_runtime),
		)

	elapsed_all = time.time() - start_all
	columns = [
		"algorithm",
		"n_nodes",
		"c",
		"seed",
		"total_time",
		"K",
		"selected_seed_nodes",
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
		"result_columns": columns,
	}

	with static_params_path.open("w", encoding="utf-8") as file:
		json.dump(static_params, file, indent=2)
	print(f"Static parameters saved to: {static_params_path}")


if __name__ == "__main__":
	main()
