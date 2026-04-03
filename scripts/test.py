from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.technology_diffusion.helpers import create_pa_graph, connected_component_spread
from src.technology_diffusion.heuristics import random_start


def run_benchmark(
	n_nodes: int,
	c: int,
	repeats: int,
	seed_fraction: float,
	connected: int,
	graph_seed: int,
	run_seed_start: int,
) -> None:
	g, thetas = create_pa_graph(n_nodes=n_nodes, c=c, seed=graph_seed)
	k = max(1, int(n_nodes * seed_fraction))

	random.seed(run_seed_start - 1)
	warmup_x0 = random_start(g, n_nodes, k, thetas, connected)
	connected_component_spread(g, warmup_x0, thetas)

	times = []
	final_spreads = []
	hist_lengths = []

	for i in range(repeats):
		random.seed(run_seed_start + i)
		x0 = random_start(g, n_nodes, k, thetas, connected)

		t0 = perf_counter()
		final_spread, spread_hist, _ = connected_component_spread(g, x0, thetas)
		elapsed = perf_counter() - t0

		times.append(elapsed)
		final_spreads.append(final_spread)
		hist_lengths.append(len(spread_hist))

	mean_t = statistics.mean(times)
	med_t = statistics.median(times)
	std_t = statistics.pstdev(times) if len(times) > 1 else 0.0

	print("=== connected_component_spread benchmark ===")
	print(f"n_nodes:            {n_nodes}")
	print(f"c:                  {c}")
	print(f"k (initial seeds):  {k}")
	print(f"connected starts:   {connected}")
	print(f"seed_fraction:      {seed_fraction}")
	print(f"repeats:            {repeats}")
	print(f"graph_seed:         {graph_seed}")
	print("--- timing (seconds) ---")
	print(f"mean:               {mean_t:.6f}")
	print(f"median:             {med_t:.6f}")
	print(f"std (population):   {std_t:.6f}")
	print(f"min:                {min(times):.6f}")
	print(f"max:                {max(times):.6f}")
	print("--- run outcomes ---")
	print(f"avg final spread:   {statistics.mean(final_spreads):.2f}")
	print(f"avg #iterations:    {statistics.mean(hist_lengths):.2f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Benchmark connected_component_spread on a synthetic PA graph."
	)
	parser.add_argument("--n-nodes", type=int, default=5000, help="Number of graph nodes.")
	parser.add_argument("--c", type=int, default=5, help="Theta spacing parameter.")
	parser.add_argument("--repeats", type=int, default=10, help="Number of timed runs.")
	parser.add_argument(
		"--connected",
		type=int,
		default=1,
		help="Pass-through connected flag for random_start (0 or 1).",
	)
	parser.add_argument(
		"--seed-fraction",
		type=float,
		default=0.01,
		help="Fraction of initially active nodes in each run.",
	)
	parser.add_argument("--graph-seed", type=int, default=42, help="Seed for graph generation.")
	parser.add_argument(
		"--run-seed-start",
		type=int,
		default=1000,
		help="Starting seed for sampling initial active sets.",
	)
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()
	run_benchmark(
		n_nodes=args.n_nodes,
		c=args.c,
		repeats=args.repeats,
		seed_fraction=args.seed_fraction,
		connected=args.connected,
		graph_seed=args.graph_seed,
		run_seed_start=args.run_seed_start,
	)
