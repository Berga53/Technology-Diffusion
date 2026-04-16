
import os
import sys
import time
import random
from typing import Iterable, Mapping, Optional

import networkx as nx
import numpy as np


def suppress_print() -> None:
    sys._stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")


def resume_print() -> None:
    if hasattr(sys, "_stdout"):
        sys.stdout.close()
        sys.stdout = sys._stdout


def time_exceeded(start_time: float, max_time: Optional[float]) -> bool:
    return max_time is not None and (time.perf_counter() - start_time) > max_time


def make_vector(init_seeds: Iterable[int], n_nodes: int) -> np.ndarray:
    x0 = np.zeros(n_nodes, dtype=float)
    x0[list(init_seeds)] = 1.0
    return x0


def create_pa_graph(
    n_nodes: int,
    c: int,
    seed: Optional[int] = None,
    init_nodes: Optional[int] = None,
    init_mode: str = "complete",
) -> tuple[nx.Graph, dict[int, int]]:
    if c <= 0:
        raise ValueError("c must be a positive integer.")
    if init_nodes is None:
        init_nodes = max(2, c)
    if not (2 <= init_nodes <= n_nodes):
        raise ValueError("init_nodes must satisfy 2 <= init_nodes <= n_nodes")

    rng_graph = np.random.default_rng(seed)
    rng_theta = np.random.default_rng(seed)

    if init_mode == "complete":
        g = nx.complete_graph(init_nodes)
    elif init_mode == "tree":
        g = nx.random_tree(init_nodes, seed=seed)
    else:
        raise ValueError("init_mode must be 'complete' or 'tree'")

    m_choices = (1, 2, 3, 4)

    for u in range(init_nodes, n_nodes):
        g.add_node(u)
        m = int(rng_graph.choice(m_choices))
        existing = np.array([v for v in g.nodes if v != u], dtype=int)
        m = min(m, len(existing))
        degrees = np.array([g.degree(v) for v in existing], dtype=float)
        probs = degrees / degrees.sum()
        targets = rng_graph.choice(existing, size=m, replace=False, p=probs)
        for v in targets:
            g.add_edge(u, int(v))

    k_max = int(np.ceil(n_nodes / c))
    values = list(range(max(2, c), k_max * c + 1, c))
    theta = {v: int(rng_theta.choice(values)) for v in g.nodes()}
    nx.set_node_attributes(g, theta, name="theta")
    return g, theta

def make_subset_connected(
    g: nx.Graph,
    x: Iterable[int],
    use_thetas: int = 0,
    thetas: Mapping[int, int] | None = None,
) -> tuple[int, ...]:
    s = set(x)
    target_size = len(s)

    if not s:
        return tuple()

    if len(s) > 1:
        terminals = sorted(s)
        closure = nx.Graph()
        closure.add_nodes_from(terminals)

        for i, u in enumerate(terminals[:-1]):
            lengths, paths = nx.single_source_dijkstra(g, source=u)
            for v in terminals[i + 1 :]:
                if v in lengths:
                    closure.add_edge(u, v, weight=float(lengths[v]), path=paths[v])

        if closure.number_of_edges() > 0:
            mst = nx.minimum_spanning_tree(closure, weight="weight")
            connected_nodes = set(s)
            for u, v in mst.edges():
                connected_nodes.update(closure[u][v]["path"])
            s = connected_nodes

    while len(s) > target_size:
        induced = g.subgraph(s)
        articulation = set(nx.articulation_points(induced)) if len(s) > 2 else set()
        candidates = [node for node in s if node not in articulation]

        if not candidates:
            break

        if use_thetas and thetas is not None:
            candidates.sort(key=lambda node: (thetas[node], node))
        else:
            candidates.sort()

        s.remove(candidates[0])

    return tuple(sorted(s))


def connected_component_update(
    g: nx.Graph,
    active_mask: np.ndarray,
    thetas: Mapping[int, int] | np.ndarray,
) -> np.ndarray:
    n_nodes = g.number_of_nodes()
    actives = np.zeros(n_nodes, dtype=bool)

    if not np.any(active_mask):
        return actives

    active_nodes = np.flatnonzero(active_mask)
    comp_id_of = {}
    comp_sizes = []
    active_set = set(active_nodes)
    visited = set()
    cid = 0

    for u in active_nodes:
        if u in visited:
            continue
        stack = [u]
        visited.add(u)
        comp_nodes = [u]

        while stack:
            x = stack.pop()
            for nbr in g.neighbors(x):
                if nbr in active_set and nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
                    comp_nodes.append(nbr)

        comp_sizes.append(len(comp_nodes))
        for w in comp_nodes:
            comp_id_of[w] = cid
        cid += 1

    comp_sizes = np.asarray(comp_sizes, dtype=int)
    inactive_nodes = np.flatnonzero(~active_mask)

    for v in inactive_nodes:
        touched = set()
        for u in g.neighbors(v):
            if active_mask[u]:
                c = comp_id_of.get(u)
                if c is not None:
                    touched.add(c)

        if touched:
            reachable = int(comp_sizes[list(touched)].sum())
            if reachable >= (int(thetas[v])-1):
                actives[v] = True

    return actives


def connected_component_spread(
    g: nx.Graph,
    x0: np.ndarray,
    thetas: Mapping[int, int] | np.ndarray,
    max_t: int = 1000,
) -> tuple[int, list[int], np.ndarray]:
    n_nodes = g.number_of_nodes()
    active = (x0 > 0).astype(bool)
    spread_hist = [int(np.sum(active))]

    for _ in range(max_t):
        prev = active.copy()
        new = connected_component_update(g, prev, thetas)
        active = prev | new
        if np.array_equal(active, prev):
            break
        spread_hist.append(int(np.sum(active)))

    final_spread = int(np.sum(active))
    final_x = np.zeros(n_nodes, dtype=float)
    final_x[active] = 1.0
    return final_spread, spread_hist, final_x
