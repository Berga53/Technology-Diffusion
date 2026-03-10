import os
import sys
import time
import random
from typing import Iterable, Optional

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
    return max_time is not None and (time.time() - start_time) > max_time


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
):
    if c <= 0:
        raise ValueError("c must be a positive integer.")
    if init_nodes is None:
        init_nodes = max(2, c)
    if not (2 <= init_nodes <= n_nodes):
        raise ValueError("init_nodes must satisfy 2 <= init_nodes <= n_nodes")

    rng_graph = np.random.default_rng(seed)
    rng_theta = np.random.default_rng(random.seed())

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


def make_subset_connected(g: nx.Graph, x, use_thetas: int = 0, thetas=None):
    s = set(x)
    while True:
        comps = list(nx.connected_components(g.subgraph(s)))
        if len(comps) <= 1:
            break

        c1, c2 = comps[0], comps[1]
        u, v = next(iter(c1)), next(iter(c2))
        path = nx.shortest_path(g, u, v)

        to_add = [node for node in path if node not in s]
        removable_pool = list(s - c1 - c2)

        if use_thetas and thetas is not None:
            removable_pool = sorted(removable_pool, key=lambda node: thetas[node])

        to_remove = removable_pool[: len(to_add)]
        for node in to_remove:
            s.remove(node)
        for node in to_add:
            s.add(node)

    return tuple(sorted(s))


def connected_component_update(g: nx.Graph, active_mask: np.ndarray, thetas) -> np.ndarray:
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
    thetas,
    max_t: int = 1000,
):
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
