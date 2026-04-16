
import random
import time
from typing import Callable, Mapping

import networkx as nx
import numpy as np

from .helpers import connected_component_spread, make_subset_connected, make_vector


def SingleDiscount(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
    k: int,
) -> tuple[int, np.ndarray]:
    w1 = nx.to_numpy_array(g)
    idx = []
    target_k = min(k, len(g.nodes))
    selected = np.zeros(len(g.nodes), dtype=bool)
    ddv = np.sum(w1 > 0, axis=0)
    tv = np.zeros(len(g.nodes))

    for _ in range(target_k):
        available = np.flatnonzero(~selected)
        if len(available) == 0:
            break

        available_scores = ddv[available]
        max_discount_degree = float(np.max(available_scores))
        candidates = available[available_scores == max_discount_degree]

        u = int(random.choice(candidates.tolist()))
        idx.append(u)
        selected[u] = True
        v = list(np.nonzero(w1[u])[0])
        tv[v] += 1
        ddv[v] = ddv[v] - tv[v]
        ddv[u] = -1
        w1[u, :] = 0
        w1[:, u] = 0

    if len(idx) < target_k:
        remaining = np.flatnonzero(~selected)
        if len(remaining) > 0:
            remaining_scores = ddv[remaining]
            ordered_remaining = remaining[np.argsort(-remaining_scores)]
            shortfall = target_k - len(idx)
            idx.extend(ordered_remaining[:shortfall].tolist())

    x = np.zeros(len(g.nodes))
    x[idx] = 1.0
    s = connected_component_spread(g, x, thetas, max_t=1000)[0]
    return s, x


def degree(
    g: nx.Graph,
    n_nodes: int,
    k: int,
    thetas: Mapping[int, int] | None = None,
    connected: int = 0,
) -> np.ndarray:
    degs = np.array([g.degree(v) for v in g.nodes()])
    top_k = np.argsort(-degs)[:k]
    if connected:
        top_k = list(make_subset_connected(g, top_k, use_thetas=1, thetas=thetas))
    return make_vector(top_k, n_nodes)


def degree_threshold(
    g: nx.Graph,
    n_nodes: int,
    k: int,
    thetas: Mapping[int, int] | None = None,
    connected: int = 0,
) -> np.ndarray:
    dt = np.array([g.degree(v) for v in g.nodes()]) * np.array(list(thetas.values()))
    top_k = np.argsort(-dt)[:k]
    if connected:
        top_k = list(make_subset_connected(g, top_k, use_thetas=1, thetas=thetas))
    return make_vector(top_k, n_nodes)


def betweenness(
    g: nx.Graph,
    n_nodes: int,
    k: int,
    thetas: Mapping[int, int] | None = None,
    connected: int = 0,
) -> np.ndarray:
    bets = nx.betweenness_centrality(g)
    bet_values = np.array([bets[v] for v in g.nodes()])
    top_k = np.argsort(-bet_values)[:k]
    if connected:
        top_k = list(make_subset_connected(g, top_k, use_thetas=1, thetas=thetas))
    return make_vector(top_k, n_nodes)


def degree_connected(
    g: nx.Graph,
    n_nodes: int,
    k: int,
    thetas: Mapping[int, int] | None = None,
    connected: int | None = None,
) -> np.ndarray:
    degs = np.array([g.degree(v) for v in g.nodes()])
    target_k = min(k, len(g.nodes))
    if target_k == 0:
        return make_vector([], n_nodes)

    top_k = [int(np.argsort(-degs)[0])]
    neighbors = set(g.neighbors(top_k[0]))

    for _ in range(1, target_k):
        candidates = [v for v in g.nodes() if v not in top_k and v in neighbors]
        if not candidates:
            candidates = [v for v in g.nodes() if v not in top_k]
            if not candidates:
                break
        c_degs = np.array([g.degree(v) for v in candidates])
        top_candidate = int(candidates[np.argsort(-c_degs)[0]])
        top_k.append(top_candidate)
        neighbors.update(g.neighbors(top_candidate))

    return make_vector(top_k, n_nodes)


def high_thetas(
    g: nx.Graph,
    n_nodes: int,
    k: int,
    thetas: Mapping[int, int] | None = None,
    connected: int = 1,
) -> np.ndarray:
    sorted_nodes = sorted(g.nodes(), key=lambda x: thetas[x], reverse=True)
    top_k = sorted_nodes[:k]
    if connected:
        top_k = list(make_subset_connected(g, top_k, use_thetas=1, thetas=thetas))
    return make_vector(top_k, n_nodes)


def degree_discount(
    g: nx.Graph,
    n_nodes: int,
    k: int,
    thetas: Mapping[int, int] | None = None,
    connected: int = 1,
) -> np.ndarray:
    _, x0 = SingleDiscount(g, thetas, k)
    if connected:
        x0 = make_vector(
            make_subset_connected(g, list(np.nonzero(x0)[0]), use_thetas=1, thetas=thetas),
            n_nodes,
        )
    return x0


def random_start(
    g: nx.Graph,
    n_nodes: int,
    k: int,
    thetas: Mapping[int, int] | None = None,
    connected: int = 1,
) -> np.ndarray:
    nodes = list(g.nodes())
    random.shuffle(nodes)
    top_k = nodes[:k]
    if connected:
        top_k = list(make_subset_connected(g, top_k, use_thetas=1, thetas=thetas))
    return make_vector(top_k, n_nodes)


def technology_diffusion_heuristics(
    g: nx.Graph,
    n_nodes: int,
    thetas: Mapping[int, int] | None = None,
    connected: int = 1,
    heuristic: Callable[[nx.Graph, int, int, Mapping[int, int] | None, int], np.ndarray] | None = None,
) -> tuple[np.ndarray, int, list[tuple[int, float]]]:
    if heuristic is None:
        heuristic = degree

    if n_nodes <= 0:
        return np.zeros(n_nodes), 0, [(0, 0.0)]

    start = time.perf_counter()
    history: list[tuple[int, float]] = [(n_nodes, 0.0)]

    def evaluate(k: int) -> tuple[np.ndarray, int, np.ndarray]:
        x_eval = heuristic(g, n_nodes, k, thetas, connected)
        s_eval, _, active_after = connected_component_spread(g, x_eval, thetas, max_t=1000)
        return x_eval, int(s_eval), np.array(active_after, dtype=float)

    tried_k = set()
    best_k: int | None = None
    best_solution_x: np.ndarray | None = None
    top_k, bottom_k = n_nodes, 1

    while bottom_k <= top_k:
        k = (top_k + bottom_k) // 2
        if k in tried_k:
            break
        tried_k.add(k)

        x_k, spread_k, active_after = evaluate(k)

        if spread_k == n_nodes:
            if best_k is None or k < best_k:
                best_k = k
                best_solution_x = np.array(x_k, dtype=float, copy=True)
                history.append((best_k, round(time.perf_counter() - start, 4)))
            top_k = k - 1
        else:
            inferred_success_k = k + (n_nodes - spread_k)
            if best_k is None or inferred_success_k < best_k:
                inferred_x = np.array(x_k, dtype=float, copy=True)
                inferred_x[np.where(active_after == 0)[0]] = 1.0
                best_k = inferred_success_k
                best_solution_x = inferred_x
                history.append((best_k, round(time.perf_counter() - start, 4)))

            if inferred_success_k <= top_k:
                top_k = inferred_success_k - 1
            bottom_k = k + 1

    history.append((int(best_k), round(time.perf_counter() - start, 4)))
    return best_solution_x, int(best_k), round(float(time.perf_counter() - start), 4), history