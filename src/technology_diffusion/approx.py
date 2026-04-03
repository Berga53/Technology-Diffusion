from collections import deque
from typing import Mapping

import networkx as nx
import numpy as np

from .helpers import connected_component_spread, make_vector


def _threshold_levels(thetas: Mapping[int, int] | np.ndarray) -> list[int]:
    if hasattr(thetas, "values"):
        values = thetas.values()
    else:
        values = thetas
    return sorted({int(value) for value in values})


def _build_gamma_data(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
) -> tuple[list[int], list[list[int]], list[dict[int, int]]]:
    n_nodes = g.number_of_nodes()
    levels = _threshold_levels(thetas)

    comp_id_by_level: list[list[int]] = []
    comp_size_by_level: list[dict[int, int]] = []

    for level in levels:
        allowed_nodes = [v for v in g.nodes() if int(thetas[v]) <= level]
        subgraph = g.subgraph(allowed_nodes)

        comp_id = [-1] * n_nodes
        comp_size: dict[int, int] = {}

        component_id = 0
        for component in nx.connected_components(subgraph):
            component = list(component)
            for v in component:
                comp_id[v] = component_id
            comp_size[component_id] = len(component)
            component_id += 1

        comp_id_by_level.append(comp_id)
        comp_size_by_level.append(comp_size)

    return levels, comp_id_by_level, comp_size_by_level


def _gamma_size(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
    seed_set: set[int],
    level_idx: int,
    levels: list[int],
    comp_id_by_level: list[list[int]],
    comp_size_by_level: list[dict[int, int]],
) -> int:
    level = levels[level_idx]

    touched_components: set[int] = set()
    high_nodes: set[int] = set()

    for v in seed_set:
        if int(thetas[v]) <= level:
            component_id = comp_id_by_level[level_idx][v]
            if component_id != -1:
                touched_components.add(component_id)
        else:
            high_nodes.add(v)
            for u in g.neighbors(v):
                component_id = comp_id_by_level[level_idx][u]
                if component_id != -1:
                    touched_components.add(component_id)

    total = len(high_nodes)
    total += sum(
        comp_size_by_level[level_idx][component_id]
        for component_id in touched_components
    )

    return total


def _f_value(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
    seed_set: set[int],
    levels: list[int],
    comp_id_by_level: list[list[int]],
    comp_size_by_level: list[dict[int, int]],
) -> int:
    level_count = len(levels)

    if level_count == 1:
        return max(0, min(len(seed_set), levels[0] - 1))

    total = 0
    for level_idx in range(level_count - 1):
        target = levels[level_idx + 1] - 1
        total += min(
            _gamma_size(g, thetas, seed_set, level_idx, levels, comp_id_by_level, comp_size_by_level),
            target,
        )

    return total


def _f_target(levels: list[int]) -> int:
    level_count = len(levels)

    if level_count == 1:
        return max(0, levels[0] - 1)

    return sum(levels[level_idx + 1] - 1 for level_idx in range(level_count - 1))


def _greedy_seed_set(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
    levels: list[int],
    comp_id_by_level: list[list[int]],
    comp_size_by_level: list[dict[int, int]],
) -> set[int]:
    seed_set: set[int] = set()
    target = _f_target(levels)
    remaining_nodes = set(g.nodes())

    while _f_value(g, thetas, seed_set, levels, comp_id_by_level, comp_size_by_level) < target:
        best_node = None
        best_gain = -1

        current_value = _f_value(
            g,
            thetas,
            seed_set,
            levels,
            comp_id_by_level,
            comp_size_by_level,
        )

        for node in remaining_nodes:
            new_value = _f_value(
                g,
                thetas,
                seed_set | {node},
                levels,
                comp_id_by_level,
                comp_size_by_level,
            )
            gain = new_value - current_value

            if gain > best_gain:
                best_gain = gain
                best_node = node

        if best_node is None or best_gain <= 0:
            break

        seed_set.add(best_node)
        remaining_nodes.remove(best_node)

    return seed_set


def _shortest_path_to_set(g: nx.Graph, source: int, targets: set[int]) -> list[int]:
    if source in targets:
        return [source]

    visited = {source}
    parent = {source: None}
    queue = deque([source])

    while queue:
        u = queue.popleft()

        if u in targets:
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return path[::-1]

        for w in g.neighbors(u):
            if w not in visited:
                visited.add(w)
                parent[w] = u
                queue.append(w)

    return []


def _connect_seed_set(g: nx.Graph, seed_set: set[int]) -> set[int]:
    seeds = list(seed_set)

    if not seeds:
        return set()

    connected_seed_set = {seeds[0]}

    for node in seeds[1:]:
        path = _shortest_path_to_set(g, node, connected_seed_set)
        connected_seed_set.update(path)

    return connected_seed_set


def approx(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
) -> dict[str, object]:
    
    n_nodes = g.number_of_nodes()
    levels, comp_id_by_level, comp_size_by_level = _build_gamma_data(g, thetas)

    core_seed_set = _greedy_seed_set(g, thetas, levels, comp_id_by_level, comp_size_by_level)
    seed_set = _connect_seed_set(g, core_seed_set)

    final_x = make_vector(seed_set, n_nodes)

    spread, _, _ = connected_component_spread(
        g,
        final_x,
        thetas,
    )

    k = len(seed_set)

    if spread == n_nodes:
        return k, final_x
    else:
        return None, final_x