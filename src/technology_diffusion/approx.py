from collections import deque
import time
from typing import Mapping

import networkx as nx
import numpy as np

from .helpers import connected_component_spread, make_vector, time_exceeded


def _threshold_levels(thetas: Mapping[int, int] | np.ndarray) -> list[int]:
    if hasattr(thetas, "values"):
        values = thetas.values()
    else:
        values = thetas
    return sorted({int(value) for value in values})


def _build_gamma_data(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
    start: float,
    max_time: float | None,
) -> tuple[list[int], list[list[int]], list[dict[int, int]], bool]:
    n_nodes = g.number_of_nodes()
    levels = _threshold_levels(thetas)

    comp_id_by_level: list[list[int]] = []
    comp_size_by_level: list[dict[int, int]] = []
    timed_out = False

    for level in levels:
        if time_exceeded(start, max_time):
            timed_out = True
            break

        allowed_nodes: list[int] = []
        for v in g.nodes():
            if time_exceeded(start, max_time):
                timed_out = True
                break
            if int(thetas[v]) <= level:
                allowed_nodes.append(v)

        if timed_out:
            break

        subgraph = g.subgraph(allowed_nodes)

        comp_id = [-1] * n_nodes
        comp_size: dict[int, int] = {}

        component_id = 0
        for component in nx.connected_components(subgraph):
            if time_exceeded(start, max_time):
                timed_out = True
                break
            component = list(component)
            for v in component:
                comp_id[v] = component_id
            comp_size[component_id] = len(component)
            component_id += 1

        if timed_out:
            break

        comp_id_by_level.append(comp_id)
        comp_size_by_level.append(comp_size)

    return levels, comp_id_by_level, comp_size_by_level, timed_out


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
    start: float,
    max_time: float | None,
) -> tuple[set[int], bool]:
    seed_set: set[int] = set()
    target = _f_target(levels)
    remaining_nodes = set(g.nodes())
    timed_out = False

    while _f_value(g, thetas, seed_set, levels, comp_id_by_level, comp_size_by_level) < target:
        if time_exceeded(start, max_time):
            timed_out = True
            break

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
            if time_exceeded(start, max_time):
                timed_out = True
                break

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

    return seed_set, timed_out


def _shortest_path_to_set(
    g: nx.Graph,
    source: int,
    targets: set[int],
    start: float,
    max_time: float | None,
) -> tuple[list[int], bool]:
    if source in targets:
        return [source], False

    visited = {source}
    parent = {source: None}
    queue = deque([source])

    while queue:
        if time_exceeded(start, max_time):
            return [], True

        u = queue.popleft()

        if u in targets:
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return path[::-1], False

        for w in g.neighbors(u):
            if time_exceeded(start, max_time):
                return [], True
            if w not in visited:
                visited.add(w)
                parent[w] = u
                queue.append(w)

    return [], False


def _connect_seed_set(
    g: nx.Graph,
    seed_set: set[int],
    start: float,
    max_time: float | None,
) -> tuple[set[int], bool]:
    seeds = list(seed_set)

    if not seeds:
        return set(), False

    connected_seed_set = {seeds[0]}
    timed_out = False

    for node in seeds[1:]:
        if time_exceeded(start, max_time):
            timed_out = True
            break

        path, sp_timed_out = _shortest_path_to_set(g, node, connected_seed_set, start, max_time)
        if sp_timed_out:
            timed_out = True
            break
        connected_seed_set.update(path)

    return connected_seed_set, timed_out


def approx(
    g: nx.Graph,
    thetas: Mapping[int, int] | np.ndarray,
    max_time: float | None = None,
) -> tuple[int | None, np.ndarray | None, float, list[tuple[int | None, float]]]:
    start = time.perf_counter()
    n_nodes = g.number_of_nodes()

    levels, comp_id_by_level, comp_size_by_level, gamma_timed_out = _build_gamma_data(
        g,
        thetas,
        start,
        max_time,
    )
    if gamma_timed_out:
        elapsed = round(float(time.perf_counter() - start), 4)
        history = [(None, elapsed)]
        return None, None, elapsed, history

    core_seed_set, timed_out = _greedy_seed_set(
        g,
        thetas,
        levels,
        comp_id_by_level,
        comp_size_by_level,
        start,
        max_time,
    )
    seed_set, connect_timed_out = _connect_seed_set(g, core_seed_set, start, max_time)
    timed_out = timed_out or connect_timed_out

    final_x = make_vector(seed_set, n_nodes)

    spread, _, _ = connected_component_spread(
        g,
        final_x,
        thetas,
    )

    k = len(seed_set)
    elapsed = round(float(time.perf_counter() - start), 4)

    if timed_out:
        history = [(None, elapsed)]
        return None, None, elapsed, history

    if spread == n_nodes:
        history = [(int(k), elapsed)]
        return int(k), final_x, elapsed, history

    history = [(None, elapsed)]
    return None, None, elapsed, history