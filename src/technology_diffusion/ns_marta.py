import collections
import itertools
import time

import networkx as nx
import numpy as np

from .helpers import connected_component_spread


def _x_from_seedset(n_nodes: int, seedset: set[int]) -> np.ndarray:
    x = np.zeros(n_nodes, dtype=float)
    x[list(seedset)] = 1.0
    return x


def _max_comp_size_in_seedset(g: nx.Graph, seedset: set[int]) -> int:
    if len(seedset) <= 1:
        return len(seedset)
    sub = g.subgraph(seedset)
    return max((len(c) for c in nx.connected_components(sub)), default=0)


def _meets_min_connectedness(g: nx.Graph, seedset: set[int], min_conn_req: int) -> bool:
    return _max_comp_size_in_seedset(g, seedset) >= min_conn_req


def _make_move(g: nx.Graph, seedset: set[int], v: int, k: int) -> set[int] | None:
    v = int(v)
    if v in seedset:
        return None

    try:
        lengths = nx.single_source_shortest_path_length(g, v)
    except Exception:
        return None

    candidates = [(lengths[u], u) for u in seedset if u in lengths]
    if not candidates:
        return None

    _, u_best = min(candidates, key=lambda t: t[0])

    try:
        path = nx.shortest_path(g, v, u_best)
    except nx.NetworkXNoPath:
        return None

    path_nodes = set(int(u) for u in path)
    if len(path_nodes) > k:
        return None

    s_new = set(path_nodes)
    remaining = list(seedset - s_new)

    while len(s_new) < k:
        added = False
        for u in list(remaining):
            for w in g.neighbors(u):
                if int(w) in s_new:
                    s_new.add(int(u))
                    remaining.remove(u)
                    added = True
                    break
            if added:
                break
        if not added:
            break

    if len(s_new) != k:
        return None

    if not nx.is_connected(g.subgraph(s_new)):
        return None

    return s_new


def NS_marta(
    g: nx.Graph,
    x0: np.ndarray,
    theta_vec: np.ndarray,
    *,
    d: int = 2,
    max_time: int = 120,
    buffer_dim: int = 5000,
    max_outer_iters: int = 1000,
    early_stop_spread: int | None = None,
    h_radius: int = 30,
):
    n_nodes = len(x0)

    if early_stop_spread is None:
        early_stop_spread = n_nodes

    k = int(np.count_nonzero(x0))
    min_conn_req = k

    x_hist = [np.array(x0, dtype=float)]
    s0 = connected_component_spread(g, x_hist[-1], theta_vec)[0]
    s_hist = [s0]

    buffer = collections.deque(maxlen=buffer_dim)
    buffer.append(tuple(x_hist[-1]))

    calls = 1
    start = time.time()
    r = 0
    stop = False

    history = [[s_hist[-1], 0.0, calls]]

    while not stop and r < max_outer_iters:
        idx = set(np.nonzero(x_hist[-1])[0])
        improved = False

        for u_sel in list(idx):
            h = min(h_radius, k, g.number_of_nodes() - 1)

            visited = {int(u_sel)}
            frontier = {int(u_sel)}

            mg_memory = collections.deque(maxlen=5)
            tried_add_nodes = set()

            for radius in range(1, h + 1):
                next_frontier = set()
                for u in frontier:
                    for v in g.neighbors(u):
                        v = int(v)
                        if v not in visited:
                            next_frontier.add(v)

                if not next_frontier:
                    break

                visited |= next_frontier
                frontier = next_frontier

                cand = [v for v in frontier if v not in idx]
                if not cand:
                    mg_memory.append([])
                    continue

                layer_entries = []
                for v in cand:
                    mg_v = 0
                    for w in g.neighbors(v):
                        if theta_vec[int(w)] <= radius:
                            mg_v += 1
                    layer_entries.append((int(mg_v), int(v)))

                layer_entries.sort(key=lambda t: (t[0], -t[1]), reverse=True)
                mg_memory.append(layer_entries)

                best_current = None
                for mg_v, v in layer_entries:
                    if v not in tried_add_nodes:
                        best_current = (mg_v, v)
                        break

                if best_current is None:
                    continue

                best_overall = None
                for entries in mg_memory:
                    for mg_v, v in entries:
                        if v in tried_add_nodes:
                            continue
                        if (
                            best_overall is None
                            or mg_v > best_overall[0]
                            or (mg_v == best_overall[0] and v < best_overall[1])
                        ):
                            best_overall = (mg_v, v)

                attempt_list = []
                if best_overall is not None:
                    attempt_list.append(best_overall[1])
                if best_current is not None and best_current[1] not in attempt_list:
                    attempt_list.append(best_current[1])

                for best_v in attempt_list:
                    best_v = int(best_v)
                    if best_v in tried_add_nodes:
                        continue
                    tried_add_nodes.add(best_v)

                    s_new = _make_move(g, idx, best_v, k)
                    if s_new is None:
                        continue

                    x_prov = _x_from_seedset(n_nodes, s_new)
                    key = tuple(x_prov)
                    if key in buffer:
                        continue

                    if time.time() - start >= max_time:
                        stop = True
                        break

                    s_prov = connected_component_spread(g, x_prov, theta_vec)[0]
                    buffer.append(key)
                    calls += 1

                    if s_prov > s_hist[-1]:
                        x_hist.append(x_prov)
                        s_hist.append(s_prov)
                        history.append([s_prov, time.time() - start, calls])
                        improved = True

                        if s_prov >= early_stop_spread:
                            stop = True
                        break

                if stop or improved:
                    break

            if stop or improved:
                break

        if not stop and not improved:
            idx = set(np.nonzero(x_hist[-1])[0])
            idx_temp = set(range(n_nodes)) - idx
            neighbors = []

            for i in range(d // 2):
                for rem in itertools.combinations(idx, len(idx) - 1 - i):
                    for add in itertools.combinations(idx_temp, i + 1):
                        neighbors.append(set(rem) | set(add))

            s_temp = s_hist[-1]
            x_temp = x_hist[-1]

            for s_new in neighbors:
                if not _meets_min_connectedness(g, s_new, min_conn_req):
                    continue

                x_new = _x_from_seedset(n_nodes, s_new)
                key = tuple(x_new)
                if key in buffer:
                    continue

                if time.time() - start >= max_time:
                    stop = True
                    break

                s_new_val = connected_component_spread(g, x_new, theta_vec)[0]
                buffer.append(key)
                calls += 1

                if s_new_val > s_temp:
                    s_temp = s_new_val
                    x_temp = x_new
                    history.append([s_new_val, time.time() - start, calls])
                    if s_new_val >= early_stop_spread:
                        stop = True
                    break

            if not stop:
                if s_temp > s_hist[-1]:
                    x_hist.append(x_temp)
                    s_hist.append(s_temp)
                else:
                    stop = True

        r += 1

        if time.time() - start >= max_time:
            stop = True

    history.append([s_hist[-1], time.time() - start, calls])
    return s_hist, x_hist, history
