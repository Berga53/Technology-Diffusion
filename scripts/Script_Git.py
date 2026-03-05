import os
import time
import math
import json
import pickle
import itertools
import collections
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
import networkx as nx

import gurobipy as gp
from gurobipy import GRB


# ============================================================
#Graph Generation
# ============================================================
def preferential_attachment_variable_m(
    N: int,
    seed: int | None = None,
    init_nodes: int | None = None,
    init_mode: str = "complete",
    M=(1, 2, 3, 4),
):
    rng = np.random.default_rng(seed)
    M = tuple(M)

    if N < 2:
        raise ValueError("N deve essere >= 2.")

    if init_nodes is None:
        init_nodes = min(5, N)
    init_nodes = int(max(2, min(init_nodes, N)))

    if init_mode == "complete":
        G = nx.complete_graph(init_nodes)
    else:
        raise ValueError("init_mode deve essere 'complete'")

    for u in range(init_nodes, N):
        G.add_node(u)

        m = int(rng.choice(M))
        existing = np.array(list(G.nodes()))
        existing = existing[existing != u]
        m = min(m, len(existing))

        degrees = np.array([G.degree(v) for v in existing], dtype=float)
        s = degrees.sum()
        probs = degrees / s if s > 0 else np.ones_like(degrees) / len(degrees)

        targets = rng.choice(existing, size=m, replace=False, p=probs)
        for v in targets:
            G.add_edge(u, int(v))

    return G


# ============================================================
#G&L IP
# ============================================================
def build_golberg_liu_ip(G: nx.Graph, theta: dict):
    V = list(G.nodes())
    n = len(V)
    T = range(1, n + 1)

    m = gp.Model("golberg_liu_ip")
    x = m.addVars(V, T, vtype=GRB.BINARY, name="x")

    m.setObjective(
        gp.quicksum(x[u, t] for u in V for t in T if t <= int(theta[u])),
        GRB.MINIMIZE,
    )

    for u in V:
        m.addConstr(gp.quicksum(x[u, t] for t in T) == 1, name=f"perm_node[{u}]")
    for t in T:
        m.addConstr(gp.quicksum(x[u, t] for u in V) == 1, name=f"perm_time[{t}]")

    for u in V:
        neighbors = list(G.neighbors(u))
        for t in range(2, n + 1):
            if neighbors:
                m.addConstr(
                    gp.quicksum(x[v, tp] for v in neighbors for tp in range(1, t))
                    >= x[u, t],
                    name=f"conn[{u},{t}]",
                )
            else:
                m.addConstr(x[u, t] == 0, name=f"isolated[{u},{t}]")

    m.update()
    return m, x


# ============================================================
#Diffusion
# ============================================================
def connected_component_threshold_update_theta(g, active_mask, theta_vec):
    N = g.number_of_nodes()
    newly_active = np.zeros(N, dtype=bool)

    if not np.any(active_mask):
        return newly_active

    active_nodes = np.flatnonzero(active_mask)
    active_set = set(active_nodes.tolist())

    comp_id_of = {}
    comp_sizes = []
    cid = 0
    visited = set()

    for u in active_nodes:
        u = int(u)
        if u in visited:
            continue
        stack = [u]
        visited.add(u)
        comp_nodes = [u]
        while stack:
            x = stack.pop()
            for nbr in g.neighbors(x):
                nbr = int(nbr)
                if nbr in active_set and nbr not in visited:
                    visited.add(nbr)
                    stack.append(nbr)
                    comp_nodes.append(nbr)
        size = len(comp_nodes)
        comp_sizes.append(size)
        for w in comp_nodes:
            comp_id_of[w] = cid
        cid += 1

    comp_sizes = np.asarray(comp_sizes, dtype=int)

    inactive_nodes = np.flatnonzero(~active_mask)
    for v in inactive_nodes:
        v = int(v)
        touched_cids = set()
        for u in g.neighbors(v):
            u = int(u)
            if active_mask[u]:
                c = comp_id_of.get(u, None)
                if c is not None:
                    touched_cids.add(c)
        if touched_cids:
            total_reachable = int(comp_sizes[list(touched_cids)].sum())
            if total_reachable + 1 >= int(theta_vec[v]):
                newly_active[v] = True

    return newly_active


def Influence_evaluation_comp_threshold_theta(
    g, W, x0, params=None, max_t=999, *, theta_vec, h0=1.0
):
    N = g.number_of_nodes()
    active = (x0 > 0).astype(bool)
    spread_hist = [int(np.sum(active))]

    for _ in range(max_t):
        prev = active.copy()
        newly = connected_component_threshold_update_theta(g, prev, theta_vec)
        active = prev | newly
        if np.array_equal(active, prev):
            break
        spread_hist.append(int(np.sum(active)))

    final_spread = int(np.sum(active))
    final_x = np.zeros(N, dtype=float)
    final_x[active] = h0
    return final_spread, spread_hist, [final_x]


def assign_thresholds_paper_experiments(G: nx.Graph, c: int, seed: int | None = None, attr: str = "theta"):
    if c <= 0:
        raise ValueError("c must be a positive integer.")
    rng = np.random.default_rng(seed)
    n = G.number_of_nodes()

    k_max = math.ceil(n / c)
    allowed_values = list(range(max(2, c), k_max * c + 1, c))

    theta = {int(v): int(rng.choice(allowed_values)) for v in G.nodes()}
    nx.set_node_attributes(G, theta, name=attr)
    return theta, allowed_values


# ============================================================
#NS modificato

# ============================================================
def NS_marginal_gain_minconn_progressive(
    g, W, x0, params, delta, xi, d, max_time, buffer_dim,
    thresholds,
    influence_eval_fn=None, eval_kwargs=None,
    max_outer_iters=1000,
    min_conn_req=None,
    early_stop_spread=None,
):
    if influence_eval_fn is None:
        influence_eval_fn = Influence_evaluation_comp_threshold_theta
    eval_kwargs = dict(eval_kwargs or {})

    N = len(x0)
    K = int(np.count_nonzero(x0))
    X = [np.array(x0, dtype=float)]
    l0, h0, theta_l, theta_h, gamma, eps = (params + [None]*6)[:6]

    if early_stop_spread is None:
        early_stop_spread = N


    if "theta_vec" in eval_kwargs and eval_kwargs["theta_vec"] is not None:
        if min_conn_req is None:
            min_conn_req = K
    else:
        thresholds_arr = np.asarray(thresholds)
        if min_conn_req is None:
            min_conn_req = int(thresholds_arr.min())

    if min_conn_req > K:
        raise ValueError(f"min_conn_req={min_conn_req} > K={K}. Requirement impossible.")

    def x_from_seedset(S):
        x = np.zeros(N)
        x[list(S)] = h0
        return x

    def max_comp_size_in_seedset(S):
        if len(S) == 0:
            return 0
        if len(S) == 1:
            return 1
        sub = g.subgraph(S)
        return max((len(c) for c in nx.connected_components(sub)), default=0)

    def meets_min_connectedness(S):
        return max_comp_size_in_seedset(S) >= min_conn_req


    s0 = influence_eval_fn(g, W, X[-1], params, **eval_kwargs)[0]
    s = [s0]
    xi_t = xi
    r = 0

    buffer = collections.deque(maxlen=buffer_dim)
    calls = 1
    start = time.time()
    stop = False
    history = [[s[-1], time.time() - start, calls]]

    print(
        "\rNS_marginal_minconn_progressive... Calls:{} Time:{}/{} Spread:{} (min_conn_req={})".format(
            calls, round(time.time() - start), max_time, s[-1], min_conn_req
        ),
        end="",
        flush=True,
    )


    if early_stop_spread is not None and s[-1] >= early_stop_spread:
        history.append([s[-1], time.time() - start, calls])
        print(
            "\rNS_marginal_minconn_progressive DONE. Calls:{} Time:{}/{} Spread:{} (min_conn_req={})".format(
                calls, round(time.time() - start), max_time, s[-1], min_conn_req
            ),
            flush=True,
        )
        return s, X, history

    while not stop and r < max_outer_iters:
        idx = set(np.nonzero(X[-1])[0])
        improved = False


        for u_sel in list(idx):

            seed_sub = g.subgraph(idx)
            if u_sel not in seed_sub:
                continue

            comp_u = nx.node_connected_component(seed_sub, u_sel)
            h = len(comp_u)
            if h <= 1:
                continue

            dist_map = nx.single_source_shortest_path_length(g, u_sel, cutoff=h - 1)

            by_dist = {}
            for v, dist in dist_map.items():
                v = int(v)
                if dist == 0:
                    continue
                by_dist.setdefault(dist, []).append(v)

            for radius in range(1, h):
                cand = by_dist.get(radius, [])
                if not cand:
                    continue

                cand = [v for v in cand if (v not in idx) and (thresholds[v] > min_conn_req)]
                if not cand:
                    continue

                best_v = None
                best_mg = -1
                for v in cand:
                    mg_v = 0
                    for w in g.neighbors(v):
                        w = int(w)
                        if thresholds[w] <= radius:
                            mg_v += 1
                    if mg_v > best_mg:
                        best_mg = mg_v
                        best_v = v

                if best_v is None:
                    continue

                S_prov = (idx - {u_sel}) | {best_v}
                if not meets_min_connectedness(S_prov):
                    continue

                x_prov = x_from_seedset(S_prov)

                if list(x_prov) in buffer:
                    continue
                if time.time() - start >= max_time:
                    stop = True
                    break

                s_prov = influence_eval_fn(g, W, x_prov, params, **eval_kwargs)[0]
                buffer.append(list(x_prov))
                calls += 1

                if s_prov > s[-1]:
                    history.append([s_prov, time.time() - start, calls])
                    X.append(x_prov)
                    s.append(s_prov)
                    improved = True

                    if early_stop_spread is not None and s[-1] >= early_stop_spread:
                        stop = True
                    break

            if stop:
                break
            if improved:
                break


        if not stop and not improved:
            idx = set(np.nonzero(X[-1])[0])
            idx_temp = set(range(N)) - idx
            neighbors = []

            for i in range(d // 2):
                for rem in itertools.combinations(idx, len(idx) - 1 - i):
                    for add in itertools.combinations(idx_temp, i + 1):
                        neighbors.append(set(rem) | set(add))

            s_temp = s[-1]
            x_temp = X[-1]

            for S_new in neighbors:
                if not meets_min_connectedness(S_new):
                    continue

                x_new = x_from_seedset(S_new)
                if list(x_new) in buffer:
                    continue
                if time.time() - start >= max_time:
                    stop = True
                    break

                s_new = influence_eval_fn(g, W, x_new, params, **eval_kwargs)[0]
                buffer.append(list(x_new))
                calls += 1

                if s_new > s_temp:
                    history.append([s_new, time.time() - start, calls])
                    s_temp = s_new
                    x_temp = x_new

                    if early_stop_spread is not None and s_temp >= early_stop_spread:
                        stop = True
                    break

            if not stop:
                X.append(x_temp)
                s.append(s_temp)

                if early_stop_spread is not None and s[-1] >= early_stop_spread:
                    stop = True
                else:
                    if len(s) >= 2 and s[-1] == s[-2]:
                        X.pop()
                        s.pop()
                        stop = True
                    elif len(s) >= 2 and s[-1] > s[-2]:
                        xi_t *= delta

        r += 1

        if time.time() - start >= max_time:
            stop = True
            break

        print(
            "\rNS_marginal_minconn_progressive... Calls:{} Time:{}/{} Spread:{} (min_conn_req={})".format(
                calls, round(time.time() - start), max_time, s[-1], min_conn_req
            ),
            end="",
            flush=True,
        )

    history.append([s[-1], time.time() - start, calls])
    print(
        "\rNS_marginal_minconn_progressive DONE. Calls:{} Time:{}/{} Spread:{} (min_conn_req={})".format(
            calls, round(time.time() - start), max_time, s[-1], min_conn_req
        ),
        flush=True,
    )
    return s, X, history


# ============================================================
#Generazione seedset iniziale
# ============================================================
def seeds_to_x0(N, seeds, h0=1.0):
    x0 = np.zeros(N, dtype=float)
    x0[list(seeds)] = h0
    return x0


def vec_to_seed_indices(x_vec):
    return tuple(sorted(np.where(np.asarray(x_vec) > 0)[0].tolist()))


def _grow_connected_from_start(g: nx.Graph, start: int, k: int, rng: np.random.Generator):
    if k <= 1:
        return (int(start),)
    seedset = {int(start)}
    frontier = deque([int(start)])
    while frontier and len(seedset) < k:
        u = frontier.popleft()
        neigh = list(g.neighbors(u))
        rng.shuffle(neigh)
        for v in neigh:
            v = int(v)
            if v not in seedset:
                seedset.add(v)
                frontier.append(v)
                if len(seedset) >= k:
                    break
    if len(seedset) < k:
        return None
    return tuple(sorted(seedset))


def sample_connected_seedset_random(g: nx.Graph, k: int, rng: np.random.Generator, max_tries: int = 2000):
    nodes = list(g.nodes())
    if k <= 0:
        return tuple()
    if k == 1:
        return (int(rng.choice(nodes)),)
    if k > g.number_of_nodes():
        raise ValueError("k cannot exceed number of nodes")

    for _ in range(max_tries):
        start = int(rng.choice(nodes))
        S = _grow_connected_from_start(g, start, k, rng)
        if S is not None:
            return S
    raise RuntimeError(f"Could not sample a connected seedset of size {k} in {max_tries} tries.")


def seedset_high_degree_connected(g: nx.Graph, k: int, rng: np.random.Generator, top_m: int = 30, max_tries: int = 300):
    nodes = list(g.nodes())
    if k <= 0:
        return tuple()
    if k == 1:
        u = max(nodes, key=lambda x: (g.degree(int(x)), -int(x)))
        return (int(u),)

    ranked = sorted(nodes, key=lambda x: (-g.degree(int(x)), int(x)))
    top = ranked[: min(top_m, len(ranked))]

    for _ in range(max_tries):
        start = int(rng.choice(top))
        S = _grow_connected_from_start(g, start, k, rng)
        if S is not None:
            return S
    return sample_connected_seedset_random(g, k, rng)


def seedset_kcore_connected(g: nx.Graph, k: int, rng: np.random.Generator, max_tries: int = 300):
    nodes = list(g.nodes())
    if k <= 0:
        return tuple()
    if k == 1:
        return (int(rng.choice(nodes)),)

    core_num = nx.core_number(g)
    max_core = max(core_num.values()) if core_num else 0
    core_nodes = [int(u) for u, c in core_num.items() if c == max_core]
    if not core_nodes:
        return sample_connected_seedset_random(g, k, rng)

    for _ in range(max_tries):
        start = int(rng.choice(core_nodes))
        S = _grow_connected_from_start(g, start, k, rng)
        if S is not None:
            return S
    return seedset_high_degree_connected(g, k, rng)


def seedset_betweenness_connected(g: nx.Graph, k: int, rng: np.random.Generator, top_m: int = 30, max_tries: int = 300):
    nodes = list(g.nodes())
    if k <= 0:
        return tuple()
    if k == 1:
        bc = nx.betweenness_centrality(g, normalized=True)
        u = max(nodes, key=lambda x: (bc[int(x)], g.degree(int(x)), -int(x)))
        return (int(u),)

    bc = nx.betweenness_centrality(g, normalized=True)
    ranked = sorted(nodes, key=lambda x: (-bc[int(x)], -g.degree(int(x)), int(x)))
    top = [int(u) for u in ranked[: min(top_m, len(ranked))]]

    for _ in range(max_tries):
        start = int(rng.choice(top))
        S = _grow_connected_from_start(g, start, k, rng)
        if S is not None:
            return S
    return seedset_high_degree_connected(g, k, rng)


# ============================================================
#Seedset iniziale alternativo 
# ============================================================
def build_max_theta_heavy_connected_seedset(g: nx.Graph, theta_vec: np.ndarray, K: int, rng: np.random.Generator):
    n = g.number_of_nodes()
    if K <= 0:
        theta_max = int(theta_vec.max())
        max_nodes = [int(v) for v in range(n) if int(theta_vec[int(v)]) == theta_max]
        return tuple(), max_nodes, theta_max

    theta_max = int(theta_vec.max())
    max_nodes = [int(v) for v in range(n) if int(theta_vec[int(v)]) == theta_max]


    if not max_nodes:
        S = sample_connected_seedset_random(g, K, rng)
        return S, [], theta_max

    start = max(max_nodes, key=lambda u: (g.degree(int(u)), -int(u)))
    S = {int(start)}

    remaining = [u for u in max_nodes if u != start]
    remaining.sort(key=lambda u: (g.degree(int(u)), -int(u)), reverse=True)


    for u in remaining:
        if len(S) >= K:
            break
        best_path = None
        best_len = None
        for s in list(S):
            try:
                p = nx.shortest_path(g, source=int(s), target=int(u))
            except nx.NetworkXNoPath:
                continue
            if best_len is None or len(p) < best_len:
                best_len = len(p)
                best_path = p
        if best_path is None:
            continue
        new_nodes = [int(x) for x in best_path if int(x) not in S]
        if len(S) + len(new_nodes) <= K:
            for x in new_nodes:
                S.add(int(x))


    q = deque(list(S))
    while q and len(S) < K:
        u = q.popleft()
        neigh = list(g.neighbors(u))
        rng.shuffle(neigh)
        for v in neigh:
            v = int(v)
            if v not in S:
                S.add(v)
                q.append(v)
            if len(S) >= K:
                break


    nodes = list(g.nodes())
    while len(S) < K:
        S.add(int(rng.choice(nodes)))


    if not nx.is_connected(g.subgraph(list(S))):
        bfs_nodes = list(nx.bfs_tree(g, int(start)).nodes())
        S = set(int(x) for x in bfs_nodes[:K])

    return tuple(sorted(S)), max_nodes, theta_max


def generate_multistart_seedsets(
    g: nx.Graph,
    k: int,
    rng: np.random.Generator,
    *,
    per_family: int = 2,
    include_betweenness: bool = True,
    special_first: tuple | None = None,
):
    seedsets = []
    seen = set()

    def add(S):
        if S is None:
            return
        S = tuple(sorted(int(x) for x in S))
        if len(S) != k:
            return
        if not nx.is_connected(g.subgraph(S)):
            return
        if S in seen:
            return
        seen.add(S)
        seedsets.append(S)


    if special_first is not None:
        add(special_first)

    for _ in range(per_family):
        add(seedset_high_degree_connected(g, k, rng))
    for _ in range(per_family):
        add(seedset_kcore_connected(g, k, rng))
    if include_betweenness:
        for _ in range(per_family):
            add(seedset_betweenness_connected(g, k, rng))
    for _ in range(per_family):
        add(sample_connected_seedset_random(g, k, rng))

    return seedsets


# ============================================================
#Euristiche e run dell'esperimento 
# ============================================================
def run_solver_best_of_multistart(
    solver_fn,
    g, W, params, eval_fn, eval_kwargs,
    seedsets,
    DELTA, XI, D, MAX_TIME, BUFFER_DIM,
    *,
    h0=1.0,
    solver_kwargs=None,
    early_stop_spread=None,
    target_spread=None,
    close_margin=2,
    extra_random_starts=0,
    rng_for_extra=None,
):
    solver_kwargs = dict(solver_kwargs or {})
    N = g.number_of_nodes()
    if early_stop_spread is None:
        early_stop_spread = N

    best_spread = -1
    best_seeds = None
    best_history = None
    best_calls_last = None
    runs_used = 0

    def run_one(init_seeds):
        nonlocal best_spread, best_seeds, best_history, best_calls_last, runs_used
        runs_used += 1
        x0 = seeds_to_x0(N, init_seeds, h0=h0)

        s, X, history = solver_fn(
            g, W, x0, params, DELTA, XI, D, MAX_TIME, BUFFER_DIM,
            influence_eval_fn=eval_fn,
            eval_kwargs=eval_kwargs,
            early_stop_spread=early_stop_spread,
            **solver_kwargs
        )

        best_idx = int(np.argmax(s))
        spread = int(s[best_idx])
        seeds = vec_to_seed_indices(X[best_idx])

        hist_arr = np.asarray(history, dtype=float)
        calls_last = int(hist_arr[:, 2].max()) if hist_arr.size else 0

        if spread > best_spread:
            best_spread = spread
            best_seeds = seeds
            best_history = history
            best_calls_last = calls_last

    for S in seedsets:
        run_one(S)
        if best_spread >= early_stop_spread:
            break

    if (
        (best_spread < early_stop_spread)
        and (target_spread is not None)
        and (best_spread >= target_spread - close_margin)
        and (extra_random_starts > 0)
        and (rng_for_extra is not None)
    ):
        k = len(seedsets[0]) if seedsets else None
        if k is not None:
            for _ in range(extra_random_starts):
                S = sample_connected_seedset_random(g, k, rng_for_extra)
                run_one(S)
                if best_spread >= early_stop_spread:
                    break

    return {
        "k": len(best_seeds) if best_seeds is not None else None,
        "best_spread": best_spread,
        "best_seeds": best_seeds,
        "calls_to_last": best_calls_last,
        "best_history": best_history,
        "runs_used": runs_used,
    }


# ============================================================
#Helpers
# ============================================================

def _complete_seedset_by_adding_missing_max_theta(
    g: nx.Graph,
    theta_vec: np.ndarray,
    base_seeds: list[int],
    max_theta_nodes: list[int],
    target_spread: int | None,
    eval_fn,
    W,
    params,
    eval_kwargs,
    *,
    h0: float = 1.0,
):
    base_seeds = [int(x) for x in base_seeds]
    S = set(base_seeds)

    max_theta_nodes = [int(x) for x in max_theta_nodes]
    missing_max = [int(v) for v in max_theta_nodes if int(v) not in S]


    added_max = []
    for v in missing_max:
        if v not in S:
            S.add(v)
            added_max.append(int(v))


    if target_spread is None:
        sp = spread_from_seeds(g, eval_fn, W, params, eval_kwargs, list(S), h0=h0)
        return {
            "completed_seeds": tuple(sorted(S)),
            "completed_k": int(len(S)),
            "completed_spread": int(sp),
            "added_max_theta_nodes": added_max,
            "added_extra_nodes": [],
        }


    comp = _complete_seedset_to_target(
        g=g,
        theta_vec=theta_vec,
        base_seeds=list(S),
        target_spread=int(target_spread),
        eval_fn=eval_fn,
        W=W,
        params=params,
        eval_kwargs=eval_kwargs,
        h0=h0,
    )

    return {
        "completed_seeds": comp["completed_seeds"],
        "completed_k": int(len(comp["completed_seeds"])),
        "completed_spread": int(comp["completed_spread"]),
        "added_max_theta_nodes": added_max,
        "added_extra_nodes": list(comp["added_nodes"]),
    }


def _complete_seedset_to_target(
    g: nx.Graph,
    theta_vec: np.ndarray,
    base_seeds: list[int],
    target_spread: int,
    eval_fn,
    W,
    params,
    eval_kwargs,
    *,
    h0: float = 1.0,
):
    base_seeds = [int(x) for x in base_seeds]
    S = set(base_seeds)

    def spread_of(seed_list):
        return spread_from_seeds(g, eval_fn, W, params, eval_kwargs, seed_list, h0=h0)

    spread0 = spread_of(list(S))
    if spread0 >= target_spread:
        return {
            "completed_seeds": tuple(sorted(S)),
            "completed_spread": int(spread0),
            "added_nodes": [],
        }


    x0 = np.zeros(g.number_of_nodes(), dtype=float)
    x0[list(S)] = h0
    sp, _, final_list = eval_fn(g, W, x0, params, **eval_kwargs)
    final_x = final_list[0]
    active_mask = (final_x > 0)
    inactive = [int(i) for i in np.flatnonzero(~active_mask) if int(i) not in S]

    inactive.sort(key=lambda v: (int(theta_vec[v]), g.degree(int(v)), -int(v)), reverse=True)

    added = []
    for v in inactive:
        if spread0 >= target_spread:
            break
        S.add(int(v))
        added.append(int(v))
        spread0 = spread_of(list(S))

    return {
        "completed_seeds": tuple(sorted(S)),
        "completed_spread": int(spread0),
        "added_nodes": added,
    }


# ============================================================

#  INTERNAL "DIVIDE THE SEEDSET" BISECTION (compression)
#  If a seedset reaches target_spread, we try to find a smaller CONNECTED subset
#  by bisection over a connected BFS-prefix of the seedset.

# ============================================================
def bfs_order_in_seedset(g: nx.Graph, S):
    S = list(map(int, S))
    if not S:
        return []
    sub = g.subgraph(S)


    root = max(S, key=lambda u: (sub.degree(int(u)), -int(u)))

    order = []
    seen = set([root])
    q = deque([root])

    while q:
        u = q.popleft()
        order.append(int(u))
        neigh = [int(v) for v in sub.neighbors(u) if int(v) not in seen]
        neigh.sort()
        for v in neigh:
            seen.add(v)
            q.append(v)


    if len(order) < len(S):
        rest = sorted([u for u in S if u not in seen])
        for start in rest:
            if start in seen:
                continue
            seen.add(start)
            q = deque([start])
            while q:
                u = q.popleft()
                order.append(int(u))
                neigh = [int(v) for v in sub.neighbors(u) if int(v) not in seen]
                neigh.sort()
                for v in neigh:
                    seen.add(v)
                    q.append(v)

    return order


def compress_seedset_by_internal_bisection(
    g: nx.Graph,
    seeds,
    *,
    min_conn_req: int,
    target_spread: int,
    eval_fn,
    W,
    params,
    eval_kwargs,
    h0: float = 1.0,
):
    seeds = list(map(int, seeds))
    k = len(seeds)
    if k == 0:
        return [], 0, 0

    order = bfs_order_in_seedset(g, seeds)




    COMPRESS_MIN_K = 1
    lo = max(1, int(COMPRESS_MIN_K))
    hi = k
    if lo > k:
        sp = spread_from_seeds(g, eval_fn, W, params, eval_kwargs, order, h0=h0)
        return order, k, sp

    sp_full = spread_from_seeds(g, eval_fn, W, params, eval_kwargs, order, h0=h0)
    if sp_full < target_spread:
        return order, k, sp_full

    best_m = hi
    best_seeds = order
    best_sp = sp_full

    while lo <= hi:
        mid = (lo + hi) // 2
        cand = order[:mid]
        sp = spread_from_seeds(g, eval_fn, W, params, eval_kwargs, cand, h0=h0)
        if sp >= target_spread:
            best_m = mid
            best_seeds = cand
            best_sp = sp
            hi = mid - 1
        else:
            lo = mid + 1

    return list(best_seeds), int(best_m), int(best_sp)

# ============================================================

#  BISECTION MULTISTART (alpha dipende da c, target ad-hoc, Xi=1e-4, LP time=1000)
#  + NEW: special starts (theta-max heavy)
#  + NEW: postprocess pick min(k + missing_to_target) over explored k

# ============================================================
def bisection_multistart(
    solver_name,
    solver_fn,
    g, W, params, eval_fn, eval_kwargs,
    target_spread,
    k_low, k_high,
    rng_seed,
    DELTA, XI, D, MAX_TIME, BUFFER_DIM,
    *,
    h0=1.0,
    solver_kwargs_base=None,
    early_stop_spread=None,
    per_family=2,
    include_betweenness=True,
    close_margin=2,
    extra_random_starts=4,
    min_conn_alpha=None,

):
    solver_kwargs_base = dict(solver_kwargs_base or {})

    def rng_for_k(k, salt):
        return np.random.default_rng(rng_seed + 1000 * k + salt)

    tested = []
    best_solution_at_k = {}

    theta_vec = eval_kwargs.get("theta_vec", None)
    if theta_vec is None:
        raise ValueError("eval_kwargs must contain 'theta_vec' for the new bisection logic.")
    theta_vec = np.asarray(theta_vec, dtype=int)

    def solver_kwargs_for_k(k):

        kw = dict(solver_kwargs_base)
        kw["min_conn_req"] = None
        return kw

    def special_start_for_k(k):
        rr = rng_for_k(k, 555)
        try:
            S, _, _ = build_max_theta_heavy_connected_seedset(g, theta_vec, k, rr)
            return S
        except Exception:
            return None

    def _maybe_compress_result(res: dict, k_val: int):
        if res is None:
            return res
        if res.get("best_seeds") is None:
            res["compressed_seeds"] = None
            res["compressed_k"] = None
            res["compressed_spread"] = None
            return res
        if int(res.get("best_spread", -1)) < int(target_spread):
            res["compressed_seeds"] = None
            res["compressed_k"] = None
            res["compressed_spread"] = None
            return res

        try:
            min_conn_req_k = 1
            comp_seeds, comp_k, comp_sp = compress_seedset_by_internal_bisection(
                g, res["best_seeds"],
                min_conn_req=min_conn_req_k,
                target_spread=int(target_spread),
                eval_fn=eval_fn, W=W, params=params, eval_kwargs=eval_kwargs,
                h0=h0,
            )
            res["compressed_seeds"] = tuple(comp_seeds)
            res["compressed_k"] = int(comp_k)
            res["compressed_spread"] = int(comp_sp)
            if comp_k is not None and res.get("best_seeds") is not None:
                try:
                    k_raw = len(res["best_seeds"])
                    if int(comp_k) < int(k_raw):
                        print(f"[k={k_val}] internal-bisect: {k_raw} -> {int(comp_k)} (spread={int(comp_sp)})", flush=True)
                except Exception:
                    pass
        except Exception:

            res["compressed_seeds"] = None
            res["compressed_k"] = None
            res["compressed_spread"] = None

        return res


    k = k_high
    seedsets = generate_multistart_seedsets(
        g, k, rng_for_k(k, 1),
        per_family=per_family,
        include_betweenness=include_betweenness,
        special_first=special_start_for_k(k),
    )
    res_high = run_solver_best_of_multistart(
        solver_fn, g, W, params, eval_fn, eval_kwargs,
        seedsets,
        DELTA, XI, D, MAX_TIME, BUFFER_DIM,
        h0=h0,
        solver_kwargs=solver_kwargs_for_k(k),
        early_stop_spread=early_stop_spread,
        target_spread=target_spread,
        close_margin=close_margin,
        extra_random_starts=extra_random_starts,
        rng_for_extra=rng_for_k(k, 99),
    )
    res_high = _maybe_compress_result(res_high, k_high)
    res_high["k"] = k_high
    tested.append({**res_high, "solver": solver_name})
    best_solution_at_k[k_high] = res_high

    if res_high["best_spread"] < target_spread:
        df = pd.DataFrame(tested)
        return None, df, None

    lo, hi = k_low, k_high
    while lo < hi:
        mid = (lo + hi) // 2

        if mid not in best_solution_at_k:
            seedsets = generate_multistart_seedsets(
                g, mid, rng_for_k(mid, 1),
                per_family=per_family,
                include_betweenness=include_betweenness,
                special_first=special_start_for_k(mid),
            )
            res_mid = run_solver_best_of_multistart(
                solver_fn, g, W, params, eval_fn, eval_kwargs,
                seedsets,
                DELTA, XI, D, MAX_TIME, BUFFER_DIM,
                h0=h0,
                solver_kwargs=solver_kwargs_for_k(mid),
                early_stop_spread=early_stop_spread,
                target_spread=target_spread,
                close_margin=close_margin,
                extra_random_starts=extra_random_starts,
                rng_for_extra=rng_for_k(mid, 99),
            )
            res_mid = _maybe_compress_result(res_mid, mid)
            res_mid["k"] = mid
            tested.append({**res_mid, "solver": solver_name})
            best_solution_at_k[mid] = res_mid
        else:
            res_mid = best_solution_at_k[mid]
            if res_mid.get('compressed_k', None) is None and int(res_mid.get('best_spread',-1)) >= int(target_spread):
                res_mid = _maybe_compress_result(res_mid, mid)
                best_solution_at_k[mid] = res_mid






        near_ratio = 0.90
        soft_target = int(math.ceil(near_ratio * int(target_spread)))

        sp_mid = int(res_mid.get("best_spread", -1))

        if sp_mid >= int(target_spread):
            hi = mid
        elif sp_mid >= soft_target:

            print(f"[BISECTION] soft-feasible at k={mid}: spread={sp_mid} >= {soft_target} (90% target). Continue left.", flush=True)
            hi = mid
        else:
            lo = mid + 1

    best_k_bisect = lo







    best_post = None
    for k, res in best_solution_at_k.items():
        if res.get("best_seeds") is None:
            continue


        if res.get("compressed_seeds") is not None and int(res.get("compressed_spread", -1)) >= int(target_spread):
            base_seeds = tuple(res["compressed_seeds"])
            base_k = int(res.get("compressed_k", len(base_seeds)))
            base_spread = int(res.get("compressed_spread", -1))
            used_compressed = True
        else:
            base_seeds = tuple(res["best_seeds"])
            base_k = int(k)
            base_spread = int(res.get("best_spread", -1))
            used_compressed = False

        missing_to_target = max(0, int(target_spread) - int(base_spread))
        k_eff = int(base_k) + int(missing_to_target)

        cand = {
            "k_tested": int(k),
            "base_k": int(base_k),
            "base_spread": int(base_spread),
            "base_seeds": tuple(base_seeds),
            "used_compressed": bool(used_compressed),
            "missing_to_target": int(missing_to_target),
            "k_eff": int(k_eff),
            "runs_used": int(res.get("runs_used", 0) or 0),
            "calls_to_last": int(res.get("calls_to_last", 0) or 0),
        }

        if best_post is None:
            best_post = cand
        else:
            if cand["k_eff"] < best_post["k_eff"]:
                best_post = cand
            elif cand["k_eff"] == best_post["k_eff"]:

                if cand["base_k"] < best_post["base_k"] or (
                    cand["base_k"] == best_post["base_k"] and cand["base_spread"] > best_post["base_spread"]
                ):
                    best_post = cand


    if best_post is not None:
        comp = _complete_seedset_to_target(
            g=g,
            theta_vec=theta_vec,
            base_seeds=list(best_post["base_seeds"]),
            target_spread=int(target_spread),
            eval_fn=eval_fn,
            W=W,
            params=params,
            eval_kwargs=eval_kwargs,
            h0=h0,
        )
        best_post.update({
            "completed_seeds": comp["completed_seeds"],
            "completed_k": int(len(comp["completed_seeds"])),
            "completed_spread": int(comp["completed_spread"]),
            "added_nodes": list(comp["added_nodes"]),
        })
    df = pd.DataFrame(tested).sort_values(["solver", "k"]).reset_index(drop=True)





    if best_post is not None:
        sol_out = {
            "best_k_bisect": int(best_k_bisect),
            "best_k_eff": int(best_post["k_eff"]),
            "k_tested": int(best_post["k_tested"]),
            "best_spread": int(best_post["base_spread"]),
            "best_seeds": tuple(best_post["base_seeds"]),
            "missing_to_target": int(best_post["missing_to_target"]),
            "completed_seeds": tuple(best_post["completed_seeds"]),
            "completed_k": int(best_post["completed_k"]),
            "completed_spread": int(best_post["completed_spread"]),
            "added_nodes": list(best_post["added_nodes"]),
        }
        return int(sol_out["completed_k"]), df, sol_out

    return best_k_bisect, df, best_solution_at_k.get(best_k_bisect)

# ============================================================
#Euristiche
# ============================================================
def _cascade_fixpoint_mask(g: nx.Graph, active_mask: np.ndarray, theta_vec: np.ndarray, max_t: int = 999):
    t = 0
    while t < max_t:
        newly = connected_component_threshold_update_theta(g, active_mask, theta_vec)
        if not np.any(newly):
            break
        active_mask = active_mask | newly
        t += 1
    return active_mask


def run_heuristic_direct(
    heuristic_name: str,
    g: nx.Graph,
    theta_vec: np.ndarray,
    target_spread: int,
    *,
    force_max_degree_seed: bool = True,
    max_t_cascade: int = 999,
):
    N = g.number_of_nodes()
    t0 = time.time()

    active_mask = np.zeros(N, dtype=bool)
    seed_mask = np.zeros(N, dtype=bool)
    seeds = []

    bc = None
    if heuristic_name == "betweenness":
        bc = nx.betweenness_centrality(g, normalized=True)

    def add_seed(u: int):
        seed_mask[u] = True
        seeds.append(int(u))
        active_mask[u] = True

    if force_max_degree_seed:
        max_deg_node = max(g.nodes(), key=lambda u: (g.degree(int(u)), -int(u)))
        add_seed(int(max_deg_node))
        active_mask[:] = _cascade_fixpoint_mask(g, active_mask, theta_vec, max_t=max_t_cascade)

    def pick_degree():
        inactive = np.flatnonzero(~active_mask)
        return int(max(inactive, key=lambda u: (g.degree(int(u)), -int(u))))

    def pick_degree_threshold():
        inactive = np.flatnonzero(~active_mask)
        return int(max(
            inactive,
            key=lambda u: (g.degree(int(u)) * int(theta_vec[int(u)]), g.degree(int(u)), -int(u))
        ))

    def pick_betweenness():
        inactive = np.flatnonzero(~active_mask)
        return int(max(inactive, key=lambda u: (bc[int(u)], g.degree(int(u)), -int(u))))

    def pick_degree_discounted():
        inactive = np.flatnonzero(~active_mask)
        H = g.subgraph(inactive.tolist())
        return int(max(inactive, key=lambda u: (H.degree(int(u)), g.degree(int(u)), -int(u))))

    def pick_degree_connected():
        inactive = np.flatnonzero(~active_mask)
        if np.any(active_mask):
            active_nodes = set(np.flatnonzero(active_mask).tolist())
            frontier = []
            for u in inactive:
                u = int(u)
                for nb in g.neighbors(u):
                    if int(nb) in active_nodes:
                        frontier.append(u)
                        break
            pool = frontier if frontier else inactive.tolist()
        else:
            pool = inactive.tolist()
        return int(max(pool, key=lambda u: (g.degree(int(u)), -int(u))))

    while int(active_mask.sum()) < min(target_spread, N):
        if heuristic_name == "degree":
            u = pick_degree()
        elif heuristic_name == "degree-threshold":
            u = pick_degree_threshold()
        elif heuristic_name == "betweenness":
            u = pick_betweenness()
        elif heuristic_name == "degree discounted":
            u = pick_degree_discounted()
        elif heuristic_name == "degree connected":
            u = pick_degree_connected()
        else:
            raise ValueError(f"Unknown heuristic_name: {heuristic_name}")

        if seed_mask[u]:
            break
        add_seed(u)
        active_mask[:] = _cascade_fixpoint_mask(g, active_mask, theta_vec, max_t=max_t_cascade)

        if len(seeds) >= N:
            break

    seconds = time.time() - t0
    return {
        "solver": f"heur_{heuristic_name.replace(' ', '_').replace('-', '_')}",
        "k": int(seed_mask.sum()),
        "spread": int(active_mask.sum()),
        "seeds": tuple(sorted(seeds)),
        "seconds": round(seconds, 3),
    }


# ============================================================

# ============================================================
def run_ip_golberg_liu(g: nx.Graph, theta_vec: np.ndarray, time_limit: int, outdir: str):
    N = g.number_of_nodes()
    V = list(g.nodes())
    t0 = time.time()

    try:
        model, xvars = build_golberg_liu_ip(g, {i: int(theta_vec[i]) for i in range(N)})
        model.setParam("TimeLimit", float(time_limit))
        model.setParam("OutputFlag", 1)
        model.setParam("Threads", 0)
        model.optimize()

        status = int(model.Status)
        runtime = float(time.time() - t0)

        n = len(V)
        T_of = {int(u): None for u in V}
        for u in V:
            u = int(u)
            for t in range(1, n + 1):
                try:
                    val = xvars[u, t].X
                except Exception:
                    val = 0.0
                if val > 0.5:
                    T_of[u] = int(t)
                    break

        ip_seeds = [u for u in V if (T_of.get(int(u)) is not None and T_of[int(u)] <= int(theta_vec[int(u)]))]
        ip_seeds = [int(u) for u in ip_seeds]

        return {
            "ok": True,
            "status": status,
            "runtime": runtime,
            "seeds": ip_seeds,
            "T_of": T_of,
            "obj": float(model.ObjVal) if (hasattr(model, "ObjVal") and model.SolCount > 0) else None,
            "solcount": int(model.SolCount) if hasattr(model, "SolCount") else None,
        }

    except Exception as e:
        runtime = float(time.time() - t0)
        try:
            with open(os.path.join(outdir, "ip_error.txt"), "w", encoding="utf-8") as f:
                f.write("IP FAILED\n")
                f.write(f"Runtime: {runtime}\n")
                f.write(f"Error: {repr(e)}\n")
        except Exception:
            pass
        return {
            "ok": False,
            "status": None,
            "runtime": runtime,
            "seeds": [],
            "T_of": None,
            "obj": None,
            "solcount": None,
            "error": repr(e),
        }


def spread_from_seeds(g, eval_fn, W, params, eval_kwargs, seeds, h0=1.0):
    x0 = np.zeros(g.number_of_nodes(), dtype=float)
    x0[list(seeds)] = h0
    spread, _, _ = eval_fn(g, W, x0, params, **eval_kwargs)
    return int(spread)


# ============================================================
#MAIN
# ============================================================
def main():
    OUTROOT = "./results_ns_ip_batch"
    os.makedirs(OUTROOT, exist_ok=True)

    N = 300
    INIT_NODES = 5
    M = (1, 2, 3, 4)

    C_LIST = [1, 5, 10, 20]
    RUNS_PER_C = 4


    K_SEEDS = 50
    DELTA = 0.5
    XI = 0.01
    D = 2
    MAX_TIME = 120
    BUFFER_DIM = 5000


    PER_FAMILY = 1
    INCLUDE_BETWEENNESS = True
    CLOSE_MARGIN = 2
    EXTRA_RANDOM_STARTS = 4


    GUROBI_TIME_LIMIT = 1000


    h0 = 1.0
    W = np.zeros((N, N))
    params = [None, h0]
    eval_fn = Influence_evaluation_comp_threshold_theta


    K_LOW = 1
    K_HIGH = K_SEEDS

    BASE_SEED = 123
    all_rows = []

    heuristics_list = [
        "degree",
        "degree-threshold",
        "betweenness",
        "degree discounted",
        "degree connected",
    ]

    print(f"=== BATCH START {datetime.now().isoformat(timespec='seconds')} ===", flush=True)
    print(f"N={N}, c in {C_LIST}, runs per c={RUNS_PER_C}", flush=True)

    for c in C_LIST:
        min_conn_alpha = 0.7 if c in (1, 5) else 1.0

        for run_id in range(1, RUNS_PER_C + 1):
            exp_seed = BASE_SEED + 10_000 * c + run_id
            tag = f"n{N}_c{c}_run{run_id}"
            outdir = os.path.join(OUTROOT, tag)
            os.makedirs(outdir, exist_ok=True)

            print("\n" + "=" * 80, flush=True)
            print(f"[{tag}] seed={exp_seed}", flush=True)

            g = preferential_attachment_variable_m(
                N=N,
                seed=exp_seed,
                init_nodes=INIT_NODES,
                init_mode="complete",
                M=M,
            )

            theta_dict, allowed_values = assign_thresholds_paper_experiments(
                g, c=c, seed=exp_seed, attr="theta"
            )
            theta_vec = np.array([theta_dict[i] for i in range(N)], dtype=int)
            eval_kwargs = {"theta_vec": theta_vec, "h0": h0}




            rng_tmp = np.random.default_rng(exp_seed + 999)
            special_seedset_high, max_nodes, theta_max = build_max_theta_heavy_connected_seedset(
                g, theta_vec, K_HIGH, rng_tmp
            )
            max_nodes_set = set(int(x) for x in max_nodes)
            special_set = set(int(x) for x in special_seedset_high)
            missing_max_nodes = sorted(list(max_nodes_set - special_set))
            missing_count = len(missing_max_nodes)

            TARGET_SPREAD = N
            EARLY_STOP_SPREAD = TARGET_SPREAD

            with open(os.path.join(outdir, "instance_meta.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "N": N,
                        "c": c,
                        "run": run_id,
                        "seed": exp_seed,
                        "allowed_values": allowed_values,
                        "target_spread": TARGET_SPREAD,
                        "theta_max": int(theta_max),
                        "num_max_theta_nodes": int(len(max_nodes)),
                        "missing_max_theta_nodes_count": int(missing_count),
                        "missing_max_theta_nodes": missing_max_nodes,
                        "min_conn_alpha": min_conn_alpha,
                        "multistart": {
                            "per_family": PER_FAMILY,
                            "include_betweenness": INCLUDE_BETWEENNESS,
                            "close_margin": CLOSE_MARGIN,
                            "extra_random_starts": EXTRA_RANDOM_STARTS,
                            "special_first_start": "theta-max-heavy",
                        },
                        "ns": {
                            "DELTA": DELTA,
                            "XI": XI,
                            "D": D,
                            "MAX_TIME": MAX_TIME,
                            "BUFFER_DIM": BUFFER_DIM,
                        },
                        "ip": {
                            "TimeLimit": GUROBI_TIME_LIMIT
                        }
                    },
                    f,
                    indent=2,
                )

            solver_kwargs_base = {
                "thresholds": theta_vec,
                "max_outer_iters": 1000,
            }

            print(f"[{tag}] NS bisection (multistart)... alpha={min_conn_alpha}", flush=True)
            print(f"[{tag}] theta_max={theta_max} | max_nodes={len(max_nodes)} | missing_max={missing_count} -> TARGET={TARGET_SPREAD}", flush=True)
            t0 = time.time()
            best_k_prog, log_prog, sol_prog = bisection_multistart(
                solver_name="NS_min_conn_marginal",
                solver_fn=NS_marginal_gain_minconn_progressive,
                g=g, W=W, params=params, eval_fn=eval_fn, eval_kwargs=eval_kwargs,
                target_spread=TARGET_SPREAD,
                k_low=K_LOW, k_high=K_HIGH,
                rng_seed=exp_seed,
                DELTA=DELTA, XI=XI, D=D, MAX_TIME=MAX_TIME, BUFFER_DIM=BUFFER_DIM,
                h0=h0,
                solver_kwargs_base=solver_kwargs_base,
                early_stop_spread=EARLY_STOP_SPREAD,
                per_family=PER_FAMILY,
                include_betweenness=INCLUDE_BETWEENNESS,
                close_margin=CLOSE_MARGIN,
                extra_random_starts=EXTRA_RANDOM_STARTS,
                min_conn_alpha=min_conn_alpha,
            )
            t1 = time.time()
            print(f"[{tag}] NS done in {t1-t0:.1f}s; best_k={best_k_prog}", flush=True)




            ns_seeds = []
            ns_k = 0
            ns_spread = None

            completed_seeds = []
            completed_k = None
            completed_spread = None
            k_eff = None
            missing_to_target = None
            added_nodes = None

            try:
                if sol_prog is not None:
                    if sol_prog.get("best_seeds") is not None:
                        ns_seeds = list(sol_prog["best_seeds"])
                    ns_k = len(ns_seeds)
                    ns_spread = spread_from_seeds(g, eval_fn, W, params, eval_kwargs, ns_seeds, h0=h0)

                    if sol_prog.get("completed_seeds") is not None:
                        completed_seeds = list(sol_prog["completed_seeds"])
                        completed_k = int(sol_prog.get("completed_k", len(completed_seeds)))
                        completed_spread = int(sol_prog.get("completed_spread", spread_from_seeds(g, eval_fn, W, params, eval_kwargs, completed_seeds, h0=h0)))
                    k_eff = sol_prog.get("best_k_eff", None)
                    missing_to_target = sol_prog.get("missing_to_target", None)
                    added_nodes = sol_prog.get("added_nodes", None)
            except Exception:
                ns_seeds = []
                ns_k = 0
                ns_spread = None

            try:
                pd.DataFrame(log_prog).to_csv(os.path.join(outdir, "log_ns_min_conn.csv"), index=False)
            except Exception:
                pass
            with open(os.path.join(outdir, "sol_ns_min_conn.pkl"), "wb") as f:
                pickle.dump(sol_prog, f)
            with open(os.path.join(outdir, "ns_min_conn_seeds.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "k": ns_k,
                        "spread": ns_spread,
                        "seeds": ns_seeds,
                        "target_spread": TARGET_SPREAD,
                        "k_eff": k_eff,
                        
                        
                        "missing_to_target": missing_to_target,
                        
                        "added_nodes": added_nodes,
                        "completed_k": completed_k,
                        "completed_spread": completed_spread,
                        "completed_seeds": completed_seeds,
                    },
                    f,
                    indent=2
                )

            print(f"[{tag}] IP (Golberg-Liu)...", flush=True)
            ip_res = run_ip_golberg_liu(g, theta_vec, GUROBI_TIME_LIMIT, outdir=outdir)
            ip_seeds = ip_res.get("seeds", [])
            ip_k = len(ip_seeds)
            try:
                ip_spread = spread_from_seeds(g, eval_fn, W, params, eval_kwargs, ip_seeds, h0=h0) if ip_seeds else 0
            except Exception:
                ip_spread = None

            with open(os.path.join(outdir, "ip_result.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ok": ip_res.get("ok", False),
                        "status": ip_res.get("status", None),
                        "runtime": ip_res.get("runtime", None),
                        "solcount": ip_res.get("solcount", None),
                        "obj": ip_res.get("obj", None),
                        "k": ip_k,
                        "spread": ip_spread,
                        "seeds": ip_seeds,
                        "error": ip_res.get("error", None),
                    },
                    f,
                    indent=2,
                )
            if ip_res.get("T_of") is not None:
                with open(os.path.join(outdir, "ip_T_of.pkl"), "wb") as f:
                    pickle.dump(ip_res["T_of"], f)

            print(f"[{tag}] Heuristics...", flush=True)
            heur_results = []
            for hname in heuristics_list:
                r = run_heuristic_direct(
                    hname, g, theta_vec,
                    target_spread=TARGET_SPREAD,
                    force_max_degree_seed=True,
                    max_t_cascade=999,
                )
                heur_results.append(r)

            with open(os.path.join(outdir, "heuristic_seeds.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {r["solver"]: {"k": r["k"], "spread": r["spread"], "seeds": list(r["seeds"])} for r in heur_results},
                    f,
                    indent=2,
                )

            rows = []
            rows.append({"solver": "NS_min_conn_marginal", "k": ns_k, "spread": ns_spread})

            if completed_k is not None:
                rows.append({"solver": "NS_completed", "k": completed_k, "spread": completed_spread, "k_eff": k_eff,   "missing_to_target": missing_to_target})
            rows.append({"solver": "IP", "k": ip_k, "spread": ip_spread, "ip_ok": ip_res.get("ok"), "ip_status": ip_res.get("status")})
            for r in heur_results:
                rows.append({"solver": r["solver"], "k": r["k"], "spread": r["spread"]})
            df_summary = pd.DataFrame(rows).sort_values(["k", "solver"], na_position="last").reset_index(drop=True)
            df_summary.to_csv(os.path.join(outdir, "summary_IP_NS_heuristics.csv"), index=False)

            master_row = {
                "tag": tag,
                "N": N,
                "c": c,
                "run": run_id,
                "seed": exp_seed,
                "target_spread": TARGET_SPREAD,
                "theta_max": int(theta_max),
                "num_max_theta_nodes": int(len(max_nodes)),
                "missing_max_theta_nodes_count": int(missing_count),
                "min_conn_alpha": min_conn_alpha,
                "ns_k": ns_k,
                "ns_spread": ns_spread,
                "ns_k_eff": k_eff,
                
                
                "ns_missing_to_target": missing_to_target,
                "ns_completed_k": completed_k,
                "ns_completed_spread": completed_spread,
                "ip_ok": bool(ip_res.get("ok", False)),
                "ip_status": ip_res.get("status", None),
                "ip_k": ip_k,
                "ip_spread": ip_spread,
                "outdir": outdir,
            }
            for r in heur_results:
                pref = r["solver"]
                master_row[f"{pref}_k"] = r["k"]
                master_row[f"{pref}_spread"] = r["spread"]

            all_rows.append(master_row)

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(OUTROOT, "MASTER_summary.csv"), index=False)

    print("\n=== BATCH DONE ===", flush=True)
    print("Master:", os.path.join(OUTROOT, "MASTER_summary.csv"), flush=True)


if __name__ == "__main__":
    main()