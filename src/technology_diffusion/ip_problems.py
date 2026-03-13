from __future__ import annotations

import time
from typing import Any, Mapping, Optional

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

from .helpers import time_exceeded


def build_golberg_liu_ip(
    g: nx.Graph,
    theta: Mapping[int, int],
    max_time: Optional[float] = None,
    env: Optional[gp.Env] = None,
) -> tuple[Any | None, Any | None]:
    start_time = time.time()
    v = list(g.nodes())
    n = len(v)
    t_range = range(1, n + 1)

    m = gp.Model("golberg_liu_ip", env=env)
    x = m.addVars(v, t_range, vtype=GRB.BINARY, name="x")

    m.setObjective(
        gp.quicksum(x[u, t] for u in v for t in t_range if t <= int(theta[u])),
        GRB.MINIMIZE,
    )

    for u in v:
        if time_exceeded(start_time, max_time):
            return None, None
        m.addConstr(gp.quicksum(x[u, t] for t in t_range) == 1, name=f"perm_node[{u}]")

    for t in t_range:
        if time_exceeded(start_time, max_time):
            return None, None
        m.addConstr(gp.quicksum(x[u, t] for u in v) == 1, name=f"perm_time[{t}]")

    for u in v:
        if time_exceeded(start_time, max_time):
            return None, None
        neighbors = list(g.neighbors(u))
        for t in range(2, n + 1):
            if time_exceeded(start_time, max_time):
                return None, None
            if neighbors:
                m.addConstr(
                    gp.quicksum(x[w, tp] for w in neighbors for tp in range(1, t)) >= x[u, t],
                    name=f"conn[{u},{t}]",
                )
            else:
                m.addConstr(x[u, t] == 0, name=f"isolated[{u},{t}]")

    m.update()
    return m, x


def get_activation_sequence_and_seeds(
    g: nx.Graph,
    theta: Mapping[int, int],
    xvars: Any,
) -> tuple[dict[int, int], list[int], list[int]]:
    v = list(g.nodes())
    n = len(v)
    t_of = {}

    for i in v:
        for t in range(1, n + 1):
            if xvars[i, t].X > 0.5:
                t_of[i] = t
                break

    seeds = [i for i in v if t_of[i] <= int(theta[i])]
    order = sorted(v, key=lambda i: t_of[i])
    return t_of, order, seeds


def build_marta_ip(
    g: nx.Graph,
    theta: Mapping[int, int],
    k: int,
    use_simultaneous: bool = True,
    time_horizon: Optional[int] = None,
    max_time: Optional[float] = None,
) -> tuple[Any | None, Any | None, Any | None, Any | None]:
    start_time = time.time()
    v = list(g.nodes())
    n_nodes = len(v)
    if time_horizon is None:
        time_horizon = n_nodes

    t = range(1, time_horizon + 1)
    t0 = range(0, time_horizon + 1)
    r = range(1, k + 1)

    model = gp.Model("component_threshold_diffusion")
    x = model.addVars(v, t0, vtype=GRB.BINARY, name="x")
    y = model.addVars(v, r, t0, vtype=GRB.BINARY, name="y")
    nvar = model.addVars(v, r, t, vtype=GRB.INTEGER, name="n")

    model.setObjective(gp.quicksum(x[i, 0] for i in v), GRB.MINIMIZE)

    if time_exceeded(start_time, max_time):
        return None, None, None, None
    model.addConstr(gp.quicksum(x[i, tp] for i in v for tp in t0) == n_nodes, name="total_activations_equals_n")

    for i in v:
        if time_exceeded(start_time, max_time):
            return None, None, None, None
        model.addConstr(gp.quicksum(x[i, tp] for tp in t0) == 1, name=f"activate_once[{i}]")

    for tp in t:
        if time_exceeded(start_time, max_time):
            return None, None, None, None
        model.addConstr(gp.quicksum(x[i, tp] for i in v) <= 1, name=f"one_per_time[{tp}]")

    for i in v:
        th = float(theta[i])
        for tp in t:
            if time_exceeded(start_time, max_time):
                return None, None, None, None
            model.addConstr(
                th * x[i, tp] <= gp.quicksum(nvar[i, rr, tp] for rr in r),
                name=f"threshold[{i},{tp}]",
            )

    for i in v:
        for rr in r:
            for tp in t:
                if time_exceeded(start_time, max_time):
                    return None, None, None, None
                growth = (
                    gp.quicksum(y[j, rr, 0] for j in v)
                    + gp.quicksum(y[j, rr, tau] - y[j, rr, tau - 1] for j in v for tau in range(1, tp + 1))
                )
                model.addConstr(nvar[i, rr, tp] <= growth, name=f"growth_ub[{i},{rr},{tp}]")

    for i in v:
        ni = list(g.neighbors(i))
        for rr in r:
            for tp in t:
                if time_exceeded(start_time, max_time):
                    return None, None, None, None
                model.addConstr(
                    nvar[i, rr, tp] <= n_nodes * gp.quicksum(y[j, rr, tp] for j in ni),
                    name=f"visible[{i},{rr},{tp}]",
                )

    for i in v:
        for tp in t0:
            if time_exceeded(start_time, max_time):
                return None, None, None, None
            model.addConstr(gp.quicksum(y[i, rr, tp] for rr in r) <= 1, name=f"unique_label[{i},{tp}]")

    for i in v:
        for rr in r:
            for tp in t0:
                if time_exceeded(start_time, max_time):
                    return None, None, None, None
                model.addConstr(
                    y[i, rr, tp] <= gp.quicksum(x[i, tau] for tau in range(0, tp + 1)),
                    name=f"y_implies_active[{i},{rr},{tp}]",
                )

    for rr in r:
        if time_exceeded(start_time, max_time):
            return None, None, None, None
        model.addConstr(gp.quicksum(y[i, rr, 0] for i in v) <= 1, name=f"one_seed[{rr}]")

    for i in v:
        ni = list(g.neighbors(i))
        for rr in r:
            for tp in t:
                if time_exceeded(start_time, max_time):
                    return None, None, None, None
                rhs = y[i, rr, tp - 1] + gp.quicksum(
                    y[j, rr, tp] if use_simultaneous else y[j, rr, tp - 1] for j in ni
                )
                model.addConstr(y[i, rr, tp] <= rhs, name=f"prop[{i},{rr},{tp}]")

    model.update()
    return model, x, y, nvar
