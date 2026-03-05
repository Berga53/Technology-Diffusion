import random

import networkx as nx
import numpy as np

from .helpers import connected_component_spread, make_subset_connected, make_vector


def SingleDiscount(g, thetas, k):
    w1 = nx.to_numpy_array(g)
    idx = []
    ddv = np.sum(w1 > 0, axis=0)
    tv = np.zeros(len(g.nodes))

    for _ in range(k):
        u = int(np.argmax(ddv))
        idx.append(u)
        v = list(np.nonzero(w1[u])[0])
        tv[v] += 1
        ddv[v] = ddv[v] - tv[v]
        ddv[u] = -1
        w1[u, :] = 0
        w1[:, u] = 0

    x = np.zeros(len(g.nodes))
    x[idx] = 1.0
    s = connected_component_spread(g, x, thetas, max_t=1000)[0]
    return s, x


def degree(g, n_nodes, k, thetas=None, connected=0):
    degs = np.array([g.degree(v) for v in g.nodes()])
    top_k = np.argsort(-degs)[:k]
    if connected:
        top_k = list(make_subset_connected(g, top_k, use_thetas=1, thetas=thetas))
    return make_vector(top_k, n_nodes)


def degree_threshold(g, n_nodes, k, thetas=None, connected=0):
    dt = np.array([g.degree(v) for v in g.nodes()]) * np.array(list(thetas.values()))
    top_k = np.argsort(-dt)[:k]
    if connected:
        top_k = list(make_subset_connected(g, top_k, use_thetas=1, thetas=thetas))
    return make_vector(top_k, n_nodes)


def betweenness(g, n_nodes, k, thetas=None, connected=0):
    bets = nx.betweenness_centrality(g)
    bet_values = np.array([bets[v] for v in g.nodes()])
    top_k = np.argsort(-bet_values)[:k]
    if connected:
        top_k = list(make_subset_connected(g, top_k, use_thetas=1, thetas=thetas))
    return make_vector(top_k, n_nodes)


def degree_connected(g, n_nodes, k, thetas=None, connected=None):
    degs = np.array([g.degree(v) for v in g.nodes()])
    top_k = [np.argsort(-degs)[0]]
    neighbors = set(g.neighbors(top_k[0]))

    for _ in range(1, k):
        candidates = [v for v in g.nodes() if v not in top_k and v in neighbors]
        if not candidates:
            break
        c_degs = np.array([g.degree(v) for v in candidates])
        top_candidate = candidates[np.argsort(-c_degs)[0]]
        top_k.append(top_candidate)
        neighbors.update(g.neighbors(top_candidate))

    return make_vector(top_k, n_nodes)


def high_thetas_start(g, n_nodes, k, thetas=None, connected=1):
    sorted_nodes = sorted(g.nodes(), key=lambda x: thetas[x], reverse=True)
    top_k = sorted_nodes[:k]
    if connected:
        top_k = list(make_subset_connected(g, top_k, use_thetas=1, thetas=thetas))
    return make_vector(top_k, n_nodes)


def SD_start(g, n_nodes, k, thetas=None, connected=1):
    _, x0 = SingleDiscount(g, thetas, k)
    if connected:
        x0 = make_vector(
            make_subset_connected(g, list(np.nonzero(x0)[0]), use_thetas=1, thetas=thetas),
            n_nodes,
        )
    return x0


def random_start(g, n_nodes, k, thetas=None, connected=1):
    nodes = list(g.nodes())
    random.shuffle(nodes)
    top_k = nodes[:k]
    if connected:
        top_k = list(make_subset_connected(g, top_k, use_thetas=1, thetas=thetas))
    return make_vector(top_k, n_nodes)
