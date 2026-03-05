import collections
import itertools
import math
import random
import time

import numpy as np

from .helpers import connected_component_spread, time_exceeded


def Neighbor_Search(g, thetas, x0, delta, xi, d, max_time, buffer_dim, verbose=0):
    n_nodes = len(x0)
    x_hist = [np.array(x0, dtype=float)]
    s_hist = [connected_component_spread(g, x0, thetas=thetas, max_t=1000)[0]]
    r = 0
    xi_t = xi
    stop = False
    buffer = collections.deque(maxlen=buffer_dim)
    calls = 1
    start = time.time()
    history = [[s_hist[-1], time.time() - start, calls]]

    def print_status(spread, done=False):
        if not verbose:
            return
        msg = (
            f"Neighbors search... Done! Calls:{calls}. Time: {round(time.time()-start)}/{max_time} s. "
            f"Influence spread: {spread}."
            if done
            else f"Neighbors search... Calls:{calls}. Time: {round(time.time()-start)}/{max_time} s. "
            f"Influence spread: {spread}."
        )
        print("\r" + msg + " " * 10, end="" if not done else None)

    def evaluate_neighbors(neighbors, s_base, x_base):
        nonlocal calls, stop
        s_temp = s_base
        x_temp = x_base

        for elem in neighbors:
            x_elem = np.zeros(n_nodes)
            x_elem[list(elem)] = 1.0
            key = tuple(x_elem)

            if key in buffer:
                continue
            if time_exceeded(start, max_time):
                stop = True
                break

            s_elem = connected_component_spread(g, x_elem, thetas=thetas, max_t=1000)[0]
            buffer.append(key)
            calls += 1
            print_status(s_temp)

            if s_elem > s_temp:
                history.append([s_elem, time.time() - start, calls])
                x_temp = x_elem
                s_temp = s_elem

                if s_elem == n_nodes:
                    return True, x_temp, s_temp
                if s_temp > (1 + xi_t) * s_base:
                    break

        return False, x_temp, s_temp

    print_status(s_hist[-1])

    if s_hist[-1] == n_nodes:
        print_status(s_hist[-1], done=True)
        return s_hist, x_hist, history

    while (not stop) and (r < 1000):
        if time_exceeded(start, max_time):
            break

        idx = set(np.nonzero(x_hist[-1])[0])
        neighbors = []

        for elem1 in idx:
            for elem2 in set(g.neighbors(elem1)) - idx:
                temp = idx.copy()
                temp.add(elem2)
                temp.remove(elem1)
                neighbors.append(temp)

        found_opt, x_temp, s_temp = evaluate_neighbors(neighbors, s_hist[-1], x_hist[-1])
        if found_opt:
            x_hist.append(x_temp)
            s_hist.append(s_temp)
            print_status(s_hist[-1], done=True)
            return s_hist, x_hist, history

        x_hist.append(x_temp)
        s_hist.append(s_temp)
        r += 1

        if s_hist[-1] > (1 + xi_t) * s_hist[-2]:
            continue
        if s_hist[-1] > s_hist[-2]:
            xi_t *= delta
            continue

        x_hist.pop()
        s_hist.pop()

        idx = set(np.nonzero(x_hist[-1])[0])
        idx_temp = set(range(n_nodes)) - idx
        neighbors = []

        for i in range(d // 2):
            for elem1 in itertools.combinations(idx, len(idx) - 1 - i):
                for elem2 in itertools.combinations(idx_temp, i + 1):
                    neighbors.append(list(set(elem1) | set(elem2)))

        found_opt, x_temp, s_temp = evaluate_neighbors(neighbors, s_hist[-1], x_hist[-1])
        if found_opt:
            x_hist.append(x_temp)
            s_hist.append(s_temp)
            print_status(s_hist[-1], done=True)
            return s_hist, x_hist, history

        x_hist.append(x_temp)
        s_hist.append(s_temp)
        r += 1

        if s_hist[-1] == s_hist[-2]:
            x_hist.pop()
            s_hist.pop()
            stop = True
        elif s_hist[-1] <= (1 + xi_t) * s_hist[-2]:
            xi_t *= delta

    history.append([s_hist[-1], time.time() - start, calls])
    print_status(s_hist[-1], done=True)
    return s_hist, x_hist, history


def NS_technology_diffusion_binary_search(g, thetas, strategy, delta, xi, d, max_time, buffer_dim, verbose=0):
    start = time.time()
    n_nodes = g.number_of_nodes()
    tried_k = set()
    best_k = None
    best_solution_x = None
    top_k, bottom_k = n_nodes, 1
    time_single = max_time / max(1, math.ceil(math.log2(n_nodes)))
    times = {}
    strategy_tried = {}
    temp_x = {}

    while bottom_k <= top_k and time.time() - start < max_time:
        k = (top_k + bottom_k) // 2
        if k in tried_k:
            break
        tried_k.add(k)

        x = strategy[0](g, n_nodes, k, thetas=thetas, connected=1)
        s, final_x, history = Neighbor_Search(g, thetas, x, delta, xi, d, time_single, buffer_dim, verbose=0)
        spread = s[-1]
        x_last = np.array(final_x[-1], dtype=float)
        times[k] = history[-1][1]
        strategy_tried[k] = -1 if times[k] < 0.9 * time_single else 0
        if strategy_tried[k] == -1:
            temp_x[k] = x_last

        if spread == n_nodes:
            if best_k is None or k < best_k:
                best_k = k
                best_solution_x = x_last.copy()
            top_k = k - 1
        else:
            inferred_success_k = k + (n_nodes - spread)
            if best_k is None or inferred_success_k < best_k:
                _, _, active_after = connected_component_spread(g, x_last, thetas, max_t=1000)
                inferred_x = x_last.copy()
                inferred_x[np.where(active_after == 0)[0]] = 1.0
                best_k = inferred_success_k
                best_solution_x = inferred_x

            if inferred_success_k <= top_k:
                top_k = inferred_success_k - 1
            bottom_k = k + 1

    random.seed(42)
    while time.time() - start < max_time and best_k is not None and best_k > 1:
        k = best_k - 1
        if strategy_tried.get(k) is None:
            strat = 0
            x = strategy[strat](g, n_nodes, k, thetas=thetas, connected=1)
        elif strategy_tried[k] == -1:
            strat = 0
            x = temp_x[k]
        else:
            strat = min(len(strategy) - 1, strategy_tried[k] + 1)
            x = strategy[strat](g, n_nodes, k, thetas=thetas, connected=1)

        remaining = max_time - (time.time() - start)
        if remaining <= 0:
            break

        s, final_x, _ = Neighbor_Search(
            g, thetas, x, delta, xi, d, min(time_single, remaining), buffer_dim, verbose=0
        )
        strategy_tried[k] = strat

        if s[-1] == n_nodes:
            best_k = k
            best_solution_x = np.array(final_x[-1], dtype=float)
        else:
            break

    return best_k, best_solution_x, time.time() - start
