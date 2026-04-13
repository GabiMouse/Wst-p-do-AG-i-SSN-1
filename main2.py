# ============================================================
# SWARM OPTIMIZATION PROJECT
# PSO (continuous + discrete), ACO, ABC
# + benchmarking on test functions and TSP
# ============================================================

import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from itertools import permutations
import math
from matplotlib.animation import FuncAnimation

# ============================================================
# REPRODUCIBILITY
# ============================================================

GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)
random.seed(GLOBAL_SEED)


# ============================================================
# TEST FUNCTIONS (OPTIMIZATION BENCHMARKS)
# ============================================================

def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


def schwefel(x):
    return 418.9829 * len(x) - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])


def eggholder(x):
    x1, x2 = x
    return -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))


# ============================================================
# INITIALIZATION STRATEGIES
# ============================================================

def init_uniform(pop, dim, bounds):
    return np.random.uniform(bounds[0], bounds[1], (pop, dim))


def init_lhs(pop, dim, bounds):
    """
    Simple Latin Hypercube Sampling (LHS)
    """
    result = np.zeros((pop, dim))
    seg = np.linspace(0, 1, pop + 1)

    for d in range(dim):
        perm = np.random.permutation(pop)
        for i in range(pop):
            low, high = seg[i], seg[i + 1]
            result[i, d] = np.random.uniform(low, high)

        result[:, d] = result[perm, d]

    return result * (bounds[1] - bounds[0]) + bounds[0]


# ============================================================
# STOP CRITERIA
# ============================================================

def should_stop(history, patience=20):
    """
    Stop if no improvement in last `patience` iterations.
    Safe version (handles short history).
    """

    if len(history) < patience * 2:
        return False

    recent = history[-patience:]
    previous = history[-2 * patience:-patience]

    return np.min(recent) >= np.min(previous)


# ============================================================
# PSO CONTINUOUS (IMPROVED)
# ============================================================

def pso(func, dim, bounds, pop=30, iters=50,
        w_max=0.9, w_min=0.4, c1=1.5, c2=1.5,
        init_method="uniform"):
    if init_method == "lhs":
        particles = init_lhs(pop, dim, bounds)
    else:
        particles = init_uniform(pop, dim, bounds)

    velocity = np.zeros((pop, dim))

    pbest = particles.copy()
    pbest_val = np.array([func(p) for p in particles])

    gbest = pbest[np.argmin(pbest_val)]

    history = []

    for t in range(iters):

        w = w_max - (w_max - w_min) * (t / iters)

        for i in range(pop):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)

            velocity[i] = (
                    w * velocity[i]
                    + c1 * r1 * (pbest[i] - particles[i])
                    + c2 * r2 * (gbest - particles[i])
            )

            particles[i] += velocity[i]
            particles[i] = np.clip(particles[i], bounds[0], bounds[1])

            val = func(particles[i])

            if val < pbest_val[i]:
                pbest[i] = particles[i]
                pbest_val[i] = val

        gbest = pbest[np.argmin(pbest_val)]
        best_val = np.min(pbest_val)

        history.append(best_val)

        if should_stop(history):
            break

    return gbest, best_val, history


# ============================================================
# ABC (IMPROVED WITH SCOUT PHASE)
# ============================================================

def abc(func, dim, bounds, pop=30, iters=50):
    solutions = init_uniform(pop, dim, bounds)
    fitness = np.array([func(s) for s in solutions])

    trial = np.zeros(pop)

    history = []

    for _ in range(iters):

        # EMPLOYED BEE PHASE
        for i in range(pop):
            k = random.randint(0, pop - 1)

            phi = np.random.uniform(-1, 1, dim)
            new = solutions[i] + phi * (solutions[i] - solutions[k])
            new = np.clip(new, bounds[0], bounds[1])

            val = func(new)

            if val < fitness[i]:
                solutions[i] = new
                fitness[i] = val
                trial[i] = 0
            else:
                trial[i] += 1

        # ONLOOKER PHASE
        prob = fitness.max() - fitness
        prob = prob / prob.sum()

        for _ in range(pop):
            i = np.random.choice(range(pop), p=prob)

            k = random.randint(0, pop - 1)
            phi = np.random.uniform(-1, 1, dim)

            new = solutions[i] + phi * (solutions[i] - solutions[k])
            new = np.clip(new, bounds[0], bounds[1])

            val = func(new)

            if val < fitness[i]:
                solutions[i] = new
                fitness[i] = val
                trial[i] = 0
            else:
                trial[i] += 1

        # SCOUT PHASE
        limit = 20
        for i in range(pop):
            if trial[i] > limit:
                solutions[i] = np.random.uniform(bounds[0], bounds[1], dim)
                fitness[i] = func(solutions[i])
                trial[i] = 0

        history.append(np.min(fitness))

        if should_stop(history):
            break

    return solutions[np.argmin(fitness)], np.min(fitness), history


# ============================================================
# TSP UTILITIES
# ============================================================

def get_best_run(algo, coords, runs=20):
    best_val = float("inf")
    best_data = None

    for _ in range(runs):
        result = algo(coords)

        if len(result) == 4:
            path, val, history, paths = result
        else:
            continue

        if val < best_val:
            best_val = val
            best_data = (path, val, history, paths)

    return best_data

def distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist


def tsp_length(path, dist):
    total = sum(dist[path[i], path[i + 1]] for i in range(len(path) - 1))
    total += dist[path[-1], path[0]]  # powrót do startu
    return total


def load_tsp(filename):
    coords = []
    with open(filename, "r") as f:
        lines = f.readlines()

    reading = False

    for line in lines:
        line = line.strip()

        if line.startswith("NODE_COORD_SECTION"):
            reading = True
            continue

        if line.startswith("EOF"):
            break

        if reading:
            parts = line.split()
            if len(parts) >= 3:
                x = float(parts[1])
                y = float(parts[2])
                coords.append([x, y])

    return np.array(coords)


# ============================================================
# ACO (IMPROVED)
# ============================================================

def aco_tsp(coords, ants=20, iters=50, alpha=1, beta=2, evap=0.5):
    n = len(coords)
    dist = distance_matrix(coords)

    pher = np.ones((n, n))

    best = float("inf")
    best_path = None

    history = []
    best_paths = []

    for _ in range(iters):

        all_paths = []

        for _ in range(ants):
            path = [random.randint(0, n - 1)]

            while len(path) < n:
                current = path[-1]

                probs = []
                for j in range(n):
                    if j in path:
                        probs.append(0)
                    else:
                        eta = 1 / (dist[current][j] + 1e-10)
                        probs.append((pher[current][j] ** alpha) * (eta ** beta))

                probs = np.array(probs)
                probs = probs / probs.sum()

                path.append(np.random.choice(range(n), p=probs))

            all_paths.append(path)

        pher *= (1 - evap)

        for path in all_paths:
            L = tsp_length(path, dist)

            if L < best:
                best = L
                best_path = path.copy()

            for i in range(n - 1):
                pher[path[i], path[i + 1]] += 1 / L

        history.append(best)
        best_paths.append(best_path.copy())

        if should_stop(history):
            break

    return best_path, best, history, best_paths


# ============================================================
# DISCRETE PSO FOR TSP (IMPROVED)
# ============================================================

def pso_tsp(coords, pop=30, iters=50):
    n = len(coords)
    dist = distance_matrix(coords)

    particles = [list(np.random.permutation(n)) for _ in range(pop)]

    pbest = particles.copy()
    pbest_val = [tsp_length(p, dist) for p in particles]

    gbest = pbest[np.argmin(pbest_val)]
    best = min(pbest_val)

    history = []
    best_paths = [gbest.copy()]

    for _ in range(iters):

        for i in range(pop):

            new = particles[i].copy()

            # ruch do pbest
            for _ in range(3):
                a = random.randint(0, n - 1)
                b = new.index(pbest[i][a])
                new[a], new[b] = new[b], new[a]

            # ruch do gbest
            for _ in range(3):
                a = random.randint(0, n - 1)
                b = new.index(gbest[a])
                new[a], new[b] = new[b], new[a]

            val = tsp_length(new, dist)

            if val < pbest_val[i]:
                pbest[i] = new
                pbest_val[i] = val

        gbest = pbest[np.argmin(pbest_val)]
        best = min(pbest_val)

        history.append(best)
        best_paths.append(gbest.copy())

        if should_stop(history):
            break

    return gbest, best, history, best_paths


# ============================================================
# EXPERIMENTS (FUNCTION OPTIMIZATION)
# ============================================================

def run_function_experiments():
    functions = [
        ("Rastrigin", rastrigin, (-5.12, 5.12)),
        ("Schwefel", schwefel, (-500, 500)),
        ("Eggholder", eggholder, (-512, 512))
    ]

    dims = [2, 5, 10]
    pops = [10, 20, 50, 100]
    iters = [50, 100, 200]
    runs = 20

    results = []

    for name, func, bounds in functions:
        for dim in dims:

            if name == "Eggholder" and dim != 2:
                continue

            for pop in pops:
                for it in iters:

                    for algo_name, algo in [
                        ("PSO", pso),
                        ("ABC", abc)
                    ]:

                        vals = []
                        times = []

                        for _ in range(runs):
                            start = time.time()
                            _, val, history = algo(func, dim, bounds, pop, it)
                            times.append(time.time() - start)
                            vals.append(val)

                        results.append([
                            name, dim, algo_name, pop, it,
                            np.mean(vals), np.std(vals), np.min(vals), np.mean(times)
                        ])

                        plt.plot(history)
                        plt.title(f"{algo_name}-{name}-D{dim}")
                        plt.xlabel("Iteration")
                        plt.ylabel("Best value")
                        plt.savefig(f"{algo_name}_{name}_{dim}.png")
                        plt.clf()

    return pd.DataFrame(results, columns=[
        "Function", "Dim", "Algorithm", "Population",
        "Iterations", "Mean", "Std", "Best", "Time"
    ])


# ============================================================
# EXPERIMENTS (TSP)
# ============================================================

def run_tsp_experiments():
    tsp_files = {
        "burma14": "burma14.tsp",
        "ulysses22": "ulysses22.tsp",
        "berlin52": "berlin52.tsp"
    }

    runs = 20
    results = []

    for name, path in tsp_files.items():

        coords = load_tsp(path)
        n = len(coords)

        for algo_name, algo in [
            ("ACO", aco_tsp),
            ("PSO_TSP", pso_tsp)
        ]:

            vals = []
            times = []

            for _ in range(runs):
                start = time.time()
                _, val, history, _ = algo(coords)
                times.append(time.time() - start)
                vals.append(val)

            results.append([
                name, n, algo_name,
                np.mean(vals),
                np.std(vals),
                np.min(vals),
                np.mean(times)
            ])

            plt.plot(history)
            plt.title(f"{algo_name}-{name}")
            plt.xlabel("Iteration")
            plt.ylabel("Best distance")
            plt.savefig(f"{algo_name}_{name}.png")
            plt.clf()

    return pd.DataFrame(results, columns=[
        "Instance", "Cities", "Algorithm", "Mean", "Std", "Best", "Time"
    ])


def animate_pso(func, dim, bounds, pop=30, iters=50, save_path="pso.gif"):
    """
    Animated PSO in 2D (for visualization only).
    Works ONLY when dim == 2.
    """

    if dim != 2:
        raise ValueError("Animation supports only 2D problems")

    particles = np.random.uniform(bounds[0], bounds[1], (pop, dim))
    velocity = np.zeros((pop, dim))

    pbest = particles.copy()
    pbest_val = np.array([func(p) for p in particles])
    gbest = pbest[np.argmin(pbest_val)]

    fig, ax = plt.subplots()
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])

    scatter = ax.scatter(particles[:, 0], particles[:, 1])

    def update(frame):

        nonlocal particles, velocity, pbest, pbest_val, gbest

        w, c1, c2 = 0.7, 1.5, 1.5

        for i in range(pop):
            r1, r2 = np.random.rand(2)

            velocity[i] = (
                    w * velocity[i]
                    + c1 * r1 * (pbest[i] - particles[i])
                    + c2 * r2 * (gbest - particles[i])
            )

            particles[i] += velocity[i]
            particles[i] = np.clip(particles[i], bounds[0], bounds[1])

            val = func(particles[i])

            if val < pbest_val[i]:
                pbest[i] = particles[i]
                pbest_val[i] = val

        gbest = pbest[np.argmin(pbest_val)]

        scatter.set_offsets(particles)

        ax.set_title(f"PSO iteration {frame}")

        return scatter,

    anim = FuncAnimation(fig, update, frames=iters, interval=200, blit=True)
    anim.save(save_path, writer="pillow")
    plt.close()

    return save_path


# ============================================================
# ACO ANIMATION (TSP)
# ============================================================

def animate_aco(coords, ants=10, iters=30, save_path="aco.gif"):
    """
    Animated ACO showing best path evolution.
    """

    n = len(coords)
    dist = distance_matrix(coords)

    pher = np.ones((n, n))
    best_path = None
    best_len = float("inf")

    fig, ax = plt.subplots()

    def draw_path(path):
        ax.clear()
        ax.scatter(coords[:, 0], coords[:, 1])

        for i in range(len(path) - 1):
            a, b = path[i], path[i + 1]
            ax.plot(
                [coords[a][0], coords[b][0]],
                [coords[a][1], coords[b][1]],
                "b-"
            )

        ax.set_title(f"ACO best length: {best_len:.2f}")

    def update(frame):

        nonlocal pher, best_path, best_len

        paths = []

        for _ in range(ants):

            path = [random.randint(0, n - 1)]

            while len(path) < n:
                current = path[-1]

                probs = []
                for j in range(n):
                    if j in path:
                        probs.append(0)
                    else:
                        eta = 1 / (dist[current][j] + 1e-9)
                        probs.append(pher[current][j] * eta ** 2)

                probs = np.array(probs)
                probs = probs / probs.sum()

                path.append(np.random.choice(range(n), p=probs))

            paths.append(path)

        pher *= 0.5

        for path in paths:
            L = tsp_length(path, dist)

            if L < best_len:
                best_len = L
                best_path = path

            for i in range(n - 1):
                pher[path[i], path[i + 1]] += 1 / L

        draw_path(best_path)

        return []

    anim = FuncAnimation(fig, update, frames=iters, interval=300, blit=False)
    anim.save(save_path, writer="pillow")
    plt.close()

    return save_path


def animate_tsp(coords, best_paths, title, save_path):
    dist = distance_matrix(coords)

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        path = best_paths[frame]

        ax.scatter(coords[:, 0], coords[:, 1])

        for i in range(len(path)):
            a = path[i]
            b = path[(i + 1) % len(path)]

            ax.plot(
                [coords[a][0], coords[b][0]],
                [coords[a][1], coords[b][1]],
                color="blue", linewidth=2
            )

        ax.set_title(f"{title} | iter {frame} | len={tsp_length(path, dist):.2f}")

    anim = FuncAnimation(fig, update, frames=len(best_paths), interval=300)
    anim.save(save_path, writer="pillow")
    plt.close()


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":

    df1 = run_function_experiments()
    df2 = run_tsp_experiments()

    df1.to_csv("function_results.csv", index=False)
    df2.to_csv("tsp_results.csv", index=False)

    print(df1)
    print(df2)

    tsp_files = {
        "burma14": "burma14.tsp",
        "ulysses22": "ulysses22.tsp",
        "berlin52": "berlin52.tsp"
    }

    for name, path in tsp_files.items():

        print(f"\n=== {name} ===")

        coords = load_tsp(path)

        # --- ACO ---
        aco_best = get_best_run(aco_tsp, coords)
        path_best, val, _, paths = aco_best

        animate_tsp(
            coords,
            paths,
            f"ACO - {name}",
            f"aco_best_{name}.gif"
        )

        # --- PSO ---
        pso_best = get_best_run(pso_tsp, coords)
        path_best, val, _, paths = pso_best

        animate_tsp(
            coords,
            paths,
            f"PSO_TSP - {name}",
            f"pso_best_{name}.gif"
        )