import numpy as np
import pandas as pd
import random
import time
import matplotlib.pyplot as plt


# ==============================
# FUNKCJE TESTOWE
# ==============================

def rastrigin(x):
    A = 10
    return A * len(x) + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x])


def schwefel(x):
    return 418.9829 * len(x) - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])


def eggholder(x):
    x1, x2 = x
    return -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1 / 2 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))


# ==============================
# PSO
# ==============================

def pso(func, dim, bounds, num_particles=30, max_iter=50):
    particles = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    pbest = particles.copy()
    pbest_val = np.array([func(p) for p in particles])

    gbest = pbest[np.argmin(pbest_val)]

    w, c1, c2 = 0.7, 1.5, 1.5
    history = []

    for _ in range(max_iter):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()

            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (pbest[i] - particles[i]) +
                             c2 * r2 * (gbest - particles[i]))

            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], bounds[0], bounds[1])

            val = func(particles[i])

            if val < pbest_val[i]:
                pbest[i] = particles[i]
                pbest_val[i] = val

        gbest = pbest[np.argmin(pbest_val)]
        history.append(min(pbest_val))

    return gbest, min(pbest_val), history


# ==============================
# ABC (POPRAWIONE)
# ==============================

def abc(func, dim, bounds, num_bees=30, max_iter=50):
    solutions = np.random.uniform(bounds[0], bounds[1], (num_bees, dim))
    fitness = np.array([func(s) for s in solutions])

    history = []

    for _ in range(max_iter):
        # employed
        for i in range(num_bees):
            k = random.randint(0, num_bees - 1)
            phi = np.random.uniform(-1, 1, dim)

            new = solutions[i] + phi * (solutions[i] - solutions[k])
            new = np.clip(new, bounds[0], bounds[1])

            val = func(new)
            if val < fitness[i]:
                solutions[i] = new
                fitness[i] = val

        # POPRAWIONE PRAWDOPODOBIEŃSTWO
        prob = 1 / (1 + fitness)
        prob = prob / np.sum(prob)

        # onlooker
        for i in range(num_bees):
            if random.random() < prob[i]:
                k = random.randint(0, num_bees - 1)
                phi = np.random.uniform(-1, 1, dim)

                new = solutions[i] + phi * (solutions[i] - solutions[k])
                new = np.clip(new, bounds[0], bounds[1])

                val = func(new)
                if val < fitness[i]:
                    solutions[i] = new
                    fitness[i] = val

        history.append(min(fitness))

    best = solutions[np.argmin(fitness)]
    return best, min(fitness), history


# ==============================
# TSP
# ==============================

def distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist


def tsp_distance(path, dist):
    return sum(dist[path[i]][path[i + 1]] for i in range(len(path) - 1))


# ==============================
# ACO
# ==============================

def aco_tsp(coords, num_ants=20, max_iter=50):
    n = len(coords)
    dist = distance_matrix(coords)

    pheromone = np.ones((n, n))
    best_length = float('inf')
    history = []

    for _ in range(max_iter):
        paths = []

        for _ in range(num_ants):
            path = [random.randint(0, n - 1)]

            while len(path) < n:
                current = path[-1]
                probs = []

                for j in range(n):
                    if j not in path:
                        probs.append((1 / dist[current][j]) ** 2 * pheromone[current][j])
                    else:
                        probs.append(0)

                probs = np.array(probs)
                probs = probs / np.sum(probs)

                path.append(np.random.choice(range(n), p=probs))

            paths.append(path)

        pheromone *= 0.5

        for path in paths:
            length = tsp_distance(path, dist)

            if length < best_length:
                best_length = length

            for i in range(n - 1):
                pheromone[path[i]][path[i + 1]] += 1 / length

        history.append(best_length)

    return best_length, history


# ==============================
# PSO DYSKRETNE
# ==============================

def pso_tsp(coords, num_particles=30, max_iter=50):
    n = len(coords)
    dist = distance_matrix(coords)

    particles = [list(np.random.permutation(n)) for _ in range(num_particles)]
    pbest = particles.copy()
    pbest_val = [tsp_distance(p, dist) for p in particles]

    gbest = pbest[np.argmin(pbest_val)]
    history = []

    for _ in range(max_iter):
        for i in range(num_particles):
            new = particles[i].copy()

            for _ in range(2):
                a = random.randint(0, n - 1)
                b = new.index(pbest[i][a])
                new[a], new[b] = new[b], new[a]

            for _ in range(2):
                a = random.randint(0, n - 1)
                b = new.index(gbest[a])
                new[a], new[b] = new[b], new[a]

            val = tsp_distance(new, dist)

            if val < pbest_val[i]:
                pbest[i] = new
                pbest_val[i] = val

        gbest = pbest[np.argmin(pbest_val)]
        history.append(min(pbest_val))

    return min(pbest_val), history


# ==============================
# EKSPERYMENTY
# ==============================

def run_function_experiments():
    functions = [
        ("Rastrigin", rastrigin, (-5.12, 5.12)),
        ("Schwefel", schwefel, (-500, 500)),
        ("Eggholder", eggholder, (-512, 512))
    ]

    dims = [2, 5, 10]
    populations = [10, 20, 50, 100]
    iterations = [50, 100, 200]
    runs = 20

    results = []

    for name, func, bounds in functions:
        for dim in dims:
            if name == "Eggholder" and dim != 2:
                continue

            for pop in populations:
                for it in iterations:

                    for algo_name, algo in [("PSO", pso), ("ABC", abc)]:
                        vals = []
                        times = []

                        for _ in range(runs):
                            start = time.time()
                            _, val, history = algo(func, dim, bounds, pop, it)
                            times.append(time.time() - start)
                            vals.append(val)

                        results.append([
                            name, dim, algo_name, pop, it,
                            np.mean(vals), np.std(vals), np.mean(times)
                        ])

                        # wykres konwergencji
                        plt.plot(history)
                        plt.title(f"{algo_name}-{name}-dim{dim}")
                        plt.xlabel("Iteracje")
                        plt.ylabel("Best value")
                        plt.savefig(f"{algo_name}_{name}_{dim}.png")
                        plt.clf()

    return pd.DataFrame(results, columns=[
        "Function", "Dim", "Algorithm", "Population", "Iterations",
        "Mean", "Std", "Time"
    ])


def run_tsp_experiments():
    np.random.seed(42)
    city_sizes = [10, 20, 50]
    runs = 20

    results = []

    coords_dict = {n: np.random.rand(n, 2) * 100 for n in city_sizes}

    for n in city_sizes:
        coords = coords_dict[n]

        for algo_name, algo in [("ACO", aco_tsp), ("PSO", pso_tsp)]:
            vals = []
            times = []

            for _ in range(runs):
                start = time.time()
                val, history = algo(coords)
                times.append(time.time() - start)
                vals.append(val)

            results.append([
                n, algo_name,
                np.mean(vals), np.std(vals), np.mean(times)
            ])

            plt.plot(history)
            plt.title(f"{algo_name}-TSP-{n}")
            plt.savefig(f"{algo_name}_TSP_{n}.png")
            plt.clf()

    return pd.DataFrame(results, columns=[
        "Cities", "Algorithm", "Mean", "Std", "Time"
    ])


# ==============================
# MAIN
# ==============================


if __name__ == "__main__":
    df_func = run_function_experiments()
    df_tsp = run_tsp_experiments()

    print(df_func)
    print(df_tsp)

    df_func.to_csv("function_results.csv", index=False)
    df_tsp.to_csv("tsp_results.csv", index=False)
