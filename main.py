import numpy as np
import pandas as pd
import random
import time

# ==============================
# FUNKCJE TESTOWE
# ==============================

def rastrigin(x):
    """
    Funkcja Rastrigina (minimum globalne = 0 w x=0)
    Trudna przez dużą liczbę minimów lokalnych
    """
    A = 10
    return A * len(x) + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])


def schwefel(x):
    """
    Funkcja Schwefela
    Minimum globalne: ~0 w okolicach 420.9687
    """
    return 418.9829 * len(x) - sum([xi * np.sin(np.sqrt(abs(xi))) for xi in x])


def eggholder(x):
    """
    Funkcja Eggholder (2D)
    Bardzo nieregularna powierzchnia
    """
    x1, x2 = x
    return -(x2 + 47) * np.sin(np.sqrt(abs(x2 + x1/2 + 47))) - x1 * np.sin(np.sqrt(abs(x1 - (x2 + 47))))


# ==============================
# PSO – Particle Swarm Optimization
# ==============================

def pso(func, dim, bounds, num_particles=30, max_iter=50):
    """
    Klasyczny PSO dla funkcji ciągłych
    """

    # inicjalizacja
    particles = np.random.uniform(bounds[0], bounds[1], (num_particles, dim))
    velocities = np.zeros((num_particles, dim))

    pbest = particles.copy()
    pbest_val = np.array([func(p) for p in particles])

    gbest = pbest[np.argmin(pbest_val)]

    w = 0.7
    c1 = 1.5
    c2 = 1.5

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
# ABC – Artificial Bee Colony
# ==============================

def abc(func, dim, bounds, num_bees=30, max_iter=50):
    """
    Algorytm sztucznej kolonii pszczół
    """

    solutions = np.random.uniform(bounds[0], bounds[1], (num_bees, dim))
    fitness = np.array([func(s) for s in solutions])

    history = []

    for _ in range(max_iter):
        # employed bees
        for i in range(num_bees):
            k = random.randint(0, num_bees - 1)
            phi = np.random.uniform(-1, 1, dim)

            new = solutions[i] + phi * (solutions[i] - solutions[k])
            new = np.clip(new, bounds[0], bounds[1])

            if func(new) < fitness[i]:
                solutions[i] = new
                fitness[i] = func(new)

        # onlooker bees
        prob = fitness / np.sum(fitness)
        for i in range(num_bees):
            if random.random() < prob[i]:
                k = random.randint(0, num_bees - 1)
                phi = np.random.uniform(-1, 1, dim)

                new = solutions[i] + phi * (solutions[i] - solutions[k])
                new = np.clip(new, bounds[0], bounds[1])

                if func(new) < fitness[i]:
                    solutions[i] = new
                    fitness[i] = func(new)

        history.append(min(fitness))

    best = solutions[np.argmin(fitness)]
    return best, min(fitness), history


# ==============================
# ACO – Ant Colony Optimization (TSP)
# ==============================

def distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist[i][j] = np.linalg.norm(coords[i] - coords[j])
    return dist


def aco_tsp(coords, num_ants=20, max_iter=50, alpha=1, beta=2, rho=0.5):
    """
    ACO dla problemu TSP
    """

    n = len(coords)
    dist = distance_matrix(coords)

    pheromone = np.ones((n, n))

    best_path = None
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
                        tau = pheromone[current][j] ** alpha
                        eta = (1 / dist[current][j]) ** beta
                        probs.append(tau * eta)
                    else:
                        probs.append(0)

                probs = np.array(probs)
                probs = probs / np.sum(probs)

                next_city = np.random.choice(range(n), p=probs)
                path.append(next_city)

            paths.append(path)

        # aktualizacja feromonów
        pheromone *= (1 - rho)

        for path in paths:
            length = sum(dist[path[i]][path[i+1]] for i in range(n-1))

            if length < best_length:
                best_length = length
                best_path = path

            for i in range(n-1):
                pheromone[path[i]][path[i+1]] += 1 / length

        history.append(best_length)

    return best_path, best_length, history


# ==============================
# PSO DYSKRETNE – TSP
# ==============================

def tsp_distance(path, dist):
    return sum(dist[path[i]][path[i+1]] for i in range(len(path)-1))


def swap(path, i, j):
    new_path = path.copy()
    new_path[i], new_path[j] = new_path[j], new_path[i]
    return new_path


def pso_tsp(coords, num_particles=30, max_iter=50):
    """
    Dyskretne PSO dla problemu TSP (operacje swap jako "prędkość")
    """

    n = len(coords)
    dist = distance_matrix(coords)

    # inicjalizacja populacji (permutacje)
    particles = [list(np.random.permutation(n)) for _ in range(num_particles)]
    pbest = particles.copy()

    pbest_val = [tsp_distance(p, dist) for p in particles]
    gbest = pbest[np.argmin(pbest_val)]

    history = []

    for _ in range(max_iter):
        for i in range(num_particles):
            current = particles[i]

            # kopiujemy aktualną trasę
            new = current.copy()

            # część "indywidualna" – zbliżanie do pbest
            for _ in range(2):
                a, b = random.sample(range(n), 2)
                if new[a] != pbest[i][a]:
                    b = new.index(pbest[i][a])
                    new = swap(new, a, b)

            # część "globalna" – zbliżanie do gbest
            for _ in range(2):
                a, b = random.sample(range(n), 2)
                if new[a] != gbest[a]:
                    b = new.index(gbest[a])
                    new = swap(new, a, b)

            # aktualizacja cząstki
            particles[i] = new
            val = tsp_distance(new, dist)

            if val < pbest_val[i]:
                pbest[i] = new
                pbest_val[i] = val

        gbest = pbest[np.argmin(pbest_val)]
        history.append(min(pbest_val))

    return gbest, min(pbest_val), history


# ==============================
# PRZYKŁADOWE URUCHOMIENIE
# ==============================

# ==============================
# EKSPERYMENTY I TABELE WYNIKÓW
# ==============================


def run_function_experiments():
    results = []

    functions = [
        ("Rastrigin", rastrigin, (-5.12, 5.12)),
        ("Schwefel", schwefel, (-500, 500))
    ]

    dims = [2, 10]
    populations = [20, 50]
    iterations = [20, 50]

    for name, func, bounds in functions:
        for dim in dims:
            for pop in populations:
                for it in iterations:
                    # PSO
                    start = time.time()
                    _, val, _ = pso(func, dim, bounds, pop, it)
                    t = time.time() - start

                    results.append([name, dim, "PSO", pop, it, val, t])

                    # ABC
                    start = time.time()
                    _, val, _ = abc(func, dim, bounds, pop, it)
                    t = time.time() - start

                    results.append([name, dim, "ABC", pop, it, val, t])

    df = pd.DataFrame(results, columns=[
        "Function", "Dim", "Algorithm", "Population", "Iterations", "Best Value", "Time"
    ])

    return df


def run_tsp_experiments():
    results = []

    city_sizes = [10, 30, 50]

    for n in city_sizes:
        coords = np.random.rand(n, 2) * 100

        # ACO
        start = time.time()
        _, val, _ = aco_tsp(coords)
        t = time.time() - start

        results.append([n, "ACO", val, t])

        # PSO dyskretne
        start = time.time()
        _, val, _ = pso_tsp(coords)
        t = time.time() - start

        results.append([n, "PSO_discrete", val, t])

    df = pd.DataFrame(results, columns=[
        "Cities", "Algorithm", "Path Length", "Time"
    ])

    return df


# ==============================
# URUCHOMIENIE EKSPERYMENTÓW
# ==============================

if __name__ == "__main__":
    print("=== EKSPERYMENTY FUNKCJI ===")
    df_func = run_function_experiments()
    print(df_func)

    print("=== EKSPERYMENTY TSP ===")
    df_tsp = run_tsp_experiments()
    print(df_tsp)

    # zapis do plików CSV (do raportu)
    df_func.to_csv("function_results.csv", index=False)
    df_tsp.to_csv("tsp_results.csv", index=False)
