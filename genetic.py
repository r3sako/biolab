"""Генетический алгоритм для решения задачи о рюкзаке (KP01)."""
import random
import time
from config import (
    GA_POPULATION_SIZE,
    GA_CROSSOVER_PROB,
    GA_MUTATION_PROB,
    GA_TOURNAMENT_SIZE,
    GA_ELITISM,
)


def fitness(chromosome, weights, target):
    """Фитнесс-функция: |target - сумма выбранных весов|."""
    s = sum(w for w, bit in zip(weights, chromosome) if bit)
    return abs(target - s)


def create_individual(n):
    """Создаёт случайную хромосому длины n."""
    return [random.randint(0, 1) for _ in range(n)]


def tournament_selection(population, fitnesses, k=GA_TOURNAMENT_SIZE):
    """Турнирная селекция."""
    indices = random.sample(range(len(population)), k)
    best = min(indices, key=lambda i: fitnesses[i])
    return population[best][:]


def single_point_crossover(parent1, parent2):
    """Одноточечный кроссовер."""
    if random.random() > GA_CROSSOVER_PROB:
        return parent1[:], parent2[:]
    point = random.randint(1, len(parent1) - 1)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def mutate(chromosome):
    """Побитовая мутация с вероятностью GA_MUTATION_PROB на каждый бит."""
    return [
        (1 - bit) if random.random() < GA_MUTATION_PROB else bit
        for bit in chromosome
    ]


def solve_ga(weights, target, max_time):
    """Решение задачи о рюкзаке генетическим алгоритмом.

    Условия остановки:
      1) fitness = 0 (точное решение)
      2) нет улучшения за 2 последних поколения
      3) время работы >= 2 * время полного перебора

    Args:
        weights: список весов предметов
        target: целевой вес
        max_time: максимальное время работы (2 * время перебора)

    Returns:
        dict с результатами работы алгоритма
    """
    n = len(weights)
    pop_size = GA_POPULATION_SIZE
    time_limit = max_time

    # Инициализация популяции
    population = [create_individual(n) for _ in range(pop_size)]
    fitnesses = [fitness(ind, weights, target) for ind in population]

    best_fitness = min(fitnesses)
    best_idx = fitnesses.index(best_fitness)
    best_individual = population[best_idx][:]

    start_time = time.time()
    generation = 0
    stop_reason = ""
    fitness_history = [best_fitness]

    while True:
        # Проверка условий остановки
        if best_fitness == 0:
            stop_reason = "Точное решение (fitness=0)"
            break

        elapsed = time.time() - start_time
        if elapsed >= time_limit:
            stop_reason = "Превышение времени"
            break

        if len(fitness_history) >= 3:
            if (fitness_history[-1] == fitness_history[-2] ==
                    fitness_history[-3]):
                stop_reason = "Нет улучшения (2 поколения)"
                break

        # Формирование нового поколения
        generation += 1
        new_population = []

        # Элитизм: лучшие особи переходят без изменений
        sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i])
        for i in range(GA_ELITISM):
            new_population.append(population[sorted_indices[i]][:])

        # Создание потомков
        while len(new_population) < pop_size:
            p1 = tournament_selection(population, fitnesses)
            p2 = tournament_selection(population, fitnesses)
            c1, c2 = single_point_crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population
        fitnesses = [fitness(ind, weights, target) for ind in population]

        gen_best = min(fitnesses)
        if gen_best < best_fitness:
            best_fitness = gen_best
            best_idx = fitnesses.index(gen_best)
            best_individual = population[best_idx][:]

        fitness_history.append(best_fitness)

    elapsed = time.time() - start_time

    return {
        "time": elapsed,
        "best_fitness": best_fitness,
        "stop_reason": stop_reason,
        "last_generation": generation,
        "best_individual": best_individual,
        "exact_solution": best_fitness == 0,
    }
