import json
import subprocess
import time
import numpy as np
from config import N, AMAX, BRUTE_FORCE_BIN
from generator import generate_all_data
from genetic import solve_ga, GA_POPULATION_SIZE

# Значения элитизма для исследования
ELITISM_VALUES = [0, 1, 2, 4, 8, 16, 32]

# Генерируем данные (меньше задач — быстрее, но достаточно для статистики)
NUM_VECTORS = 30
TASKS_PER_VECTOR = 10

import genetic  # чтобы подменять GA_ELITISM


def solve_brute_force(weights, target):
    args = [BRUTE_FORCE_BIN, str(N), str(target)] + [str(w) for w in weights]
    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
    parts = result.stdout.strip().split()
    return {
        "first_time": float(parts[0]),
        "total_time": float(parts[1]),
        "num_solutions": int(parts[2]),
    }


def main():
    print(f"Исследование элитизма: n={N}, amax={AMAX}")
    print(f"Значения элитизма: {ELITISM_VALUES}")
    print(f"Генерация {NUM_VECTORS} векторов × {TASKS_PER_VECTOR} задач...")

    vectors, tasks = generate_all_data(NUM_VECTORS, TASKS_PER_VECTOR)
    total_tasks = len(tasks)
    print(f"Всего задач: {total_tasks}")

    # Сначала решаем полным перебором (один раз для всех)
    print("\nПолный перебор...")
    bf_results = []
    for i, task in enumerate(tasks):
        if (i + 1) % 50 == 0:
            print(f"  Задача {i+1}/{total_tasks}...")
        bf = solve_brute_force(task["weights"], task["target"])
        bf_results.append(bf)

    # Теперь прогоняем ГА с разными значениями элитизма
    all_results = {}

    for elitism in ELITISM_VALUES:
        print(f"\n--- Элитизм = {elitism} ---")
        genetic.GA_ELITISM = elitism  # подменяем параметр

        ga_results = []
        for i, task in enumerate(tasks):
            if (i + 1) % 100 == 0:
                print(f"  Задача {i+1}/{total_tasks}...")

            max_time = max(2.0 * bf_results[i]["total_time"], 0.5)
            ga = solve_ga(task["weights"], task["target"], max_time)
            ga_results.append(ga)

        exact_count = sum(1 for r in ga_results if r["exact_solution"])
        exact_fraction = exact_count / total_tasks
        exact_times = [r["time"] for r in ga_results if r["exact_solution"]]
        all_times = [r["time"] for r in ga_results]
        all_fitnesses = [r["best_fitness"] for r in ga_results]
        all_generations = [r["last_generation"] for r in ga_results]

        stats = {
            "elitism": elitism,
            "exact_count": exact_count,
            "exact_fraction": exact_fraction,
            "mean_time_all": float(np.mean(all_times)),
            "mean_time_exact": float(np.mean(exact_times)) if exact_times else None,
            "mean_fitness": float(np.mean(all_fitnesses)),
            "median_fitness": float(np.median(all_fitnesses)),
            "mean_generations": float(np.mean(all_generations)),
        }
        all_results[elitism] = stats

        print(f"  Точных решений: {exact_count}/{total_tasks} ({exact_fraction*100:.1f}%)")
        print(f"  Среднее время (все): {stats['mean_time_all']:.6f} с")
        if exact_times:
            print(f"  Среднее время (точные): {stats['mean_time_exact']:.6f} с")
        print(f"  Средний фитнесс: {stats['mean_fitness']:.1f}")
        print(f"  Медианный фитнесс: {stats['median_fitness']:.1f}")
        print(f"  Среднее число поколений: {stats['mean_generations']:.1f}")

    # Сохраняем результаты
    output = {
        "params": {"n": N, "amax": AMAX, "population_size": GA_POPULATION_SIZE,
                    "total_tasks": total_tasks},
        "elitism_values": ELITISM_VALUES,
        "results": {str(k): v for k, v in all_results.items()},
    }
    with open("elitism_results.json", "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Итоговая таблица
    print("\n" + "=" * 80)
    print(f"{'Элитизм':>10} | {'Точных':>8} | {'Доля,%':>8} | {'Ср.время,с':>12} | {'Ср.фитнесс':>12} | {'Ср.покол.':>10}")
    print("-" * 80)
    for e in ELITISM_VALUES:
        s = all_results[e]
        print(f"{e:>10} | {s['exact_count']:>8} | {s['exact_fraction']*100:>7.1f}% | {s['mean_time_all']:>12.6f} | {s['mean_fitness']:>12.1f} | {s['mean_generations']:>10.1f}")

    print("\nРезультаты сохранены в elitism_results.json")


if __name__ == "__main__":
    main()