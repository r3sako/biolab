"""Главный скрипт: генерация данных, решение задач, сбор результатов."""
import json
import subprocess
import time
import sys
from config import (
    N, AMAX, NUM_VECTORS, TASKS_PER_VECTOR, BRUTE_FORCE_BIN,
    GA_POPULATION_SIZE, VARIANT,
)
from generator import generate_all_data
from genetic import solve_ga


def solve_brute_force(weights, target):
    """Запуск C-солвера полного перебора."""
    args = [BRUTE_FORCE_BIN, str(N), str(target)] + [str(w) for w in weights]
    result = subprocess.run(args, capture_output=True, text=True, timeout=300)
    parts = result.stdout.strip().split()
    return {
        "first_time": float(parts[0]),
        "total_time": float(parts[1]),
        "num_solutions": int(parts[2]),
    }


def main():
    print(f"Вариант {VARIANT}: n={N}, amax={AMAX}")
    print(f"Генерация {NUM_VECTORS} векторов × {TASKS_PER_VECTOR} задач...")

    vectors, tasks = generate_all_data(NUM_VECTORS, TASKS_PER_VECTOR)
    total_tasks = len(tasks)

    results_bf = []
    results_ga = []

    print(f"Всего задач: {total_tasks}")
    print("Запуск решений...")

    for i, task in enumerate(tasks):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Задача {i + 1}/{total_tasks}...")

        # Полный перебор
        bf = solve_brute_force(task["weights"], task["target"])
        results_bf.append({
            "task_id": task["task_id"],
            "vector_id": task["vector_id"],
            "target": task["target"],
            "fraction": task["fraction"],
            **bf,
        })

        # Генетический алгоритм (макс. время = 2 × время перебора)
        max_time = 2.0 * bf["total_time"]
        # Минимум 0.5 сек чтобы ГА имел шанс
        max_time = max(max_time, 0.5)

        ga = solve_ga(task["weights"], task["target"], max_time)
        results_ga.append({
            "task_id": task["task_id"],
            "vector_id": task["vector_id"],
            "target": task["target"],
            "fraction": task["fraction"],
            **ga,
            "best_individual": None,  # Не сохраняем для JSON
        })

    # Статистическая обработка
    import numpy as np

    bf_first_times = [r["first_time"] for r in results_bf]
    bf_total_times = [r["total_time"] for r in results_bf]

    ga_exact = [r for r in results_ga if r["exact_solution"]]
    ga_exact_times = [r["time"] for r in ga_exact]
    ga_exact_fraction = len(ga_exact) / total_tasks

    stats = {
        "variant": VARIANT,
        "n": N,
        "amax": AMAX,
        "total_tasks": total_tasks,
        "population_size": GA_POPULATION_SIZE,
        "n_stats": {"mean": float(N), "var": 0.0, "std": 0.0},
        "amax_stats": {"mean": float(AMAX), "var": 0.0, "std": 0.0},
        "bf_first_time": {
            "mean": float(np.mean(bf_first_times)),
            "var": float(np.var(bf_first_times)),
            "std": float(np.std(bf_first_times)),
        },
        "bf_total_time": {
            "mean": float(np.mean(bf_total_times)),
            "var": float(np.var(bf_total_times)),
            "std": float(np.std(bf_total_times)),
        },
        "ga_exact_time": {
            "mean": float(np.mean(ga_exact_times)) if ga_exact_times else None,
            "var": float(np.var(ga_exact_times)) if ga_exact_times else None,
            "std": float(np.std(ga_exact_times)) if ga_exact_times else None,
        },
        "ga_exact_fraction": ga_exact_fraction,
    }

    # Сохранение результатов
    output = {
        "vectors": vectors[:10],  # Первые 10 для отчёта
        "tasks_sample": [
            {k: v for k, v in t.items() if k != "weights"}
            for t in tasks[:20]
        ],
        "results_bf": results_bf,
        "results_ga": [
            {k: v for k, v in r.items() if k != "best_individual"}
            for r in results_ga
        ],
        "stats": stats,
    }

    with open("results.json", "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n=== Статистика ===")
    print(f"n = {N}, amax = {AMAX}")
    print(f"Среднее время первого решения (перебор): {stats['bf_first_time']['mean']:.6f} с")
    print(f"Среднее время всех решений (перебор):    {stats['bf_total_time']['mean']:.6f} с")
    if ga_exact_times:
        print(f"Среднее время точного решения (ГА):      {stats['ga_exact_time']['mean']:.6f} с")
    else:
        print("Генетический алгоритм не нашёл точных решений")
    print(f"Доля точно решённых ГА:                  {ga_exact_fraction:.4f} ({len(ga_exact)}/{total_tasks})")
    print(f"Размер популяции:                        {GA_POPULATION_SIZE}")
    print("\nРезультаты сохранены в results.json")


if __name__ == "__main__":
    main()
