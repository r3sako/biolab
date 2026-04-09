"""Генератор рюкзачных векторов и экземпляров задачи о рюкзаке."""
import random
from config import N, AMAX, USE_MODULO, ITEM_FRACTION_RANGE


def generate_knapsack_vector():
    """Генерирует рюкзачный вектор длины N с весами из [1, AMAX]."""
    weights = [random.randint(1, AMAX) for _ in range(N)]
    if USE_MODULO:
        weights = [w % (AMAX + 1) for w in weights]
    return weights


def generate_problem_instance(weights):
    """Создаёт экземпляр задачи о рюкзаке.

    Выбирает случайную долю предметов (от 0.1 до 0.5),
    суммирует их веса — это и есть целевой вес.
    Гарантирует, что у задачи есть хотя бы одно решение.
    """
    n = len(weights)
    fraction = random.uniform(*ITEM_FRACTION_RANGE)
    num_items = max(1, int(round(fraction * n)))
    selected = random.sample(range(n), num_items)
    target = sum(weights[i] for i in selected)
    return {
        "target": target,
        "fraction": fraction,
        "num_selected": num_items,
        "selected_items": sorted(selected),
    }


def generate_all_data(num_vectors, tasks_per_vector):
    """Генерирует все рюкзачные векторы и экземпляры задач."""
    vectors = []
    tasks = []
    task_id = 0

    for vec_id in range(num_vectors):
        weights = generate_knapsack_vector()
        vectors.append({"id": vec_id + 1, "weights": weights})

        for _ in range(tasks_per_vector):
            instance = generate_problem_instance(weights)
            task_id += 1
            tasks.append({
                "task_id": task_id,
                "vector_id": vec_id + 1,
                "weights": weights,
                **instance,
            })

    return vectors, tasks
