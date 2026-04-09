# Лабораторная работа №1
**Решение задачи о рюкзаке при помощи генетического алгоритма**

Вариант 4: n=24, amax=2^(24/1.4)=144715, без модуля

## Запуск

```bash
pip install numpy
gcc -O2 -o brute_force brute_force.c
python3 main.py
```

## Файлы

- `config.py` — параметры варианта
- `generator.py` — генерация рюкзачных векторов и задач
- `genetic.py` — генетический алгоритм
- `brute_force.c` — полный перебор (C, коды Грея)
- `main.py` — запуск экспериментов
- `plots.py` — построение графиков (`pip install matplotlib`)
