"""Верифікація імітаційної моделі.

1) Порівняння з аналітичною моделлю M/M/5/m (стаціонарний режим).
2) Стабільність показників при різній тривалості моделювання.
3) Чутливість до інтенсивності вхідного потоку.
4) Чутливість до часу обслуговування.
5) Граничні режими (дуже малий і дуже великий потік).
"""
import numpy as np
import pandas as pd

from analytics import mmnm
from config import Params
from model import simulate

KEYS = ["arrived", "served", "refused", "p_refuse", "Wq_terminal", "Lq_terminal",
        "Wq", "Lq", "Lq_max", "W", "utilization"]


def mean_runs(params_list, warmup=0.0):
    """Середнє значення показників за кількома прогонами."""
    res = [simulate(p, warmup) for p in params_list]
    return {k: float(np.mean([r[k] for r in res])) for k in KEYS}


CASES = [  # (середній інтервал між приходами, місць у залі)
    (5.0, 30), (4.0, 30), (3.5, 30), (3.0, 30), (4.0, 0),
]


def compare_with_analytics(cases=CASES, runs=10, sim_time=50000.0, warmup=2000.0):
    """Стаціонарний режим: довгий прогін, перші warmup хв відкидаються.
    Термінал вимкнено, щоб структура моделі збігалася з M/M/5/m."""
    rows = []
    for am, m in cases:
        base = Params(arrival_mean=am, use_terminal=False, sim_time=sim_time,
                      queue_capacity=m)
        an = mmnm(base.lam, base.mu, base.n_cashiers, base.queue_capacity)
        sim = mean_runs([Params(**{**base.__dict__, "seed": 100 + s})
                         for s in range(runs)], warmup)
        sim["A"] = sim["served"] / sim_time
        for key, name in [("p_refuse", "Ймовірність відмови"),
                          ("Lq", "Середня довжина черги"),
                          ("Wq", "Середній час очікування, хв"),
                          ("W", "Середній час перебування, хв"),
                          ("utilization", "Коефіцієнт завантаження кас"),
                          ("A", "Абсолютна пропускна здатність, кл/хв")]:
            a_val, s_val = an[key], sim[key]
            err = abs(s_val - a_val) / a_val * 100 if a_val > 1e-3 else float("nan")
            rows.append({"Середній інтервал, хв": am, "λ, кл/хв": round(base.lam, 4),
                         "Місць у залі": m,
                         "Показник": name, "Аналітика": a_val,
                         "Імітація": s_val, "Похибка, %": err})
    return pd.DataFrame(rows)


def by_sim_time(times=(480, 960, 1440, 2400, 4800), runs=5):
    rows = []
    for t in times:
        r = mean_runs([Params(sim_time=t, seed=200 + s) for s in range(runs)])
        rows.append({"Час моделювання, хв": t, **r})
    return pd.DataFrame(rows)


def by_param(name, values, runs=10, **fixed):
    rows = []
    for v in values:
        r = mean_runs([Params(**{**fixed, name: v, "seed": 300 + s})
                       for s in range(runs)])
        rows.append({name: v, **r})
    return pd.DataFrame(rows)


def boundary_cases(runs=10):
    cases = {
        "Дуже рідкий потік (інтервал 60 хв)": dict(arrival_mean=60.0),
        "Базовий режим (інтервал 4 хв)": dict(arrival_mean=4.0),
        "Перевантаження (інтервал 1 хв)": dict(arrival_mean=1.0),
        "Одна каса, інтервал 4 хв": dict(n_cashiers=1),
        "Зала на 0 місць (без черги)": dict(queue_capacity=0),
    }
    rows = []
    for title, kw in cases.items():
        r = mean_runs([Params(**{**kw, "seed": 400 + s}) for s in range(runs)])
        rows.append({"Режим": title, **r})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(compare_with_analytics().round(4).to_string(index=False))
    print(by_sim_time().round(3).to_string(index=False))
    print(boundary_cases().round(3).to_string(index=False))
