"""Повний факторний експеримент 2^2 і статистична обробка результатів.

Відгук y: середня кількість клієнтів, обслужених за робочий день
(пропускна здатність філії, клієнтів/день).
Фактори:
    x1: кількість відкритих кас            4 ... 5
    x2: середній час обслуговування, хв    12 ... 18
Кожен дослід повторюється p разів з різними зернами генератора.
"""
import numpy as np
import pandas as pd
from scipy import stats

from config import Params
from model import simulate

ALPHA = 0.05
P_RUNS = 4

FACTORS = {
    "x1": ("Кількість кас", 4, 5),
    "x2": ("Середній час обслуговування, хв", 12.0, 18.0),
}

# Матриця планування ПФЕ 2^2 у кодованих змінних
PLAN = np.array([
    # x0  x1  x2  x1x2
    [+1, -1, -1, +1],
    [+1, +1, -1, -1],
    [+1, -1, +1, -1],
    [+1, +1, +1, +1],
])
COEF_NAMES = ["b0", "b1", "b2", "b12"]


def decode(coded, low, high):
    """Перехід від кодованого значення (-1/+1) до натурального."""
    center, delta = (low + high) / 2, (high - low) / 2
    return center + coded * delta


DAYS = 5   # один прогін = робочий тиждень (5 незалежних робочих днів по 480 хв)


def run_experiment(base: Params = None, p_runs: int = P_RUNS, days: int = DAYS,
                   seed_base: int = 0):
    """Проводить N*p прогонів. Один прогін = days робочих днів; кожен день філія
    відкривається порожньою. Відгук прогону - середня кількість обслужених
    клієнтів за день. Додатково зберігаються середній час очікування та частка відмов."""
    base = base or Params()
    rows = []
    for j, (_, x1, x2, _) in enumerate(PLAN, start=1):
        n = int(decode(x1, *FACTORS["x1"][1:]))
        s = decode(x2, *FACTORS["x2"][1:])
        for i in range(1, p_runs + 1):
            seed = seed_base + 1000 * j + 10 * i
            day_res = []
            for day in range(days):
                prm = Params(**{**base.__dict__, "n_cashiers": n,
                                "service_mean": s, "seed": seed + day})
                day_res.append(simulate(prm))
            rows.append({"експ": j, "прогін": i, "x1": x1, "x2": x2,
                         "кас": n, "обсл, хв": s, "seed": seed,
                         "y": float(np.mean([r["served"] for r in day_res])),
                         "Wq": float(np.mean([r["Wq"] for r in day_res])),
                         "p_refuse": float(np.mean([r["p_refuse"] for r in day_res]))})
    return pd.DataFrame(rows)


def analyse(df: pd.DataFrame, alpha: float = ALPHA) -> dict:
    N, d = PLAN.shape[0], PLAN.shape[1]
    Y = df.pivot(index="експ", columns="прогін", values="y").to_numpy()
    p = Y.shape[1]

    y_mean = Y.mean(axis=1)
    D = Y.var(axis=1, ddof=1)                 # дисперсії дослідів

    # 1. Критерій Кохрена: відтворюваність
    G = D.max() / D.sum()
    f1, f2 = p - 1, N
    F_c = stats.f.ppf(1 - alpha / f2, f1, (f2 - 1) * f1)
    G_cr = F_c / (F_c + f2 - 1)
    reproducible = G < G_cr
    D_rep = D.mean()                          # дисперсія відтворюваності

    # 2. Коефіцієнти регресії
    b = PLAN.T @ y_mean / N

    # 3. Критерій Стьюдента: значущість коефіцієнтів
    f_rep = N * (p - 1)
    s_b = np.sqrt(D_rep / (N * p))
    t = np.abs(b) / s_b
    t_cr = stats.t.ppf(1 - alpha / 2, f_rep)
    significant = t > t_cr

    # 4. Критерій Фішера: адекватність (за незначущих коефіцієнтів)
    b_sig = np.where(significant, b, 0.0)
    y_hat = PLAN @ b_sig
    d_sig = int(significant.sum())
    fisher = None
    if N > d_sig:
        D_ad = p / (N - d_sig) * np.sum((y_mean - y_hat) ** 2)
        F = D_ad / D_rep
        F_cr = stats.f.ppf(1 - alpha, N - d_sig, f_rep)
        fisher = {"D_ad": D_ad, "F": F, "F_cr": F_cr, "adequate": F < F_cr,
                  "f_ad": N - d_sig, "f_rep": f_rep}

    return {
        "Y": Y, "y_mean": y_mean, "D": D,
        "G": G, "G_cr": G_cr, "reproducible": reproducible, "D_rep": D_rep,
        "b": b, "s_b": s_b, "t": t, "t_cr": t_cr, "significant": significant,
        "y_hat": y_hat, "fisher": fisher, "N": N, "p": p, "d": d,
    }


def print_report(a: dict):
    print("Середні відгуки y_j:", np.round(a["y_mean"], 3))
    print("Дисперсії D_j:     ", np.round(a["D"], 4))
    print(f"Кохрен: G = {a['G']:.4f}, G_кр = {a['G_cr']:.4f} ->",
          "відтворюваний" if a["reproducible"] else "НЕ відтворюваний")
    for name, b, t, sig in zip(COEF_NAMES, a["b"], a["t"], a["significant"]):
        print(f"  {name:4s} = {b:9.4f}   t = {t:8.3f}   {'значущий' if sig else 'незначущий'}")
    print(f"t_кр = {a['t_cr']:.3f}")
    f = a["fisher"]
    if f:
        print(f"Фішер: F = {f['F']:.4f}, F_кр = {f['F_cr']:.4f} ->",
              "адекватне" if f["adequate"] else "НЕ адекватне")
    else:
        print("Фішер: усі коефіцієнти значущі, рівняння збігається з усіма точками плану "
              "(N = d), перевірка адекватності не потрібна.")


if __name__ == "__main__":
    df = run_experiment()
    print(df.to_string(index=False))
    print_report(analyse(df))
