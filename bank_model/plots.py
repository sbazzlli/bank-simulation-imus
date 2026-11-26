"""Графіки для пояснювальної записки (зберігаються у папку figures/)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import Params
from model import simulate

FIG = Path(__file__).parent / "figures"
FIG.mkdir(exist_ok=True)

C1, C2 = "#257de0", "#f47230"     # синій та помаранчевий
TEXT, GRID = "#333333", "#e3e3e3"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#9c9c9c", "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def _save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)
    return FIG / name


def queue_dynamics(seed=1):
    """Рис.: зміна довжини черги до кас протягом одного робочого дня."""
    _, m = simulate(Params(seed=seed), keep_model=True)
    t, q = zip(*m.q_hall.history)
    t = list(t) + [m.p.sim_time]
    q = list(q) + [q[-1]]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.step(t, q, where="post", color=C1)
    ax.set_xlabel("Модельний час, хв")
    ax.set_ylabel("Клієнтів у черзі до кас")
    ax.set_xlim(0, m.p.sim_time)
    ax.set_ylim(bottom=0)
    return _save(fig, "queue_dynamics.png")


def wait_histogram(days=20):
    """Рис.: гістограма часу очікування в черзі до кас (базовий режим)."""
    waits = []
    for d in range(days):
        _, m = simulate(Params(seed=500 + d), keep_model=True)
        waits += m.waits_queue
    waits = np.array(waits)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(waits, bins=np.arange(0, waits.max() + 2, 2), color=C1,
            edgecolor="white", linewidth=1)
    ax.axvline(waits.mean(), color=TEXT, linestyle="--", linewidth=1.5)
    ax.annotate(f"середнє {waits.mean():.1f} хв", (waits.mean(), ax.get_ylim()[1] * 0.9),
                xytext=(6, 0), textcoords="offset points", color=TEXT)
    ax.set_xlabel("Час очікування в черзі до кас, хв")
    ax.set_ylabel("Кількість клієнтів")
    return _save(fig, "wait_histogram.png"), waits


def desk_utilization(days=20):
    """Рис.: коефіцієнт завантаження кожної каси (середнє за days днів)."""
    util = np.zeros(5)
    for d in range(days):
        r = simulate(Params(seed=600 + d))
        util += np.array([r["util_by_desk"][k] for k in range(1, 6)])
    util /= days
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar([f"Каса {k}" for k in range(1, 6)], util, color=C1, width=0.6)
    for b, u in zip(bars, util):
        ax.text(b.get_x() + b.get_width() / 2, u + 0.01, f"{u:.2f}",
                ha="center", va="bottom", color=TEXT)
    ax.set_ylabel("Коефіцієнт завантаження")
    ax.set_ylim(0, 1)
    ax.grid(axis="x", visible=False)
    return _save(fig, "desk_utilization.png"), util


def sensitivity_arrivals(intervals=(2.5, 3, 3.5, 4, 5, 6, 8), runs=10):
    """Рис.: середній час очікування та частка відмов залежно від
    середнього інтервалу між приходами, для 4 і 5 кас."""
    out = {}
    for n in (4, 5):
        wq, pr = [], []
        for am in intervals:
            rs = [simulate(Params(arrival_mean=am, n_cashiers=n, seed=700 + s))
                  for s in range(runs)]
            wq.append(np.mean([r["Wq"] for r in rs]))
            pr.append(np.mean([r["p_refuse"] for r in rs]) * 100)
        out[n] = (wq, pr)

    figs = []
    for idx, (ylabel, fname) in enumerate([
            ("Середній час очікування, хв", "sens_wait.png"),
            ("Частка відмов, %", "sens_refuse.png")]):
        fig, ax = plt.subplots(figsize=(8, 3.8))
        for n, color, marker in ((4, C2, "s"), (5, C1, "o")):
            ax.plot(intervals, out[n][idx], color=color, marker=marker,
                    markersize=7, label=f"{n} каси" if n == 4 else f"{n} кас")
        ax.set_xlabel("Середній інтервал між приходами клієнтів, хв")
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False)
        figs.append(_save(fig, fname))
    return figs, out


def experiment_bars(y_mean):
    """Рис.: середня кількість обслужених клієнтів за день у 4 дослідах ПФЕ."""
    labels = ["4 каси\n12 хв", "5 кас\n12 хв", "4 каси\n18 хв", "5 кас\n18 хв"]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(labels, y_mean, color=C1, width=0.6)
    for b, v in zip(bars, y_mean):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                ha="center", va="bottom", color=TEXT)
    ax.set_ylabel("Обслужено клієнтів за день")
    ax.set_ylim(0, max(y_mean) * 1.15)
    ax.grid(axis="x", visible=False)
    return _save(fig, "experiment_bars.png")
