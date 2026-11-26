"""Запуск усіх етапів: базовий прогін, верифікація, факторний експеримент, графіки.

Результати:
    results/results.xlsx  - усі таблиці (кожна на окремому аркуші)
    figures/*.png         - рисунки для пояснювальної записки
"""
import sys
from pathlib import Path

import pandas as pd

import experiment
import plots
import verification
from analytics import mmnm
from config import Params
from model import simulate

OUT = Path(__file__).parent / "results"

LABELS = {
    "arrived": "Прийшло клієнтів",
    "served": "Обслужено клієнтів",
    "refused": "Відмов (немає місць у залі)",
    "Wq_terminal": "Сер. очікування біля терміналу, хв",
    "Wq": "Сер. час очікування в черзі, хв",
    "Wq_max": "Макс. час очікування, хв",
    "Lq": "Сер. довжина черги",
    "Lq_max": "Макс. довжина черги",
    "W": "Сер. час перебування у філії, хв",
    "utilization": "Коеф. завантаження кас",
    "util_by_desk": "Завантаження кас 1-5",
    "served_by_desk": "Обслужено касами 1-5",
}
OUT.mkdir(exist_ok=True)


class Tee:
    """Дублює вивід програми в консоль і у файл results/main_output.txt (UTF-8)."""

    def __init__(self, console, path):
        self.console = console
        self.file = open(path, "w", encoding="utf-8-sig")  # з BOM: коректно відкривається в Блокноті

    def write(self, text):
        self.file.write(text)
        try:
            self.console.write(text)
        except UnicodeEncodeError:  # консоль без підтримки символу (напр. λ у cp1251)
            enc = self.console.encoding or "ascii"
            self.console.write(text.encode(enc, "replace").decode(enc))

    def flush(self):
        self.file.flush()
        self.console.flush()


def main():
    tables = {}

    # 1. Базовий прогін: один робочий день
    p = Params()
    r = simulate(p)
    print("=== Базовий прогін (1 робочий день, seed = 1) ===")
    for k, name in LABELS.items():
        v = r[k]
        if isinstance(v, dict):
            v = "  ".join(f"{d}: {x:.2f}" if isinstance(x, float) else f"{d}: {x}"
                          for d, x in v.items())
        elif isinstance(v, float):
            v = f"{v:.3f}"
        print(f"  {name:38s} {v}")
    tables["1_Базовий прогін"] = pd.DataFrame(
        [(k, v) for k, v in r.items() if not isinstance(v, dict)],
        columns=["Показник", "Значення"])

    # 2. Верифікація
    print("\n=== Верифікація: аналітика vs імітація ===")
    t = verification.compare_with_analytics()
    for (ia, m), g in t.groupby(["Середній інтервал, хв", "Місць у залі"], sort=False):
        print(f"--- 1/λ = {ia} хв, місць у залі: {m}")
        print(g[["Показник", "Аналітика", "Імітація", "Похибка, %"]]
              .round(4).to_string(index=False))
    tables["2_Аналітика vs імітація"] = t
    tables["3_Час моделювання"] = verification.by_sim_time()
    tables["4_Інтенсивність потоку"] = verification.by_param(
        "arrival_mean", [2.5, 3, 3.5, 4, 5, 6, 8])
    tables["5_Час обслуговування"] = verification.by_param(
        "service_mean", [8, 10, 12, 15, 18, 20, 22])
    tables["6_Граничні режими"] = verification.boundary_cases()

    # 3. Факторний експеримент
    print("\n=== Повний факторний експеримент 2^2 ===")
    df = experiment.run_experiment()
    a = experiment.analyse(df)
    print(df.to_string(index=False))
    experiment.print_report(a)
    tables["7_ПФЕ прогони"] = df
    tables["8_ПФЕ обробка"] = pd.DataFrame({
        "Коефіцієнт": experiment.COEF_NAMES, "Значення": a["b"],
        "t": a["t"], "Значущий": a["significant"]})

    # 4. Графіки
    plots.queue_dynamics()
    plots.wait_histogram()
    plots.desk_utilization()
    plots.sensitivity_arrivals()
    plots.experiment_bars(a["y_mean"])

    with pd.ExcelWriter(OUT / "results.xlsx") as xw:
        for name, tbl in tables.items():
            tbl.to_excel(xw, sheet_name=name[:31], index=False)
    print(f"\nТаблиці: {OUT / 'results.xlsx'}\nРисунки: {plots.FIG}")


if __name__ == "__main__":
    sys.stdout = Tee(sys.stdout, OUT / "main_output.txt")
    main()
    sys.stdout.flush()
