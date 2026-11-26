"""Аналітичні формули багатоканальної СМО M/M/n/m
(n каналів, m місць у черзі, найпростіший вхідний потік, експоненційне обслуговування).
Використовуються для верифікації імітаційної моделі.
"""
from math import factorial


def mmnm(lam: float, mu: float, n: int, m: int) -> dict:
    rho = lam / mu            # зведена інтенсивність потоку
    chi = rho / n             # навантаження на один канал

    # Ймовірність того, що всі канали вільні
    s1 = sum(rho ** k / factorial(k) for k in range(n + 1))
    s2 = rho ** n / factorial(n) * sum(chi ** s for s in range(1, m + 1))
    p0 = 1.0 / (s1 + s2)

    # Ймовірність відмови: усі канали зайняті і всі m місць у черзі теж
    p_refuse = rho ** n / factorial(n) * chi ** m * p0
    q = 1.0 - p_refuse                    # відносна пропускна здатність
    a = lam * q                           # абсолютна пропускна здатність

    # Середня довжина черги
    lq = rho ** n / factorial(n) * p0 * sum(s * chi ** s for s in range(1, m + 1))
    k_busy = a / mu                       # середня кількість зайнятих кас
    wq = lq / a                           # середній час очікування (формула Літтла)
    w = wq + 1.0 / mu                     # середній час перебування в системі

    return {
        "p0": p0,
        "p_refuse": p_refuse,
        "Q": q,
        "A": a,
        "Lq": lq,
        "k_busy": k_busy,
        "utilization": k_busy / n,
        "Wq": wq,
        "W": w,
    }


if __name__ == "__main__":
    for k, v in mmnm(0.25, 1 / 15, 5, 30).items():
        print(f"{k:12s} {v:.5f}")
