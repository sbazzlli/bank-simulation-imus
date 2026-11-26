"""Імітаційна модель роботи філії банку з п'ятьма касами (бібліотека SimPy).

Маршрут клієнта:
    прихід -> черга до терміналу -> термінал (талон) -> зала очікування (FIFO)
    -> перша вільна каса -> вихід.
Якщо в залі немає вільних місць, клієнт отримує відмову і залишає банк.

Модельний час просувається за принципом особливих станів (від події до події):
SimPy зберігає календар майбутніх подій і щоразу переходить до найближчої.
"""
import numpy as np
import simpy

from config import Params


class TimeAverage:
    """Середнє за часом для величини, що змінюється стрибками (довжина черги)."""

    def __init__(self, env):
        self.env = env
        self.last_t = 0.0
        self.value = 0
        self.area = 0.0
        self.max_value = 0
        self.history = [(0.0, 0)]          # (час, значення) для графіка

    def update(self, new_value):
        now = self.env.now
        self.area += self.value * (now - self.last_t)
        self.last_t = now
        self.value = new_value
        self.max_value = max(self.max_value, new_value)
        self.history.append((now, new_value))

    def mean(self, t_end):
        return (self.area + self.value * (t_end - self.last_t)) / t_end


class BankBranch:
    """Філія банку: термінал талонів, зала очікування та n касах."""

    def __init__(self, p: Params):
        self.p = p
        self.env = simpy.Environment()
        self.rng = np.random.default_rng(p.seed)

        # Ресурси моделі
        self.terminal = simpy.Resource(self.env, capacity=1)
        self.cashiers = simpy.Resource(self.env, capacity=p.n_cashiers)
        self.free_desks = list(range(1, p.n_cashiers + 1))   # номери вільних кас

        # Лічильники
        self.arrived = 0           # прийшло клієнтів
        self.refused = 0           # відмова (немає місць у залі)
        self.served = 0            # обслужено
        self.waits_terminal = []   # час очікування талону
        self.waits_queue = []      # час очікування в черзі до кас
        self.times_in_system = []  # повний час перебування в банку
        self.served_by_desk = {k: 0 for k in self.free_desks}
        self.busy_time = {k: 0.0 for k in self.free_desks}
        self.busy_since = {}       # каса -> момент початку поточного обслуговування

        # Довжини черг, середні за часом
        self.q_terminal = TimeAverage(self.env)
        self.q_hall = TimeAverage(self.env)

    # ---------- генератори випадкових величин ----------
    def interarrival(self):
        return self.rng.exponential(self.p.arrival_mean)

    def terminal_time(self):
        return self.rng.uniform(self.p.terminal_min, self.p.terminal_max)

    def service_time(self):
        return self.rng.exponential(self.p.service_mean)

    # ---------- процеси ----------
    def source(self):
        """Генератор клієнтів."""
        while True:
            yield self.env.timeout(self.interarrival())
            self.arrived += 1
            self.env.process(self.client())

    def client(self):
        """Шлях одного клієнта через філію."""
        t_arrive = self.env.now

        # 1. Отримання талону в терміналі
        if self.p.use_terminal:
            req = self.terminal.request()
            self.q_terminal.update(len(self.terminal.queue))
            yield req
            self.q_terminal.update(len(self.terminal.queue))
            self.waits_terminal.append(self.env.now - t_arrive)
            yield self.env.timeout(self.terminal_time())
            self.terminal.release(req)

        # 2. Зала очікування: перевірка наявності вільного місця
        all_busy = self.cashiers.count >= self.p.n_cashiers
        if all_busy and len(self.cashiers.queue) >= self.p.queue_capacity:
            self.refused += 1
            return

        t_queue = self.env.now
        req = self.cashiers.request()
        self.q_hall.update(len(self.cashiers.queue))
        yield req
        self.q_hall.update(len(self.cashiers.queue))
        self.waits_queue.append(self.env.now - t_queue)

        # 3. Обслуговування на першій вільній касі
        desk = min(self.free_desks)
        self.free_desks.remove(desk)
        self.busy_since[desk] = self.env.now
        yield self.env.timeout(self.service_time())
        self.busy_time[desk] += self.env.now - self.busy_since.pop(desk)
        self.free_desks.append(desk)
        self.cashiers.release(req)

        # 4. Вихід
        self.served += 1
        self.served_by_desk[desk] += 1
        self.times_in_system.append(self.env.now - t_arrive)

    # ---------- запуск ----------
    def run(self, warmup: float = 0.0):
        """Запускає модель. Якщо warmup > 0, статистика перших warmup хв
        відкидається (потрібно для порівняння зі стаціонарною аналітикою)."""
        self.env.process(self.source())
        if warmup > 0:
            self.env.run(until=warmup)
            self._reset_stats()
        self.env.run(until=warmup + self.p.sim_time)
        return self.results()

    def _reset_stats(self):
        now = self.env.now
        self.arrived = self.refused = self.served = 0
        self.waits_terminal, self.waits_queue, self.times_in_system = [], [], []
        self.served_by_desk = {k: 0 for k in self.served_by_desk}
        self.busy_time = {k: 0.0 for k in self.busy_time}
        self.busy_since = {k: now for k in self.busy_since}
        for q in (self.q_terminal, self.q_hall):
            q.area, q.last_t, q.max_value = 0.0, now, q.value
            q.history = [(now, q.value)]
        self._t0 = now

    def results(self) -> dict:
        t_end = self.env.now
        t0 = getattr(self, "_t0", 0.0)
        dur = t_end - t0

        # каси, що працюють у момент завершення моделювання
        busy = dict(self.busy_time)
        for desk, start in self.busy_since.items():
            busy[desk] += t_end - start
        util = {k: v / dur for k, v in busy.items()}

        def avg(x):
            return float(np.mean(x)) if x else 0.0

        def tavg(q):
            return (q.area + q.value * (t_end - q.last_t)) / dur

        return {
            "arrived": self.arrived,
            "refused": self.refused,
            "served": self.served,
            "p_refuse": self.refused / self.arrived if self.arrived else 0.0,
            "Wq_terminal": avg(self.waits_terminal),
            "Lq_terminal": tavg(self.q_terminal),
            "Wq": avg(self.waits_queue),
            "Wq_max": max(self.waits_queue) if self.waits_queue else 0.0,
            "Lq": tavg(self.q_hall),
            "Lq_max": self.q_hall.max_value,
            "W": avg(self.times_in_system),
            "utilization": float(np.mean(list(util.values()))),
            "util_by_desk": util,
            "served_by_desk": dict(self.served_by_desk),
            "throughput": self.served / dur,     # клієнтів/хв
        }


def simulate(p: Params, warmup: float = 0.0, keep_model: bool = False):
    """Один прогін моделі. Повертає словник показників (і модель, якщо треба)."""
    m = BankBranch(p)
    res = m.run(warmup)
    return (res, m) if keep_model else res


if __name__ == "__main__":
    r = simulate(Params())
    for k, v in r.items():
        print(f"{k:15s} {v}")
