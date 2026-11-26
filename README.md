# Імітаційна модель роботи філії банку з п'ятьма касами

Курсова робота з ІМУС, тема № 24. Мова: Python 3, бібліотека SimPy.

## Запуск
Код програми лежить у папці [`bank_model`](bank_model).
```
cd bank_model
pip install -r requirements.txt
python main.py
```
Результати: `results/results.xlsx` (усі таблиці) і `figures/*.png` (рисунки).

## Файли
| Файл | Що робить |
|---|---|
| [`config.py`](bank_model/config.py) | усі параметри моделі |
| [`model.py`](bank_model/model.py) | модель філії: термінал талонів, зала очікування, 5 кас, збір статистики |
| [`analytics.py`](bank_model/analytics.py) | формули багатоканальної СМО M/M/n/m для верифікації |
| [`verification.py`](bank_model/verification.py) | порівняння з аналітикою, серії прогонів, граничні режими |
| [`experiment.py`](bank_model/experiment.py) | повний факторний експеримент 2², критерії Кохрена, Стьюдента, Фішера |
| [`plots.py`](bank_model/plots.py) | графіки |
| [`main.py`](bank_model/main.py) | запуск усього разом |

Окремі модулі теж можна запускати: `python model.py`, `python experiment.py`, `python verification.py`.
