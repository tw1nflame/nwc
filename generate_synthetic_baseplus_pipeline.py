# Генерация синтетических данных для BASE pipeline
# Этот скрипт создает синтетический Excel-файл, полностью совместимый с BASE pipeline.
# Даты — всегда последний день месяца, структура и названия столбцов соответствуют требованиям pipeline.

import pandas as pd
import numpy as np
import os
from pandas.tseries.offsets import MonthEnd

np.random.seed(42)

# Параметры
START_DATE = '2023-01-31'
END_DATE = '2025-06-30'
DATE_RANGE = pd.date_range(start=START_DATE, end=END_DATE, freq='M')
N_MONTHS = len(DATE_RANGE)

# Список статей из yaml (ЧОК ...)
ARTICLES = [
    'Торговая ДЗ_USD',
    'Прочая ДЗ',
    'Авансы выданные и расходы будущих периодов',
    'Прочие налоги к возмещению ST',
    'Прочие налоговые обязательства',
    'Задолженность перед персоналом',
    'Резерв по неиспользованным отпускам',
    'Краткосрочный резерв по премиям',
    'Прочее',
    'Кредиторская задлолженность по ОС',
    'Авансы полученные',
    'Авансы полученные(металлы)',
    'Торговая КЗ',
    'Авансовые платежи по налогу на прибыль',
    'Обязательства по налогу на прибыль',
    'ЧОК (нормализовано на расчеты с акционерами)'
]

# Основные колонки
base_cols = [
    'predict_naive', 'predict_autoARIMA', 'predict_TFT', 'predict_PatchTST', 'predict_Chronos_base',
    'predict_TS_ML', 'predict_svm6', 'predict_svm9', 'predict_svm12',
    'predict_linreg6_with_bias', 'predict_linreg9_with_bias', 'predict_linreg12_with_bias',
    'predict_linreg6_no_bias', 'predict_linreg9_no_bias', 'predict_linreg12_no_bias',
    'predict_stacking_RFR'
]

# Генерация данных
rows = []
for article in ARTICLES:
    fact = np.linspace(100, 200, N_MONTHS) + np.random.normal(0, 10, N_MONTHS)
    data = {col: fact + np.random.normal(0, 8, N_MONTHS) for col in base_cols}
    # Для разнообразия
    data['predict_autoARIMA'] = fact * np.random.uniform(0.96, 1.04, N_MONTHS) + np.random.normal(0, 6, N_MONTHS)
    data['predict_TFT'] = fact * np.random.uniform(0.95, 1.05, N_MONTHS) + np.random.normal(0, 7, N_MONTHS)
    data['predict_PatchTST'] = fact * np.random.uniform(0.97, 1.03, N_MONTHS) + np.random.normal(0, 4, N_MONTHS)
    data['predict_Chronos_base'] = fact * np.random.uniform(0.98, 1.02, N_MONTHS) + np.random.normal(0, 4, N_MONTHS)
    data['predict_TS_ML'] = fact * np.random.uniform(0.96, 1.04, N_MONTHS) + np.random.normal(0, 5, N_MONTHS)
    for i in range(N_MONTHS):
        row = {
            'Дата': DATE_RANGE[i].date(),  # <-- только дата, без времени
            'Fact': fact[i],
            'Статья': article
        }
        for col in base_cols:
            row[col] = data[col][i]
        row['pipeline'] = 'base'
        rows.append(row)
# Формируем датафрейм с синтетическими данными
df = pd.DataFrame(rows)
# Убедимся, что 'Дата' всегда datetime64[ns]
df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=False)

# Добавляем разницы и отклонения
for col in base_cols:
    df[f'{col} разница'] = df[col] - df['Fact']
    df[f'{col} отклонение  %'] = 100 * (df[col] - df['Fact']) / df['Fact']

# Итоговый порядок колонок
final_cols = ['Дата', 'Fact'] + base_cols + [f'{col} разница' for col in base_cols] + [f'{col} отклонение  %' for col in base_cols] + ['Статья', 'pipeline']
df = df[final_cols]

# УБРАТЬ преобразование даты в строку! Оставить как datetime64[ns] только с датой (без времени)
# Оставить только дату (datetime.date), чтобы в Excel не было времени
# Но для pandas merge/sort это будет тип object (date), а не datetime64[ns]
df['Дата'] = pd.to_datetime(df['Дата'], dayfirst=False).dt.date

# Сохраняем
os.makedirs('results', exist_ok=True)
with pd.ExcelWriter('results/synthetic_base_pipeline_BASE.xlsx', engine='openpyxl') as writer:
    # Все листы: дата только date
    df['Дата'] = df['Дата'].apply(lambda x: x if isinstance(x, (pd.Timestamp,)) else pd.to_datetime(x)).apply(lambda x: x.date())
    df.to_excel(writer, index=False, sheet_name='data')

    # Генерируем лист coeffs_with_intercept
    coeffs_rows = []
    windows = [6, 9, 12]
    for article in ARTICLES:
        for win in windows:
            for date in DATE_RANGE:
                coeffs_rows.append({
                    'Дата': date.date(),  # только дата
                    'window': win,
                    'w_predict_naive': np.round(np.random.uniform(0.1, 1.0), 3),
                    'w_predict_TS_ML': np.round(np.random.uniform(0.1, 1.0), 3),
                    'bias': np.round(np.random.uniform(-10, 10), 2),
                    'Статья': article
                })
    coeffs_df = pd.DataFrame(coeffs_rows)
    coeffs_df['Дата'] = coeffs_df['Дата'].apply(lambda x: x if isinstance(x, (pd.Timestamp,)) else pd.to_datetime(x)).apply(lambda x: x.date())
    coeffs_df.to_excel(writer, index=False, sheet_name='coeffs_with_intercept')

    # Генерируем лист coeffs_no_intercept (bias=0)
    coeffs_no_intercept_rows = []
    for article in ARTICLES:
        for win in windows:
            for date in DATE_RANGE:
                coeffs_no_intercept_rows.append({
                    'Дата': date.date(),
                    'window': win,
                    'w_predict_naive': np.round(np.random.uniform(0.1, 1.0), 3),
                    'w_predict_TS_ML': np.round(np.random.uniform(0.1, 1.0), 3),
                    'bias': 0.0,
                    'Статья': article
                })
    coeffs_no_intercept_df = pd.DataFrame(coeffs_no_intercept_rows)
    coeffs_no_intercept_df['Дата'] = coeffs_no_intercept_df['Дата'].apply(lambda x: x if isinstance(x, (pd.Timestamp,)) else pd.to_datetime(x)).apply(lambda x: x.date())
    coeffs_no_intercept_df.to_excel(writer, index=False, sheet_name='coeffs_no_intercept')

    # Генерируем лист с ансамблями
    ensemble_models = [
        'predict_autoARIMA', 'predict_TFT', 'predict_PatchTST', 'predict_Chronos_base', 'predict_TS_ML',
        'predict_naive', 'predict_stacking_RFR'
    ]
    ensemble_rows = []
    for article in ARTICLES:
        for date in DATE_RANGE:
            n_models = np.random.randint(2, min(5, len(ensemble_models))+1)
            chosen = np.random.choice(ensemble_models, size=n_models, replace=False)
            weights = np.random.dirichlet(np.ones(n_models))
            ensemble_dict = {model.replace('predict_', ''): float(np.round(w, 3)) for model, w in zip(chosen, weights)}
            ensemble_rows.append({
                'Дата': date.date(),
                'Статья': article,
                'Ансамбль': str(ensemble_dict)
            })
    ensemble_df = pd.DataFrame(ensemble_rows)
    ensemble_df['Дата'] = ensemble_df['Дата'].apply(lambda x: x if isinstance(x, (pd.Timestamp,)) else pd.to_datetime(x)).apply(lambda x: x.date())
    ensemble_df.to_excel(writer, index=False, sheet_name='TimeSeries_ensemble_models_info')
print('Файл сохранён: results/synthetic_base_pipeline_BASE.xlsx')

# После генерации df проверим, что нет пропусков по датам
expected_dates = pd.date_range(DATE_RANGE[0], DATE_RANGE[-1], freq='M').date
missing = set(expected_dates) - set(df['Дата'])
if missing:
    print('Пропущенные даты:', missing)
    raise ValueError('В синтетических данных есть пропуски по датам!')
