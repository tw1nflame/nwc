"""
Генерация СМЕШАННОГО синтетического датасета.

Для каждой строки (Статья × Дата) случайно выбирается pipeline:
- base → набор колонок, как в BASE
- base+ → набор колонок, как в BASE+

В результате формируется единый Excel:
- Лист data: объединение колонок обоих пайплайнов, для строки неиспользуемые колонки = NaN
- Листы coeffs_with_intercept, coeffs_no_intercept
- Лист TimeSeries_ensemble_models_info
- Листы Tabular_ensemble_models_info, Tabular_feature_importance (для BASE+)

Даты только как date (последний день месяца), без времени.
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)

# Параметры дат
START_DATE = '2023-01-31'
END_DATE = '2025-06-30'
DATE_RANGE = pd.date_range(start=START_DATE, end=END_DATE, freq='M')
N_MONTHS = len(DATE_RANGE)

# Список статей
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

# Колонки BASE
BASE_COLS = [
    'predict_naive', 'predict_autoARIMA', 'predict_TFT', 'predict_PatchTST', 'predict_Chronos_base',
    'predict_TS_ML', 'predict_svm6', 'predict_svm9', 'predict_svm12',
    'predict_linreg6_with_bias', 'predict_linreg9_with_bias', 'predict_linreg12_with_bias',
    'predict_linreg6_no_bias', 'predict_linreg9_no_bias', 'predict_linreg12_no_bias',
    'predict_stacking_RFR'
]

# Колонки BASE+
BASEPLUS_COLS = [
    'predict_naive',
    'predict_TS_ML', 'predict_TS_tabular', 'predict_svm6', 'predict_svm9', 'predict_svm12',
    'predict_linreg6_with_bias', 'predict_linreg9_with_bias', 'predict_linreg12_with_bias',
    'predict_linreg6_no_bias', 'predict_linreg9_no_bias', 'predict_linreg12_no_bias',
    'predict_stacking_RFR'
]

# Объединение всех возможных predict-колонок
ALL_PREDICT_COLS = sorted(list(set(BASE_COLS + BASEPLUS_COLS)))


def generate_predictions(fact: np.ndarray, cols: list[str]) -> dict[str, np.ndarray]:
    """Генерация синтетических предсказаний под заданные колонки."""
    data: dict[str, np.ndarray] = {}
    for col in cols:
        # Базовая: шум вокруг fact
        noise_scale = 6.0
        if 'autoARIMA' in col:
            data[col] = fact * np.random.uniform(0.97, 1.03, N_MONTHS) + np.random.normal(0, noise_scale - 1, N_MONTHS)
        elif 'TFT' in col:
            data[col] = fact * np.random.uniform(0.95, 1.05, N_MONTHS) + np.random.normal(0, noise_scale, N_MONTHS)
        elif 'PatchTST' in col or 'Chronos' in col:
            data[col] = fact * np.random.uniform(0.98, 1.02, N_MONTHS) + np.random.normal(0, noise_scale - 2, N_MONTHS)
        elif 'TS_tabular' in col:
            data[col] = fact * np.random.uniform(0.96, 1.04, N_MONTHS) + np.random.normal(0, noise_scale, N_MONTHS)
        elif 'TS_ML' in col:
            data[col] = fact * np.random.uniform(0.96, 1.04, N_MONTHS) + np.random.normal(0, noise_scale - 1, N_MONTHS)
        elif 'svm' in col:
            data[col] = fact + np.random.normal(0, noise_scale + 2, N_MONTHS)
        elif 'linreg' in col:
            data[col] = fact * np.random.uniform(0.99, 1.01, N_MONTHS) + np.random.normal(0, noise_scale - 2, N_MONTHS)
        elif 'stacking_RFR' in col:
            data[col] = fact * np.random.uniform(0.98, 1.02, N_MONTHS) + np.random.normal(0, noise_scale - 1, N_MONTHS)
        else:  # naive и прочие
            data[col] = fact + np.random.normal(0, noise_scale, N_MONTHS)
    return data


def main():
    rows: list[dict] = []
    any_baseplus = False

    # Для каждой статьи синтезируем временной ряд и выбираем pipeline помесячно
    for article in ARTICLES:
        # Базовая траектория факта
        fact = np.linspace(100, 200, N_MONTHS) + np.random.normal(0, 10, N_MONTHS)

        # Предсказания для всех колонок сразу (будем выбирать под pipeline)
        preds_all = generate_predictions(fact, ALL_PREDICT_COLS)

        for i in range(N_MONTHS):
            date = DATE_RANGE[i].date()
            # Случайный выбор пайплайна на строку
            use_baseplus = (np.random.rand() < 0.5)
            pipeline = 'base+' if use_baseplus else 'base'
            any_baseplus = any_baseplus or use_baseplus

            row: dict = {
                'Дата': date,
                'Fact': fact[i],
                'Статья': article,
                'pipeline': pipeline,
            }

            active_cols = BASEPLUS_COLS if use_baseplus else BASE_COLS

            # Заполняем только активные predict-колонки, остальные — NaN
            for col in ALL_PREDICT_COLS:
                row[col] = preds_all[col][i] if col in active_cols else np.nan

            # Разницы и отклонения по всем predict-колонкам
            for col in ALL_PREDICT_COLS:
                pred_val = row[col]
                row[f'{col} разница'] = (pred_val - row['Fact']) if pd.notna(pred_val) else np.nan
                row[f'{col} отклонение  %'] = (
                    100.0 * (pred_val - row['Fact']) / row['Fact']
                ) if (pd.notna(pred_val) and row['Fact'] != 0) else np.nan

            rows.append(row)

    # Собираем датафрейм
    df = pd.DataFrame(rows)
    # Итоговый порядок колонок: Дата, Fact, все predict, их разницы, их отклонения, Статья, pipeline
    final_cols = (
        ['Дата', 'Fact']
        + ALL_PREDICT_COLS
        + [f'{c} разница' for c in ALL_PREDICT_COLS]
        + [f'{c} отклонение  %' for c in ALL_PREDICT_COLS]
        + ['Статья', 'pipeline']
    )
    df = df[final_cols]
    # Дата — только date
    df['Дата'] = pd.to_datetime(df['Дата']).dt.date

    # Сохранение
    os.makedirs('results', exist_ok=True)
    out_path = 'results/synthetic_mixed_pipeline.xlsx'
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        # Лист data (даты только date)
        df['Дата'] = pd.to_datetime(df['Дата']).dt.date
        df.to_excel(writer, index=False, sheet_name='data')

        # Лист coeffs_with_intercept — супермножество весов
        coeffs_rows = []
        windows = [6, 9, 12]
        for article in ARTICLES:
            for win in windows:
                for date in DATE_RANGE:
                    coeffs_rows.append({
                        'Дата': date.date(),
                        'window': win,
                        'w_predict_naive': np.round(np.random.uniform(0.1, 1.0), 3),
                        'w_predict_TS_ML': np.round(np.random.uniform(0.1, 1.0), 3),
                        'w_predict_ML_tabular': np.round(np.random.uniform(0.1, 1.0), 3),
                        'w_predict_TABPFNMIX': np.round(np.random.uniform(0.1, 1.0), 3),
                        'bias': np.round(np.random.uniform(-10, 10), 2),
                        'Статья': article,
                    })
        coeffs_df = pd.DataFrame(coeffs_rows)
        coeffs_df['Дата'] = pd.to_datetime(coeffs_df['Дата']).dt.date
        coeffs_df.to_excel(writer, index=False, sheet_name='coeffs_with_intercept')

        # Лист coeffs_no_intercept — bias=0
        coeffs_no_rows = []
        for article in ARTICLES:
            for win in windows:
                for date in DATE_RANGE:
                    coeffs_no_rows.append({
                        'Дата': date.date(),
                        'window': win,
                        'w_predict_naive': np.round(np.random.uniform(0.1, 1.0), 3),
                        'w_predict_TS_ML': np.round(np.random.uniform(0.1, 1.0), 3),
                        'w_predict_ML_tabular': np.round(np.random.uniform(0.1, 1.0), 3),
                        'w_predict_TABPFNMIX': np.round(np.random.uniform(0.1, 1.0), 3),
                        'bias': 0.0,
                        'Статья': article,
                    })
        coeffs_no_df = pd.DataFrame(coeffs_no_rows)
        coeffs_no_df['Дата'] = pd.to_datetime(coeffs_no_df['Дата']).dt.date
        coeffs_no_df.to_excel(writer, index=False, sheet_name='coeffs_no_intercept')

        # Лист TimeSeries_ensemble_models_info
        ensemble_models = [
            'predict_autoARIMA', 'predict_TFT', 'predict_PatchTST', 'predict_Chronos_base',
            'predict_TS_ML', 'predict_naive', 'predict_stacking_RFR'
        ]
        ts_ens_rows = []
        for article in ARTICLES:
            for date in DATE_RANGE:
                n_models = np.random.randint(2, min(5, len(ensemble_models)) + 1)
                chosen = np.random.choice(ensemble_models, size=n_models, replace=False)
                weights = np.random.dirichlet(np.ones(n_models))
                ensemble_dict = {m.replace('predict_', ''): float(np.round(w, 3)) for m, w in zip(chosen, weights)}
                factor_value = 'INDIVIDUAL' if np.random.rand() < 0.5 else ''
                ts_ens_rows.append({
                    'Дата': date.date(),
                    'Статья': article,
                    'Ансамбль': str(ensemble_dict),
                    'Factor': factor_value,
                })
        ts_ens_df = pd.DataFrame(ts_ens_rows)
        ts_ens_df['Дата'] = pd.to_datetime(ts_ens_df['Дата']).dt.date
        ts_ens_df.to_excel(writer, index=False, sheet_name='TimeSeries_ensemble_models_info')

        # Лист Tabular_ensemble_models_info (имеет смысл при наличии base+ строк, но создаём всегда)
        tab_ens_rows = []
        for article in ARTICLES:
            for date in DATE_RANGE:
                n_models = np.random.randint(2, 5)
                chosen = np.random.choice(['TS_tabular', 'TABPFNMIX', 'TS_ML', 'svm6', 'svm9'], size=n_models, replace=False)
                weights = np.random.dirichlet(np.ones(n_models))
                ensemble_dict = {m: float(np.round(w, 3)) for m, w in zip(chosen, weights)}
                tab_ens_rows.append({
                    'Дата': date.date(),
                    'Статья': article,
                    'Ансамбль': str(ensemble_dict),
                })
        tab_ens_df = pd.DataFrame(tab_ens_rows)
        tab_ens_df['Дата'] = pd.to_datetime(tab_ens_df['Дата']).dt.date
        tab_ens_df.to_excel(writer, index=False, sheet_name='Tabular_ensemble_models_info')

        # Лист Tabular_feature_importance
        tab_feat_rows = []
        for article in ARTICLES:
            for date in DATE_RANGE:
                n_features = np.random.randint(3, 7)
                for i in range(n_features):
                    tab_feat_rows.append({
                        'feature': f'feature_{i+1}_{article}',
                        'importance': np.round(np.random.uniform(0, 1), 3),
                        'stddev': np.round(np.random.uniform(0.01, 0.2), 3),
                        'p_value': np.round(np.random.uniform(0.001, 0.2), 4),
                        'n': np.random.randint(50, 200),
                        'p99_high': np.round(np.random.uniform(0.5, 1.5), 3),
                        'p99_low': np.round(np.random.uniform(-1.5, 0.5), 3),
                        'Дата': date.date(),
                        'Статья': article,
                    })
        tab_feat_df = pd.DataFrame(tab_feat_rows)
        tab_feat_df['Дата'] = pd.to_datetime(tab_feat_df['Дата']).dt.date
        tab_feat_df.to_excel(writer, index=False, sheet_name='Tabular_feature_importance')

    print(f'Файл сохранён: {out_path}')


if __name__ == '__main__':
    main()


