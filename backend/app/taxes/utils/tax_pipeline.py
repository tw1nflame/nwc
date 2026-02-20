import os
import shutil
import numpy as np
import pandas as pd
import json
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd
from typing import Any, List, Dict

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from sklearn.ensemble import RandomForestRegressor
import torch

def get_df(path,  factor,  item_id,  standart_cols,  tech_cols,  FEATURES_FOR_LAG1, TRANSFORM_DATE_COLUMN, features_to_use, TARGET_COLUMN):
    df = pd.read_excel(path)

    df["item_id"] = df['Кластер']
    df["factor"] = df['Группа']
    
    df = df.loc[(df['Группа'] == factor) & (df['Компания'] == item_id)]
    
    # for custom cols selection
    df = df.loc[:, tech_cols + standart_cols + features_to_use + [TARGET_COLUMN]]
    df['Дата'] = df['Дата'] + MonthEnd(0)


    if TARGET_COLUMN in FEATURES_FOR_LAG1:
        features_ = FEATURES_FOR_LAG1[TARGET_COLUMN]
        for feature_ in features_:
            df[f'{feature_}_lag1'] = df[feature_].shift(1)
            if feature_ != TARGET_COLUMN:
                df = df.drop(columns=[feature_])
                features_to_use.remove(feature_)
    
            features_to_use.append(f'{feature_}_lag1')

    if TRANSFORM_DATE_COLUMN:
        df['Дата_year'] = df['Дата'].dt.year
        df['Дата_month'] = df['Дата'].dt.month

        features_to_use.append(f'Дата_year')
        features_to_use.append(f'Дата_month')
        
    return df


def predict_individual(
    df, 
    MONTHES_TO_PREDICT=None, 
    METRIC=None, 
    FACTORS=None, 
    TARGETS=None,
    COMPANY=None, 
    delete_trained_model=True,
    delete_previous_models=True, 
    show_prediction_status=True,
    models=None
):
    
    ALL_MODELS_INFO = dict()
    RESULT_DFS = []
    for FACTOR in FACTORS:
        if show_prediction_status:
            print(f'Factor: {FACTOR}')
        # for loging models used in training
        ALL_MODELS_INFO.setdefault(FACTOR, dict())
        
        df_prep = df.loc[(df['factor'] == FACTOR)]
        
        if df_prep.empty:
            print(f'⚠️ Sub df for factor {FACTOR} is empty. skipping prediction for {FACTOR}')
            continue
        
        for MONTH in MONTHES_TO_PREDICT:
            if show_prediction_status:
                print(f'\tMonth: {MONTH}')
            # for loging models used in training
            DATE = MONTH.strftime('%Y-%m-%d')
            ALL_MODELS_INFO[FACTOR].setdefault(DATE, dict())
            
            MONTH_TO_PREDICT_first_day = MONTH
            MONTH_TO_PREDICT_last_day = MONTH_TO_PREDICT_first_day + MonthEnd(0)
            
            df_train = df_prep.loc[df_prep['Дата'] < MONTH_TO_PREDICT_first_day]
            #df_train = df_train.loc[:, ['item_id', 'Дата'] + TARGETS].reset_index(drop=True)
            
            for item_id in df_train['item_id'].unique():
                if show_prediction_status:
                    print(f'\t  Material: {item_id}')
                ALL_MODELS_INFO[FACTOR][DATE].setdefault(item_id, dict())
                
                df_item_id_train = df_train.loc[df_train['item_id'] == item_id]
                
                df_ts = TimeSeriesDataFrame.from_data_frame(
                    df_item_id_train,
                    id_column="item_id",
                    timestamp_column="Дата"
                )
                
                df_ts = df_ts.reset_index()
                
                # Заполняем пропуски в timeseries по месяцам нулевыми значениями
                full_index = pd.MultiIndex.from_product([
                    df_ts['item_id'].unique(),
                    pd.date_range(df_ts['timestamp'].min(), MONTH_TO_PREDICT_first_day - relativedelta(months=1) + MonthEnd(0), freq='M')
                ], names=['item_id', 'timestamp'])
                
                df_ts = df_ts.set_index(['item_id', 'timestamp']).reindex(full_index, fill_value=0).convert_frequency(freq="M")
                df_ts = df_ts.fillna(0)
                
                TARGET_PREDICTS = []
                for TARGET_COLUMN in TARGETS:
                    if show_prediction_status:
                        print(f'\t\tTarget: {TARGET_COLUMN}')
                    MODEL_PATH = f"models/{COMPANY}/base_{COMPANY}_{FACTOR}_{TARGET_COLUMN}_{MONTH_TO_PREDICT_last_day.strftime('%B%Y')}_{METRIC}"
                    
                    # delete previously trained model
                    if delete_previous_models:
                        if os.path.exists(MODEL_PATH):
                            shutil.rmtree(MODEL_PATH)
                    
                    df_for_train = df_ts.copy()
                    predictor = TimeSeriesPredictor(
                        prediction_length=1,
                        path=MODEL_PATH,
                        target=TARGET_COLUMN,
                        eval_metric=METRIC,
                        freq="M"
                    )
                    
                    predictor.fit(
                        df_for_train,
                        presets="best_quality",
                        hyperparameters=models,
                        random_seed=SEED,
                        #hyperparameters={
                        #    #'Naive': {},
                        #    #'SeasonalNaive': {},
                        #    'AutoARIMA': {},
                        #    #'RecursiveTabular': {},
                        #    #'NPTS': {},
                        #    #'DirectTabular': {},
                        #    #'CrostonSBA': {},
                        #    #'DynamicOptimizedTheta': {},
                        #},
                        #verbosity=0
                    )
                    
                    # for loging models used in training
                    model_info = predictor.info()['model_info'].copy()
                    for model_data in model_info.values():
                        model_data['eval_metric'] = METRIC
                        del model_data['quantile_levels']
                        if model_data['name'] != 'WeightedEnsemble':
                            del model_data['info_per_val_window']
                    
                    ALL_MODELS_INFO[FACTOR][DATE][item_id].setdefault(TARGET_COLUMN, dict())
                    ALL_MODELS_INFO[FACTOR][DATE][item_id][TARGET_COLUMN] = model_info
                    
                    predictions = predictor.predict(df_for_train)
                    
                    result = predictions.reset_index().loc[:, ["item_id", "timestamp", "mean"]]
                    result = result.rename(columns={"timestamp": "Дата", "mean": f"{TARGET_COLUMN}_predict"})
                    result['Дата'] = MONTH_TO_PREDICT_last_day
                    result['factor'] = FACTOR
                    
                    # QUICK FIX FOR TODO
                    RESULT_DFS.append(result)

                    # Очистка GPU памяти после обучения и предсказания
                    try:
                        del predictor
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception as e:
                        print(f'Предупреждение: не удалось очистить GPU память: {e}')

                    if delete_trained_model:
                        if os.path.exists(MODEL_PATH):
                            shutil.rmtree(MODEL_PATH)
                    
    return pd.concat(RESULT_DFS), ALL_MODELS_INFO


def get_naive_predict(df, TARGET_COLUMN):
    # naive
    naive_predict = df.loc[:, ["Дата", TARGET_COLUMN]].sort_values("Дата")
    
    last_date = naive_predict["Дата"].max() + MonthEnd(1)
    dummy_row = pd.DataFrame({"Дата": [last_date], TARGET_COLUMN: [np.nan]})
    
    naive_predict = pd.concat([naive_predict, dummy_row]).set_index("Дата")
    naive_predict = naive_predict.shift(1).reset_index().rename(columns={TARGET_COLUMN: f"{TARGET_COLUMN}_predict_naive"})
    return naive_predict


def get_svr_predict(WINDOWS, all_models, TARGET_COLUMN, MONTHES_TO_PREDICT):
    features_predicts_to_use = [
        f'{TARGET_COLUMN}_predict_naive',
        f'{TARGET_COLUMN}_predict_ML',
        #f'{TARGET_COLUMN}_predict_VAR'
    ]
    
    for WINDOW in WINDOWS:
        LINREG_MONTHES_TO_PREDICT = MONTHES_TO_PREDICT[WINDOW:]
        
        train_set_df = all_models.iloc[:WINDOW].dropna(subset=[f'{TARGET_COLUMN}_fact'])
        
        X_win = train_set_df[features_predicts_to_use]
        y_win = train_set_df[f'{TARGET_COLUMN}_fact']
        
        model_win = SVR()    #LinearRegression()   #Ridge(alpha=5.0)
        model_win.fit(X_win, y_win)
        
        X_test = all_models.loc[:WINDOW-1, features_predicts_to_use]
        
        y_win_pred = model_win.predict(X_test)
        all_models.loc[:WINDOW-1, f"{TARGET_COLUMN}_predict_svm{WINDOW}"] = y_win_pred
        
        for pred_month in LINREG_MONTHES_TO_PREDICT:
            pred_month_end = pred_month + MonthEnd(0)
            
            df_window_train = all_models.loc[all_models['Дата'] < pred_month_end].iloc[-WINDOW:]
            df_windows_pred = all_models.loc[all_models['Дата'] == pred_month_end]
            
            data = df_window_train.drop(columns=['Дата']).dropna(subset=[f'{TARGET_COLUMN}_predict_ML'])
            
            #if len(data) != WINDOW:
            #    print(f'Количество строк в датафрейме ({len(data)}) не соответствует окну ({WINDOW})')
            
            X = data[features_predicts_to_use]     
            y = data[f'{TARGET_COLUMN}_fact']
            
            model = SVR()    #LinearRegression()   #Ridge(alpha=5.0)
            model.fit(X, y)
            
            X_pred = df_windows_pred[features_predicts_to_use]  
            
            y_pred = model.predict(X_pred)
            all_models.loc[all_models['Дата'] == pred_month_end, f"{TARGET_COLUMN}_predict_svm{WINDOW}"] = y_pred
            
    return all_models


def get_linreg_with_bias_predict(WINDOWS, all_models, TARGET_COLUMN, MONTHES_TO_PREDICT):
    features_predicts_to_use = [
        f'{TARGET_COLUMN}_predict_naive',
        f'{TARGET_COLUMN}_predict_ML',
        #f'{TARGET_COLUMN}_predict_VAR'
    ]
    
    LINREG_WEIGHTS = []
    
    for WINDOW in WINDOWS:
        LINREG_MONTHES_TO_PREDICT = MONTHES_TO_PREDICT[WINDOW:]
        
        train_set_df = all_models.iloc[:WINDOW].dropna(subset=[f'{TARGET_COLUMN}_fact'])
        
        X_win = train_set_df[features_predicts_to_use]
        y_win = train_set_df[f'{TARGET_COLUMN}_fact']
        
        
        model_win = LinearRegression()   #Ridge(alpha=5.0)
        model_win.fit(X_win, y_win)
        
        coeffs_df = pd.DataFrame(
            {
                'Дата': [train_set_df.iloc[0]['Дата']],
                'window': [WINDOW],
                'w_naive': [model_win.coef_[0]], 
                'w_predict_ML': [model_win.coef_[1]],
                #'w_predict_VAR': [model_win.coef_[2]],
                'bias': [model_win.intercept_]
            }
        )
        coeffs_df['Дата'] = pd.to_datetime(coeffs_df['Дата'])
            
        LINREG_WEIGHTS.append(coeffs_df)
        
        X_test = all_models.loc[:WINDOW-1, features_predicts_to_use]
        
        y_win_pred = model_win.predict(X_test)
        all_models.loc[:WINDOW-1, f"{TARGET_COLUMN}_predict_linreg{WINDOW}"] = y_win_pred
        
        for pred_month in LINREG_MONTHES_TO_PREDICT:
            pred_month_end = pred_month + MonthEnd(0)
            
            df_window_train = all_models.loc[all_models['Дата'] < pred_month_end].iloc[-WINDOW:]
            df_windows_pred = all_models.loc[all_models['Дата'] == pred_month_end]
            
            data = df_window_train.drop(columns=['Дата']).dropna(subset=[f'{TARGET_COLUMN}_predict_ML'])
            
            #if len(data) != WINDOW:
            #    print(f'Количество строк в датафрейме ({len(data)}) не соответствует окну ({WINDOW})')
            
            X = data[features_predicts_to_use]     
            y = data[f'{TARGET_COLUMN}_fact']
            
            X_train, y_train = X, y
            
            model = LinearRegression()   #Ridge(alpha=5.0)
            model.fit(X_train, y_train)
            
            coeffs_df = pd.DataFrame(
                {
                    'Дата': [pred_month],
                    'window': [WINDOW],
                    'w_naive': [model.coef_[0]], 
                    'w_predict_ML': [model.coef_[1]],
                    #'w_predict_VAR': [model.coef_[2]],
                    'bias': [model.intercept_]
                }
            )
            coeffs_df['Дата'] = pd.to_datetime(coeffs_df['Дата'])
            
            LINREG_WEIGHTS.append(coeffs_df)
            
            X_pred = df_windows_pred[features_predicts_to_use]  
            
            y_pred = model.predict(X_pred)
            all_models.loc[all_models['Дата'] == pred_month_end, f"{TARGET_COLUMN}_predict_linreg{WINDOW}"] = y_pred
            
    LINREG_WEIGHTS_DF = pd.concat(LINREG_WEIGHTS).reset_index(drop=True)
    
    return all_models, LINREG_WEIGHTS_DF

def get_linreg_without_bias_predict(WINDOWS, all_models, TARGET_COLUMN, MONTHES_TO_PREDICT):
    features_predicts_to_use = [
        f'{TARGET_COLUMN}_predict_naive',
        f'{TARGET_COLUMN}_predict_ML',
        #f'{TARGET_COLUMN}_predict_VAR'
    ]
    
    LINREG_WEIGHTS = []
    
    for WINDOW in WINDOWS:
        LINREG_MONTHES_TO_PREDICT = MONTHES_TO_PREDICT[WINDOW:]
        
        train_set_df = all_models.iloc[:WINDOW].dropna(subset=[f'{TARGET_COLUMN}_fact'])
        
        X_win = train_set_df[features_predicts_to_use]
        y_win = train_set_df[f'{TARGET_COLUMN}_fact']
        
        
        model_win = LinearRegression(fit_intercept=False)   #Ridge(alpha=5.0)
        model_win.fit(X_win, y_win)
        
        coeffs_df = pd.DataFrame(
            {
                'Дата': [train_set_df.iloc[0]['Дата']],
                'window': [WINDOW],
                'w_naive': [model_win.coef_[0]], 
                'w_predict_ML': [model_win.coef_[1]],
                #'w_predict_VAR': [model_win.coef_[2]],
                #'bias': [model_win.intercept_]
            }
        )
        coeffs_df['Дата'] = pd.to_datetime(coeffs_df['Дата'])
            
        LINREG_WEIGHTS.append(coeffs_df)
        
        X_test = all_models.loc[:WINDOW-1, features_predicts_to_use]
        
        y_win_pred = model_win.predict(X_test)
        all_models.loc[:WINDOW-1, f"{TARGET_COLUMN}_predict_linreg_no_intercept{WINDOW}"] = y_win_pred
        
        for pred_month in LINREG_MONTHES_TO_PREDICT:
            pred_month_end = pred_month + MonthEnd(0)
            
            df_window_train = all_models.loc[all_models['Дата'] < pred_month_end].iloc[-WINDOW:]
            df_windows_pred = all_models.loc[all_models['Дата'] == pred_month_end]
            
            data = df_window_train.drop(columns=['Дата']).dropna(subset=[f'{TARGET_COLUMN}_predict_ML'])
            
            #if len(data) != WINDOW:
            #    print(f'Количество строк в датафрейме ({len(data)}) не соответствует окну ({WINDOW})')
            
            X = data[features_predicts_to_use]     
            y = data[f'{TARGET_COLUMN}_fact']
            
            X_train, y_train = X, y
            
            model = LinearRegression(fit_intercept=False)   #Ridge(alpha=5.0)
            model.fit(X_train, y_train)
            
            coeffs_df = pd.DataFrame(
                {
                    'Дата': [pred_month],
                    'window': [WINDOW],
                    'w_naive': [model.coef_[0]], 
                    'w_predict_ML': [model.coef_[1]],
                    #'w_predict_VAR': [model.coef_[2]],
                    #'bias': [model.intercept_]
                }
            )
            coeffs_df['Дата'] = pd.to_datetime(coeffs_df['Дата'])
            
            LINREG_WEIGHTS.append(coeffs_df)
            
            X_pred = df_windows_pred[features_predicts_to_use]  
            
            y_pred = model.predict(X_pred)
            all_models.loc[all_models['Дата'] == pred_month_end, f"{TARGET_COLUMN}_predict_linreg_no_intercept{WINDOW}"] = y_pred
            
    LINREG_NO_INTERCEPT_WEIGHTS_DF = pd.concat(LINREG_WEIGHTS).reset_index(drop=True)

    return all_models, LINREG_NO_INTERCEPT_WEIGHTS_DF

def get_RFR_predict(WINDOWS, all_models, TARGET_COLUMN, MONTHES_TO_PREDICT):
    features_for_stacking = [
        'Дата',
        f'{TARGET_COLUMN}_predict_naive',
        f'{TARGET_COLUMN}_predict_ML',
        f'{TARGET_COLUMN}_predict_svm9',
        f'{TARGET_COLUMN}_predict_linreg9',
        f'{TARGET_COLUMN}_fact'
    ]

    # TODO: FIX THIS ASAP
    MONTHES_TO_PREDICT_FOR_STACKING = [m for m in MONTHES_TO_PREDICT if m.year >= 2024]
    
    for pred_month in MONTHES_TO_PREDICT_FOR_STACKING:
        pred_month_end = pred_month + MonthEnd(0)
            
        df_train = all_models.loc[all_models['Дата'] < pred_month_end]
        df_test = all_models.loc[all_models['Дата'] == pred_month_end]
        
        data = df_train[features_for_stacking].drop(columns=['Дата']).dropna(subset=[f'{TARGET_COLUMN}_predict_ML'])
        
        X = data.drop(columns=[f'{TARGET_COLUMN}_fact']) 
        y = data[f'{TARGET_COLUMN}_fact']
        
        X_train, y_train = X, y
        
        model = RandomForestRegressor(random_state=SEED)    #LinearRegression()   #Ridge(alpha=5.0)
        model.fit(X_train, y_train)
        
        X_pred = df_test[features_for_stacking].drop(columns=['Дата', f'{TARGET_COLUMN}_fact'])
        
        y_pred = model.predict(X_pred)
        all_models.loc[all_models['Дата'] == pred_month_end, f"{TARGET_COLUMN}_predict_stacking"] = y_pred
    return all_models

def generate_monthly_period(
    end_date: datetime,
    months_before: int = 12,
    include_end_date: bool = True
) -> List[datetime]:
    """
    Генерирует список дат, представляющих ежемесячные точки за указанный период.
    
    Параметры:
        end_date: Конечная дата периода
        months_before: Количество месяцев до конечной даты (по умолчанию 12)
        include_end_date: Включать ли конечную дату в результат (по умолчанию True)
    
    Возвращает:
        Список объектов datetime, представляющих начало каждого месяца в периоде
    
    Пример:
        >>> generate_monthly_period(datetime(2025, 1, 1))
        [datetime(2024, 1, 1), datetime(2024, 2, 1), ..., datetime(2025, 1, 1)]
    """
    start_date = end_date - relativedelta(months=months_before)
    months = []
    current_date = start_date
    
    while current_date < end_date:
        months.append(current_date)
        current_date += relativedelta(months=1)
    
    if include_end_date and (not months or months[-1] != end_date):
        months.append(end_date)
    
    return months

def extract_ensemble_info(data: Dict,
                          factor: str,
                          DATE_COLUMN: str = 'Дата') -> pd.DataFrame:
    """
    Извлекает веса ансамбля из структуры:
    data[factor][date_str][series_group][target] -> models dict
    """
    records = []
    if factor not in data:
        return pd.DataFrame(columns=[DATE_COLUMN, 'Статья', 'Ансамбль'])

    for date_str, series_group_dict in data[factor].items():
        for series_group, target_dict in series_group_dict.items():
            # series_group, например "my_target_series"
            for target, model_info in target_dict.items():
                # target, например "RUB"
                if not isinstance(model_info, dict):
                    continue
                # ищем WeightedEnsemble
                if 'WeightedEnsemble' in model_info:
                    we_meta = model_info['WeightedEnsemble']
                    weights = we_meta.get('model_weights')
                    if weights:
                        rounded = {k: round(float(v), 4) for k,v in weights.items()}
                        records.append({
                            DATE_COLUMN: pd.to_datetime(date_str),
                            'Статья': target,
                            'Ансамбль': rounded
                        })
                    else:
                        # fallback - если нет model_weights, но есть имя
                        pass
    
    if not records:
        return pd.DataFrame(columns=[DATE_COLUMN, 'Статья', 'Ансамбль'])
        
    return pd.DataFrame(records)