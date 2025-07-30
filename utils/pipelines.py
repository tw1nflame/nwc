import os
import re
import json
import scipy
import shutil
import logging
import numpy as np
import pandas as pd

import yaml
from typing import Dict, Any

import plotly.express as px
import plotly.graph_objects as go

from datetime import datetime, timedelta
from functools import reduce
from dateutil.relativedelta import relativedelta
from pandas.tseries.offsets import MonthEnd

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.tabular import TabularPredictor

from utils.datahandler import *
from utils.config import *
from utils.predictors import *
from utils.common import *

logger = logging.getLogger(__name__)

def run_base_plus_pipeline(
        df_all_items,
        ITEMS_TO_PREDICT,
        config,
        DATE_COLUMN,
        RATE_COLUMN,
        ITEM_ID,
        FACTOR,
        models_to_use,
        TABPFNMIX_model,
        METRIC,
        CHOSEN_MONTH,
        MONTHES_TO_PREDICT,
        result_file_name,
        prev_predicts_file
    ):

    result_dfs = []
    linreg_w_intercept_weights_dfs = []
    linreg_no_intercept_weights_dfs = []
    ensemble_info_dfs = []
    tabular_ensemble_info_dfs = []
    feature_importance_dfs = []

    for target, features in ITEMS_TO_PREDICT.items():    # <!> Должны быть переданы статьи из интерфейса. st.session_state['items_to_predict']
        TARGET_COLUMN = target
        FEATURES = features
        PREDICT_TARGET_IN_USD = TARGET_COLUMN in config['Статьи для предикта в USD']
        FEATURES_TO_USD = TARGET_COLUMN in config['Фичи в Статьях для USD']
        FEATURES_TO_LAG = TARGET_COLUMN in config['Фичи в Статьях для LAG']
        TRANSFORM_DATE_COLUMN = TARGET_COLUMN in config['Статьи для раскладывания даты на месяц и год']

        df = df_all_items.copy()

        # Переводим фичи в USD, которые указаны в конфиге
        # <!> Нужно еще будет учесть, что это можно будет выбрать и через чекбокс в интерфейсе
        if FEATURES_TO_USD:
            features_to_usd = config['Фичи в Статьях для USD'][TARGET_COLUMN]
            for feature in features_to_usd:
                df[f'{feature}_USD'] = df[feature] / df[RATE_COLUMN]
                df = df.drop(columns=[feature])

                FEATURES.remove(feature)
                FEATURES.append(f'{feature}_USD')

        # Переводим таргет в USD, если это указано в конфиге.
        # <!> Нужно еще будет учесть, что это можно будет выбрать и через чекбокс в интерфейсе
        if PREDICT_TARGET_IN_USD:
            df[f'{TARGET_COLUMN}_USD'] = df[TARGET_COLUMN] / df[RATE_COLUMN]
            TARGET_COLUMN = f'{TARGET_COLUMN}_USD'

        # Делаем LAG для фичей, которые указаны в конфиге
        # <!> Нужно еще будет учесть, что это можно будет выбрать и через чекбокс в интерфейсе
        if FEATURES_TO_LAG:
            features_to_lag = config['Фичи в Статьях для LAG'][TARGET_COLUMN]
            for feature in features_to_lag:
                LAG_PERIODS = config['Фичи в Статьях для LAG'][TARGET_COLUMN][feature]
                df[f'{feature}_lag{LAG_PERIODS}'] = df[feature].shift(LAG_PERIODS)

                if feature != TARGET_COLUMN:
                    df = df.drop(columns=[feature])
                    FEATURES.remove(feature)

                FEATURES.append(f'{feature}_lag{LAG_PERIODS}')

        # Формируем фичи месяца и года из столбца Даты
        # <!> Нужно еще будет учесть, что это можно будет выбрать и через чекбокс в интерфейсе
        if TRANSFORM_DATE_COLUMN:
            df[f'{DATE_COLUMN}_year'] = df[DATE_COLUMN].dt.year
            df[f'{DATE_COLUMN}_month'] = df[DATE_COLUMN].dt.month

            FEATURES.extend([f'{DATE_COLUMN}_year', f'{DATE_COLUMN}_month'])

        # Формирование финального датафрейма с фичами для конкретной статьи
        df = df.loc[:, [DATE_COLUMN] + FEATURES + [TARGET_COLUMN]]
        df["item_id"] = ITEM_ID
        df["factor"] = FACTOR
        df = df.dropna()

        # Проверяем, что все столбцы из фич присутствуют в результирующей таблице
        missing_cols = [col for col in FEATURES if col not in df.columns]
        if missing_cols:
            raise ValueError(f'При обработке данных возникла ошибка и следующие столбцы не были добавлены в результирующий датафрейм: {missing_cols}')

        # Предикт наив | BASE+ AND BASE
        naive_predict = generate_naive_forecast(
            df=df,
            target_col=TARGET_COLUMN,
            date_col=DATE_COLUMN
        )

        # ----------------------------------
        # Предикт ENSMBLE TimeSeries AutoML | BASE+
        # ----------------------------------
        df_predict, TS_ML_model_info = generate_timeseries_predictions(
            df=df, 
            months_to_predict=[CHOSEN_MONTH], 
            metric=METRIC, 
            factors=[FACTOR], 
            targets=[TARGET_COLUMN],
            date_column=DATE_COLUMN,
            company="ALL",
            drop_covariates_features=False, # Оставляем фичи для BASE+ предикта
            delete_previous_models=False, 
            show_prediction_status=True,
            models=models_to_use
        )

        # Обработка ENSMBLE TimeSeries AutoML
        predict_TS_ML = df_predict.copy() \
                                .reset_index(drop=True) \
                                .drop(columns=["item_id", "factor"]) \
                                .rename(columns={f"{TARGET_COLUMN}_predict": "predict_TS_ML"})


        # ----------------------------------
        # Подготовка датасета для табличного предсказания
        # ----------------------------------
        df_tabular = df.copy()
        cols_for_lag = [col for col in df_tabular.columns if col not in (DATE_COLUMN, 'item_id', 'factor')]
        WINDOWS = [6, 9, 12]

        for column in cols_for_lag:
            df_tabular[f'{column}_lag1'] = df_tabular[column].shift(1)
            for WINDOW in WINDOWS:
                df_tabular[f'{column}_lag_1_MA_{WINDOW}'] =  df_tabular[column].rolling(window=WINDOW).mean().shift(1)
                df_tabular[f'{column}_lag_1_MIN_{WINDOW}'] = df_tabular[column].rolling(window=WINDOW).min().shift(1)
                df_tabular[f'{column}_lag_1_MAX_{WINDOW}'] = df_tabular[column].rolling(window=WINDOW).max().shift(1)

        df_tabular = df_tabular.drop(columns=[column for column in cols_for_lag if column != TARGET_COLUMN])
        df_tabular = df_tabular.dropna()

        # ----------------------------------
        # Предикт Tabular AutoML | BASE+
        # ----------------------------------
        predict_ML_tabular, DF_FEATURE_IMPORTANCE, DF_TABULAR_ENSEMBL_INFO = generate_tabular_predictions(
            df_tabular=df_tabular,
            target_column=TARGET_COLUMN,
            date_column=DATE_COLUMN,
            months_to_predict=[CHOSEN_MONTH],
            metric=METRIC.lower()
        )

        predict_ML_tabular = predict_ML_tabular.rename(columns={'predict': 'predict_ML_tabular'})
        DF_FEATURE_IMPORTANCE['Статья'] = TARGET_COLUMN
        DF_TABULAR_ENSEMBL_INFO['Статья'] = TARGET_COLUMN

        # ----------------------------------
        # Предикт TABPFNMIX | BASE+
        # ----------------------------------
        predict_TABPFNMIX, _, _ = generate_tabular_predictions(
            df_tabular=df_tabular,
            target_column=TARGET_COLUMN,
            date_column=DATE_COLUMN,
            months_to_predict=[CHOSEN_MONTH],
            metric=METRIC.lower(),
            models_to_use=TABPFNMIX_model
        )

        predict_TABPFNMIX = predict_TABPFNMIX.rename(columns={'predict': 'predict_TABPFNMIX'})

        # ----------------------------------
        # Формирование итогового файла с результатами предиктов | BASE+
        # ----------------------------------

        fact = df.loc[:, [DATE_COLUMN, TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "Fact"})

        all_models = reduce(
            lambda left, right: pd.merge(left, right, on=[DATE_COLUMN], how="outer"),
            [fact, naive_predict, predict_TS_ML, predict_ML_tabular, predict_TABPFNMIX]
        )
        print("[DEBUG] После merge:")
        print("  Столбцы:", all_models.columns.tolist())
        print("  Типы:", all_models.dtypes)
        print("  Уникальные даты:", all_models[DATE_COLUMN].sort_values().unique())
        print("  Head:", all_models.head(3))
        print("  Tail:", all_models.tail(3))
        all_models["Статья"] = TARGET_COLUMN

        # Проверка наличия ключевых столбцов
        required_cols = ["predict_naive", "predict_TS_ML", "predict_ML_tabular", "predict_TABPFNMIX"]
        for col in required_cols:
            if col not in all_models.columns:
                print(f"[ERROR] Нет столбца {col} в all_models!")
            else:
                print(f"[DEBUG] Столбец {col} найден, ненулевых: {all_models[col].notnull().sum()}")

        # Проверка наличия строки для CHOSEN_MONTH
        print(f"[DEBUG] Строка для даты {CHOSEN_MONTH + MonthEnd(0)}:")
        print(all_models[all_models[DATE_COLUMN] == CHOSEN_MONTH + MonthEnd(0)])

        # TEMPORARY FIX: Не дропаем строки по NaN в all_models, только по predict_TS_ML (оставляем как есть)
        # all_models = all_models.dropna(subset=["predict_TS_ML"]).reset_index(drop=True)
        # print("[DEBUG] После dropna по predict_TS_ML:")
        # print("  Уникальные даты:", all_models[DATE_COLUMN].sort_values().unique())
        # print("  Head:", all_models.head(3))
        # print("  Tail:", all_models.tail(3))
        # Вместо этого просто предупреждаем о пропусках
        print("[TEMPORARY FIX] Не дропаем строки по NaN, только предупреждение:")
        for col in ["predict_TS_ML"]:
            n_missing = all_models[col].isnull().sum()
            if n_missing > 0:
                print(f"[TEMPORARY FIX] {n_missing} пропусков в {col}")

        # ----------------------------------
        # Загрузка исторических прогнозов и присоединение к ним новых | BASE
        # ----------------------------------
        prev_predicts = pd.read_excel(prev_predicts_file, sheet_name='data')
        #prev_predicts = prev_predicts.loc[:, all_models.columns]
        prev_predicts = prev_predicts.loc[prev_predicts['Статья'] == TARGET_COLUMN]
        cols_to_use = [col for col in prev_predicts.columns if ('разница' not in col) and ('отклонение' not in col)]
        prev_predicts = prev_predicts.loc[:, cols_to_use]
        if prev_predicts[DATE_COLUMN].isin([CHOSEN_MONTH + MonthEnd(0)]).any():
            prev_predicts = prev_predicts.loc[prev_predicts[DATE_COLUMN] != CHOSEN_MONTH + MonthEnd(0)]
        
        all_models = pd.concat([prev_predicts, all_models])
        all_models = all_models.sort_values(by=DATE_COLUMN)

        prev_linreg_w_intercept = pd.read_excel(prev_predicts_file, sheet_name='coeffs_with_intercept')
        prev_linreg_w_intercept = prev_linreg_w_intercept.loc[prev_linreg_w_intercept['Статья'] == TARGET_COLUMN]
        if prev_linreg_w_intercept[DATE_COLUMN].isin([CHOSEN_MONTH + MonthEnd(0), CHOSEN_MONTH]).any():
            prev_linreg_w_intercept = prev_linreg_w_intercept.loc[prev_linreg_w_intercept[DATE_COLUMN] != CHOSEN_MONTH + MonthEnd(0)]
        linreg_w_intercept_weights_dfs.append(prev_linreg_w_intercept)

        prev_linreg_no_intercept = pd.read_excel(prev_predicts_file, sheet_name='coeffs_no_intercept')
        prev_linreg_no_intercept = prev_linreg_no_intercept.loc[prev_linreg_no_intercept['Статья'] == TARGET_COLUMN]
        if prev_linreg_no_intercept[DATE_COLUMN].isin([CHOSEN_MONTH + MonthEnd(0), CHOSEN_MONTH]).any():
            prev_linreg_no_intercept = prev_linreg_no_intercept.loc[prev_linreg_no_intercept[DATE_COLUMN] != CHOSEN_MONTH + MonthEnd(0)]
        linreg_no_intercept_weights_dfs.append(prev_linreg_no_intercept)

        prev_ensemble_info = pd.read_excel(prev_predicts_file, sheet_name='TimeSeries_ensemble_models_info')
        prev_ensemble_info = prev_ensemble_info.loc[prev_ensemble_info['Статья'] == TARGET_COLUMN]
        if prev_ensemble_info[DATE_COLUMN].isin([CHOSEN_MONTH + MonthEnd(0), CHOSEN_MONTH]).any():
            prev_ensemble_info = prev_ensemble_info.loc[prev_ensemble_info[DATE_COLUMN] != CHOSEN_MONTH + MonthEnd(0)]
        ensemble_info_dfs.append(prev_ensemble_info)

        prev_tab_ensemble_info = pd.read_excel(prev_predicts_file, sheet_name='Tabular_ensemble_models_info')
        prev_tab_ensemble_info = prev_tab_ensemble_info.loc[prev_tab_ensemble_info['Статья'] == TARGET_COLUMN]
        if prev_tab_ensemble_info[DATE_COLUMN].isin([CHOSEN_MONTH + MonthEnd(0), CHOSEN_MONTH]).any():
            prev_tab_ensemble_info = prev_tab_ensemble_info.loc[prev_tab_ensemble_info[DATE_COLUMN] != CHOSEN_MONTH + MonthEnd(0)]
        tabular_ensemble_info_dfs.append(prev_tab_ensemble_info)

        prev_feature_imp = pd.read_excel(prev_predicts_file, sheet_name='Tabular_feature_importance')
        prev_feature_imp = prev_feature_imp.loc[prev_feature_imp['Статья'] == TARGET_COLUMN]
        if prev_feature_imp[DATE_COLUMN].isin([CHOSEN_MONTH + MonthEnd(0), CHOSEN_MONTH]).any():
            prev_feature_imp = prev_feature_imp.loc[prev_feature_imp[DATE_COLUMN] != CHOSEN_MONTH + MonthEnd(0)]
        feature_importance_dfs.append(prev_feature_imp)
        
        # ----------------------------------
        # Stacking с помощью SVR 6,9,12 над полученными предиктами | BASE+
        # ----------------------------------

        WINDOWS = [6, 9, 12]

        predicts_to_use_as_features = [
            'predict_naive',
            'predict_TS_ML',
            'predict_ML_tabular',
            'predict_TABPFNMIX'
        ]

        # DEBUG: Проверка наличия всех дат и их количества до SVR (перед generate_svr_predictions)
        print("[DEBUG] Перед SVR: уникальные даты в all_models:")
        print(all_models[DATE_COLUMN].sort_values().unique())
        print(f"[DEBUG] Перед SVR: количество строк в all_models: {len(all_models)}")
        print(f"[DEBUG] Перед SVR: колонки: {all_models.columns.tolist()}")
        print(f"[DEBUG] Перед SVR: пропуски по ключевым колонкам:")
        for col in ['Fact', 'predict_naive', 'predict_TS_ML']:
            print(f"  {col}: {all_models[col].isnull().sum()} пропусков")
        print(f"[DEBUG] Перед SVR: статьи: {all_models['Статья'].unique()}")
        print(f"[DEBUG] Перед SVR: пример строк:")
        print(all_models.head(10))

        all_models = generate_svr_predictions(
            all_models=all_models,
            windows=WINDOWS,
            months_to_predict=generate_monthly_period(end_date=CHOSEN_MONTH),
            predicts_to_use_as_features=predicts_to_use_as_features,
            target_column='Fact',
            date_column=DATE_COLUMN,
            target_article=TARGET_COLUMN,
            ts_prediction_column='predict_TS_ML'
        )

        # ----------------------------------
        # Stacking с помощью LINREG 6,9,12 WITH INTERCEPT над полученными предиктами | BASE+
        # ----------------------------------

        # Вызов функции
        all_models, LINREG_WITH_INTERCEPT_WEIGHTS_DF = train_linear_models_for_windows(
            all_models=all_models,
            windows=WINDOWS,
            months_to_predict=generate_monthly_period(end_date=CHOSEN_MONTH),
            predicts_to_use_as_features=predicts_to_use_as_features,
            target_column='Fact',
            date_column=DATE_COLUMN,
            target_article=TARGET_COLUMN,
            fit_intercept=True,
            ts_prediction_column='predict_TS_ML'
        )

        LINREG_WITH_INTERCEPT_WEIGHTS_DF = LINREG_WITH_INTERCEPT_WEIGHTS_DF.loc[LINREG_WITH_INTERCEPT_WEIGHTS_DF[DATE_COLUMN] == CHOSEN_MONTH + MonthEnd(0)]

        # ----------------------------------
        # Stacking с помощью LINREG 6,9,12 NO INTERCEPT над полученными предиктами | BASE+
        # ----------------------------------

        # Вызов функции
        all_models, LINREG_NO_INTERCEPT_WEIGHTS_DF = train_linear_models_for_windows(
            all_models=all_models,
            windows=WINDOWS,
            months_to_predict=generate_monthly_period(end_date=CHOSEN_MONTH),
            predicts_to_use_as_features=predicts_to_use_as_features,
            target_column='Fact',
            date_column=DATE_COLUMN,
            target_article=TARGET_COLUMN,
            fit_intercept=False,
            ts_prediction_column='predict_TS_ML'
        )

        LINREG_NO_INTERCEPT_WEIGHTS_DF = LINREG_NO_INTERCEPT_WEIGHTS_DF.loc[LINREG_NO_INTERCEPT_WEIGHTS_DF[DATE_COLUMN] == CHOSEN_MONTH + MonthEnd(0)]

        # ----------------------------------
        # Stacking с помощью RFR над полученными предиктами | BASE+
        # ----------------------------------

        features_for_stacking = [
            DATE_COLUMN,
            'predict_naive',
            'predict_TS_ML',
            'predict_ML_tabular',
            'predict_TABPFNMIX',
            'predict_svm9',
            'predict_linreg9_no_bias',
            'Fact'          # not used while training
        ]

        all_models = train_stacking_RFR_model(
            all_models=all_models,
            prediction_date=CHOSEN_MONTH,
            features_for_stacking=features_for_stacking,
            target_article=TARGET_COLUMN,
            target_column='Fact',
            date_column=DATE_COLUMN,
            ts_prediction_column='predict_TS_ML'
        )

        # Расчет разниц и отклонений в %
        all_models = calculate_errors(all_models)

        # Извлечение информации об ансамблях TimeSeries обученных моделей
        DF_ENSMBLE_INFO = extract_ensemble_info(
            data=TS_ML_model_info,
            factor=FACTOR,
            DATE_COLUMN=DATE_COLUMN
        )

        # Saving results
        result_dfs.append(all_models)
        linreg_w_intercept_weights_dfs.append(LINREG_WITH_INTERCEPT_WEIGHTS_DF)
        linreg_no_intercept_weights_dfs.append(LINREG_NO_INTERCEPT_WEIGHTS_DF)
        ensemble_info_dfs.append(DF_ENSMBLE_INFO)
        tabular_ensemble_info_dfs.append(DF_TABULAR_ENSEMBL_INFO)
        feature_importance_dfs.append(DF_FEATURE_IMPORTANCE.astype(prev_feature_imp.dtypes)) # bug fix for GH55067


    # concats
    all_models = pd.concat(result_dfs)
    LINREG_WITH_INTERCEPT_WEIGHTS_DF = pd.concat(linreg_w_intercept_weights_dfs)
    LINREG_NO_INTERCEPT_WEIGHTS_DF = pd.concat(linreg_no_intercept_weights_dfs)
    DF_ENSMBLE_INFO = pd.concat(ensemble_info_dfs)
    DF_TABULAR_ENSEMBL_INFO = pd.concat(tabular_ensemble_info_dfs)
    DF_FEATURE_IMPORTANCE = pd.concat(feature_importance_dfs)

    # Saving to a single file
    path = "results"
    file = result_file_name
    os.makedirs(path, exist_ok=True)
    with pd.ExcelWriter(f"{path}/{file}") as writer:
        all_models.to_excel(writer, sheet_name='data', index=False)
        LINREG_WITH_INTERCEPT_WEIGHTS_DF.to_excel(writer, sheet_name='coeffs_with_intercept', index=False)
        LINREG_NO_INTERCEPT_WEIGHTS_DF.to_excel(writer, sheet_name='coeffs_no_intercept', index=False)
        DF_ENSMBLE_INFO.to_excel(writer, sheet_name='TimeSeries_ensemble_models_info', index=False)
        DF_TABULAR_ENSEMBL_INFO.to_excel(writer, sheet_name='Tabular_ensemble_models_info', index=False)
        DF_FEATURE_IMPORTANCE.to_excel(writer, sheet_name='Tabular_feature_importance', index=False)


def run_base_pipeline(
        df_all_items,
        ITEMS_TO_PREDICT,
        config,
        DATE_COLUMN,
        RATE_COLUMN,
        ITEM_ID,
        FACTOR,
        models_to_use,
        METRIC,
        CHOSEN_MONTH,
        MONTHES_TO_PREDICT,
        result_file_name,
        prev_predicts_file
    ):

    result_dfs = []
    linreg_w_intercept_weights_dfs = []
    linreg_no_intercept_weights_dfs = []
    ensemble_info_dfs = []

    for target, features in ITEMS_TO_PREDICT.items():    # <!> Должны быть переданы статьи из интерфейса. st.session_state['items_to_predict']
        TARGET_COLUMN = target
        FEATURES = features
        PREDICT_TARGET_IN_USD = TARGET_COLUMN in config['Статьи для предикта в USD']
        FEATURES_TO_USD = TARGET_COLUMN in config['Фичи в Статьях для USD']
        FEATURES_TO_LAG = TARGET_COLUMN in config['Фичи в Статьях для LAG']
        TRANSFORM_DATE_COLUMN = TARGET_COLUMN in config['Статьи для раскладывания даты на месяц и год']

        df = df_all_items.copy()

        # Переводим таргет в USD, если это указано в конфиге.
        # <!> Нужно еще будет учесть, что это можно будет выбрать и через чекбокс в интерфейсе
        if PREDICT_TARGET_IN_USD:
            df[f'{TARGET_COLUMN}_USD'] = df[TARGET_COLUMN] / df[RATE_COLUMN]
            TARGET_COLUMN = f'{TARGET_COLUMN}_USD'

        # Формирование финального датафрейма с фичами для конкретной статьи
        df = df.loc[:, [DATE_COLUMN, TARGET_COLUMN]]
        df["item_id"] = ITEM_ID
        df["factor"] = FACTOR
        df = df.dropna()

        # Предикт наив | BASE+ AND BASE
        naive_predict = generate_naive_forecast(
            df=df,
            target_col=TARGET_COLUMN,
            date_col=DATE_COLUMN
        )

        # ----------------------------------
        # Предикт ENSMBLE TimeSeries AutoML | BASE
        # ----------------------------------
        df_predict, TS_ML_model_info = generate_timeseries_predictions(
            df=df, 
            months_to_predict=[CHOSEN_MONTH], 
            metric=METRIC, 
            factors=[FACTOR], 
            targets=[TARGET_COLUMN],
            date_column=DATE_COLUMN,
            company="ALL",
            drop_covariates_features=True, # Удаляем фичи для BASE предикта
            delete_previous_models=False, 
            show_prediction_status=True,
            models=models_to_use
        )

        # Обработка ENSMBLE TimeSeries AutoML
        predict_TS_ML = df_predict.copy() \
                                .reset_index(drop=True) \
                                .drop(columns=["item_id", "factor"]) \
                                .rename(columns={f"{TARGET_COLUMN}_predict": "predict_TS_ML"})


        # ----------------------------------
        # Предикт AutoARIMA | BASE
        # ----------------------------------
        df_predict, _ = generate_timeseries_predictions(
            df=df, 
            months_to_predict=[CHOSEN_MONTH], 
            metric=METRIC, 
            factors=[FACTOR], 
            targets=[TARGET_COLUMN],
            date_column=DATE_COLUMN,
            company="ALL",
            drop_covariates_features=True, # Удаляем фичи для BASE предикта
            delete_previous_models=False, 
            show_prediction_status=True,
            models={'AutoARIMA': {}}
        )

        # Обработка AutoARIMA
        predict_autoARIMA = df_predict.copy() \
                                .reset_index(drop=True) \
                                .drop(columns=["item_id", "factor"]) \
                                .rename(columns={f"{TARGET_COLUMN}_predict": "predict_autoARIMA"})

        # ----------------------------------
        # Предикт TFT | BASE
        # ----------------------------------
        df_predict, _ = generate_timeseries_predictions(
            df=df, 
            months_to_predict=[CHOSEN_MONTH], 
            metric=METRIC, 
            factors=[FACTOR], 
            targets=[TARGET_COLUMN],
            date_column=DATE_COLUMN,
            company="ALL",
            drop_covariates_features=True, # Удаляем фичи для BASE предикта
            delete_previous_models=False, 
            show_prediction_status=True,
            models={'TemporalFusionTransformerModel': {}}
        )

        # Обработка TFT
        if not df_predict.empty:
            predict_TFT = df_predict.copy() \
                                    .reset_index(drop=True) \
                                    .drop(columns=["item_id", "factor"]) \
                                    .rename(columns={f"{TARGET_COLUMN}_predict": "predict_TFT"})
        else:
            predict_TFT = pd.DataFrame()

        # ----------------------------------
        # Предикт PatchTST | BASE
        # ----------------------------------
        df_predict, _ = generate_timeseries_predictions(
            df=df, 
            months_to_predict=[CHOSEN_MONTH], 
            metric=METRIC, 
            factors=[FACTOR], 
            targets=[TARGET_COLUMN],
            date_column=DATE_COLUMN,
            company="ALL",
            drop_covariates_features=True, # Удаляем фичи для BASE предикта
            delete_previous_models=False, 
            show_prediction_status=True,
            models={'PatchTSTModel': {}}
        )

        # Обработка PatchTST
        if not df_predict.empty:
            predict_PatchTST = df_predict.copy() \
                                    .reset_index(drop=True) \
                                    .drop(columns=["item_id", "factor"]) \
                                    .rename(columns={f"{TARGET_COLUMN}_predict": "predict_PatchTST"})
        else:
            predict_PatchTST = pd.DataFrame()

        # ----------------------------------
        # Предикт Chronos_base | BASE
        # ----------------------------------
        df_predict, _ = generate_timeseries_predictions(
            df=df, 
            months_to_predict=[CHOSEN_MONTH], 
            metric=METRIC, 
            factors=[FACTOR], 
            targets=[TARGET_COLUMN],
            date_column=DATE_COLUMN,
            company="ALL",
            drop_covariates_features=True, # Удаляем фичи для BASE предикта
            delete_previous_models=False, 
            show_prediction_status=True,
            models={'Chronos': {'model_path': 'pretrained_models/chronos-bolt-base', 'ag_args': {'name_suffix': 'ZeroShot'}}}
        )

        # Обработка Chronos_base
        if not df_predict.empty:
            predict_Chronos_base = df_predict.copy() \
                                    .reset_index(drop=True) \
                                    .drop(columns=["item_id", "factor"]) \
                                    .rename(columns={f"{TARGET_COLUMN}_predict": "predict_Chronos_base"})
        else:
            predict_Chronos_base = pd.DataFrame()

        # ----------------------------------
        # Формирование итогового файла с результатами предиктов | BASE
        # ----------------------------------
        fact = df.loc[:, [DATE_COLUMN, TARGET_COLUMN]].rename(columns={TARGET_COLUMN: "Fact"})
        predicts_to_merge = [pred for pred in [fact, naive_predict, predict_TS_ML, predict_autoARIMA, predict_TFT, predict_PatchTST, predict_Chronos_base] if not pred.empty]
        print("PREDS TO MERGE:", predicts_to_merge)
        all_models = reduce(
            lambda left, right: pd.merge(left, right, on=[DATE_COLUMN], how="outer"),
            predicts_to_merge
        )
        print('AFTER MERGE: ', all_models)
        all_models["Статья"] = TARGET_COLUMN
        all_models = all_models.dropna(subset=["predict_TS_ML"]).reset_index(drop=True)
        print('AFTER DROPNA: ', all_models)
        # ----------------------------------
        # Загрузка исторических прогнозов и присоединение к ним новых | BASE
        # ----------------------------------
        prev_predicts = pd.read_excel(prev_predicts_file, sheet_name='data')
        print('PREV PREDICTS: ', prev_predicts)
        print('TARGET_COLUMN, ', TARGET_COLUMN)
        #prev_predicts = prev_predicts.loc[:, all_models.columns]
        prev_predicts = prev_predicts.loc[prev_predicts['Статья'] == TARGET_COLUMN]
        print('PREV_PREDICTS_ARTICLE', prev_predicts)
        cols_to_use = [col for col in prev_predicts.columns if ('разница' not in col) and ('отклонение' not in col)]
        prev_predicts = prev_predicts.loc[:, cols_to_use]
        print('PREV_PREDICT COLS CHOOSE', prev_predicts)
        if prev_predicts[DATE_COLUMN].isin([CHOSEN_MONTH + MonthEnd(0)]).any():
            prev_predicts = prev_predicts.loc[prev_predicts[DATE_COLUMN] != CHOSEN_MONTH + MonthEnd(0)]
        print("PREV PREDICTS BEFORE CONCAT: ", prev_predicts)
        print('-----')
        print(all_models)
        all_models = pd.concat([prev_predicts, all_models])
        all_models = all_models.sort_values(by=DATE_COLUMN)

        prev_linreg_w_intercept = pd.read_excel(prev_predicts_file, sheet_name='coeffs_with_intercept')
        prev_linreg_w_intercept = prev_linreg_w_intercept.loc[prev_linreg_w_intercept['Статья'] == TARGET_COLUMN]
        if prev_linreg_w_intercept[DATE_COLUMN].isin([CHOSEN_MONTH + MonthEnd(0), CHOSEN_MONTH]).any():
            prev_linreg_w_intercept = prev_linreg_w_intercept.loc[prev_linreg_w_intercept[DATE_COLUMN] != CHOSEN_MONTH + MonthEnd(0)]
        linreg_w_intercept_weights_dfs.append(prev_linreg_w_intercept)

        prev_linreg_no_intercept = pd.read_excel(prev_predicts_file, sheet_name='coeffs_no_intercept')
        prev_linreg_no_intercept = prev_linreg_no_intercept.loc[prev_linreg_no_intercept['Статья'] == TARGET_COLUMN]
        if prev_linreg_no_intercept[DATE_COLUMN].isin([CHOSEN_MONTH + MonthEnd(0), CHOSEN_MONTH]).any():
            prev_linreg_no_intercept = prev_linreg_no_intercept.loc[prev_linreg_no_intercept[DATE_COLUMN] != CHOSEN_MONTH + MonthEnd(0)]
        linreg_no_intercept_weights_dfs.append(prev_linreg_no_intercept)

        prev_ensemble_info = pd.read_excel(prev_predicts_file, sheet_name='TimeSeries_ensemble_models_info')
        prev_ensemble_info = prev_ensemble_info.loc[prev_ensemble_info['Статья'] == TARGET_COLUMN]
        if prev_ensemble_info[DATE_COLUMN].isin([CHOSEN_MONTH + MonthEnd(0), CHOSEN_MONTH]).any():
            prev_ensemble_info = prev_ensemble_info.loc[prev_ensemble_info[DATE_COLUMN] != CHOSEN_MONTH + MonthEnd(0)]
        ensemble_info_dfs.append(prev_ensemble_info)
        

        # ----------------------------------
        # Stacking с помощью SVR 6,9,12 над полученными предиктами | BASE
        # ----------------------------------

        WINDOWS = [6, 9, 12]

        predicts_to_use_as_features = [
            'predict_naive',
            'predict_TS_ML',
        ]

        # DEBUG: Проверка наличия всех дат и их количества до SVR (перед generate_svr_predictions)
        print("[DEBUG] Перед SVR: уникальные даты в all_models:")
        print(all_models[DATE_COLUMN].sort_values().unique())
        print(f"[DEBUG] Перед SVR: количество строк в all_models: {len(all_models)}")
        print(f"[DEBUG] Перед SVR: колонки: {all_models.columns.tolist()}")
        print(f"[DEBUG] Перед SVR: пропуски по ключевым колонкам:")
        for col in ['Fact', 'predict_naive', 'predict_TS_ML']:
            print(f"  {col}: {all_models[col].isnull().sum()} пропусков")
        print(f"[DEBUG] Перед SVR: статьи: {all_models['Статья'].unique()}")
        print(f"[DEBUG] Перед SVR: пример строк:")
        print(all_models.head(10))

        all_models = generate_svr_predictions(
            all_models=all_models,
            windows=WINDOWS,
            months_to_predict=generate_monthly_period(end_date=CHOSEN_MONTH),
            predicts_to_use_as_features=predicts_to_use_as_features,
            target_column='Fact',
            date_column=DATE_COLUMN,
            target_article=TARGET_COLUMN,
            ts_prediction_column='predict_TS_ML'
        )

        # ----------------------------------
        # Stacking с помощью LINREG 6,9,12 WITH INTERCEPT над полученными предиктами | BASE
        # ----------------------------------

        # Вызов функции
        all_models, LINREG_WITH_INTERCEPT_WEIGHTS_DF = train_linear_models_for_windows(
            all_models=all_models,
            windows=WINDOWS,
            months_to_predict=generate_monthly_period(end_date=CHOSEN_MONTH),
            predicts_to_use_as_features=predicts_to_use_as_features,
            target_column='Fact',
            date_column=DATE_COLUMN,
            target_article=TARGET_COLUMN,
            fit_intercept=True,
            ts_prediction_column='predict_TS_ML'
        )

        LINREG_WITH_INTERCEPT_WEIGHTS_DF = LINREG_WITH_INTERCEPT_WEIGHTS_DF.loc[LINREG_WITH_INTERCEPT_WEIGHTS_DF[DATE_COLUMN] == CHOSEN_MONTH + MonthEnd(0)]

        # ----------------------------------
        # Stacking с помощью LINREG 6,9,12 NO INTERCEPT над полученными предиктами | BASE
        # ----------------------------------

        # Вызов функции
        all_models, LINREG_NO_INTERCEPT_WEIGHTS_DF = train_linear_models_for_windows(
            all_models=all_models,
            windows=WINDOWS,
            months_to_predict=generate_monthly_period(end_date=CHOSEN_MONTH),
            predicts_to_use_as_features=predicts_to_use_as_features,
            target_column='Fact',
            date_column=DATE_COLUMN,
            target_article=TARGET_COLUMN,
            fit_intercept=False,
            ts_prediction_column='predict_TS_ML'
        )

        LINREG_NO_INTERCEPT_WEIGHTS_DF = LINREG_NO_INTERCEPT_WEIGHTS_DF.loc[LINREG_NO_INTERCEPT_WEIGHTS_DF[DATE_COLUMN] == CHOSEN_MONTH + MonthEnd(0)]

        # ----------------------------------
        # Stacking с помощью RFR над полученными предиктами | BASE
        # ----------------------------------

        features_for_stacking = [
            DATE_COLUMN,
            'predict_naive',
            'predict_TS_ML',
            'predict_svm9',
            'predict_linreg9_no_bias',
            'Fact'          # not used while training
        ]

        all_models = train_stacking_RFR_model(
            all_models=all_models,
            prediction_date=CHOSEN_MONTH,
            features_for_stacking=features_for_stacking,
            target_article=TARGET_COLUMN,
            target_column='Fact',
            date_column=DATE_COLUMN,
            ts_prediction_column='predict_TS_ML'
        )

        # Расчет разниц и отклонений в %
        all_models = calculate_errors(all_models)

        # Извлечение информации об ансамблях TimeSeries обученных моделей
        DF_ENSMBLE_INFO = extract_ensemble_info(
            data=TS_ML_model_info,
            factor=FACTOR,
            DATE_COLUMN=DATE_COLUMN
        )

        # Saving results
        result_dfs.append(all_models)
        linreg_w_intercept_weights_dfs.append(LINREG_WITH_INTERCEPT_WEIGHTS_DF)
        linreg_no_intercept_weights_dfs.append(LINREG_NO_INTERCEPT_WEIGHTS_DF)
        ensemble_info_dfs.append(DF_ENSMBLE_INFO)


    # concats
    all_models = pd.concat(result_dfs)
    LINREG_WITH_INTERCEPT_WEIGHTS_DF = pd.concat(linreg_w_intercept_weights_dfs)
    LINREG_NO_INTERCEPT_WEIGHTS_DF = pd.concat(linreg_no_intercept_weights_dfs)
    DF_ENSMBLE_INFO = pd.concat(ensemble_info_dfs)

    # Saving to a single file
    path = "results"
    file = result_file_name
    os.makedirs(path, exist_ok=True)
    with pd.ExcelWriter(f"{path}/{file}") as writer:
        all_models.to_excel(writer, sheet_name='data', index=False)
        LINREG_WITH_INTERCEPT_WEIGHTS_DF.to_excel(writer, sheet_name='coeffs_with_intercept', index=False)
        LINREG_NO_INTERCEPT_WEIGHTS_DF.to_excel(writer, sheet_name='coeffs_no_intercept', index=False)
        DF_ENSMBLE_INFO.to_excel(writer, sheet_name='TimeSeries_ensemble_models_info', index=False)

        # DEBUG: Проверка наличия всех дат и их количества до merge с prev_predicts
        print("[DEBUG] Даты в all_models до merge:")
        print(all_models[DATE_COLUMN].sort_values().unique())
        print(f"[DEBUG] Количество строк в all_models до merge: {len(all_models)}")
        print(f"[DEBUG] Колонки в all_models до merge: {all_models.columns.tolist()}")
        print(f"[DEBUG] Пропуски по predict_TS_ML: {all_models['predict_TS_ML'].isnull().sum()}")
        print(f"[DEBUG] Пропуски по Fact: {all_models['Fact'].isnull().sum()}")
        print(f"[DEBUG] Пропуски по predict_naive: {all_models['predict_naive'].isnull().sum()}")
        print(f"[DEBUG] Пропуски по predict_autoARIMA: {all_models['predict_autoARIMA'].isnull().sum()}")
        print(f"[DEBUG] Пропуски по predict_TFT: {all_models['predict_TFT'].isnull().sum()}")
        print(f"[DEBUG] Пропуски по predict_PatchTST: {all_models['predict_PatchTST'].isnull().sum()}")
        print(f"[DEBUG] Пропуски по predict_Chronos_base: {all_models['predict_Chronos_base'].isnull().sum()}")
        print(f"[DEBUG] Статьи в all_models: {all_models['Статья'].unique()}")
        print(f"[DEBUG] Пример строк all_models:")
        print(all_models.head(10))