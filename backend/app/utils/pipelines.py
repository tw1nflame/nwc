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
from utils.excel_formatter import *

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
        prev_predicts_file,
        status_manager=None
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

        # Обновляем статус текущей статьи, если передан status_manager (аналогично BASE)
        if status_manager:
            status_manager.update_current_article(TARGET_COLUMN)
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
    # Предикт TABPFNMIX | BASE+ (отключено)
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
            [fact, naive_predict, predict_TS_ML, predict_ML_tabular]  # predict_TABPFNMIX отключен
        )
        all_models["Статья"] = TARGET_COLUMN

        all_models = all_models.dropna(subset=["predict_TS_ML"]).reset_index(drop=True)

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
            'predict_TABPFNMIX'  # отключено
        ]

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
        # Приводим типы только по пересечению столбцов, чтобы избежать KeyError
        common_cols = [col for col in prev_feature_imp.dtypes.index if col in DF_FEATURE_IMPORTANCE.columns]
        if common_cols:
            dtypes_to_apply = {col: prev_feature_imp.dtypes[col] for col in common_cols}
            feature_importance_dfs.append(DF_FEATURE_IMPORTANCE.astype(dtypes_to_apply))
        else:
            feature_importance_dfs.append(DF_FEATURE_IMPORTANCE)

        # Обновляем счетчик обработанных статей (теперь внутри цикла, как в base)
        if status_manager:
            status_manager.increment_processed_articles()

    # concats
    all_models = pd.concat(result_dfs)
    LINREG_WITH_INTERCEPT_WEIGHTS_DF = pd.concat(linreg_w_intercept_weights_dfs)
    LINREG_NO_INTERCEPT_WEIGHTS_DF = pd.concat(linreg_no_intercept_weights_dfs)
    DF_ENSMBLE_INFO = pd.concat(ensemble_info_dfs)
    DF_TABULAR_ENSEMBL_INFO = pd.concat(tabular_ensemble_info_dfs)
    DF_FEATURE_IMPORTANCE = pd.concat(feature_importance_dfs)

    # Saving to a single file
    # result_file_name уже содержит полный путь к файлу
    os.makedirs(os.path.dirname(result_file_name), exist_ok=True)
    save_dataframes_to_excel(
        {
            'data': all_models,
            'coeffs_with_intercept': LINREG_WITH_INTERCEPT_WEIGHTS_DF,
            'coeffs_no_intercept': LINREG_NO_INTERCEPT_WEIGHTS_DF,
            'TimeSeries_ensemble_models_info': DF_ENSMBLE_INFO,
            'Tabular_ensemble_models_info': DF_TABULAR_ENSEMBL_INFO,
            'Tabular_feature_importance': DF_FEATURE_IMPORTANCE
        },
        result_file_name
    )


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
        prev_predicts_file,
        status_manager=None
    ):

    result_dfs = []
    linreg_w_intercept_weights_dfs = []
    linreg_no_intercept_weights_dfs = []
    ensemble_info_dfs = []

    for target, features in ITEMS_TO_PREDICT.items():    # <!> Должны быть переданы статьи из интерфейса. st.session_state['items_to_predict']
        TARGET_COLUMN = target
        
        # Обновляем статус текущей статьи
        if status_manager:
            status_manager.update_current_article(TARGET_COLUMN)
        
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
        # Используем относительный путь от корня проекта (как в рабочем примере)
        chronos_model_path = "pretrained_models/chronos-bolt-base"
        logger.warning(f"[DEBUG] Using Chronos model path: {chronos_model_path}")
        
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
            models={'Chronos': {'model_path': chronos_model_path, 'ag_args': {'name_suffix': 'ZeroShot'}}}
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

        all_models = reduce(
            lambda left, right: pd.merge(left, right, on=[DATE_COLUMN], how="outer"),
            predicts_to_merge
        )
        all_models["Статья"] = TARGET_COLUMN
        all_models = all_models.dropna(subset=["predict_TS_ML"]).reset_index(drop=True)

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
        

        # ----------------------------------
        # Stacking с помощью SVR 6,9,12 над полученными предиктами | BASE
        # ----------------------------------

        WINDOWS = [6, 9, 12]

        predicts_to_use_as_features = [
            'predict_naive',
            'predict_TS_ML',
        ]

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
        
        # Обновляем счетчик обработанных статей
        if status_manager:
            status_manager.increment_processed_articles()


    # concats
    all_models = pd.concat(result_dfs)
    LINREG_WITH_INTERCEPT_WEIGHTS_DF = pd.concat(linreg_w_intercept_weights_dfs)
    LINREG_NO_INTERCEPT_WEIGHTS_DF = pd.concat(linreg_no_intercept_weights_dfs)
    DF_ENSMBLE_INFO = pd.concat(ensemble_info_dfs)

    # Saving to a single file
    # result_file_name уже содержит полный путь к файлу
    os.makedirs(os.path.dirname(result_file_name), exist_ok=True)
    save_dataframes_to_excel(
        {
            'data': all_models,
            'coeffs_with_intercept': LINREG_WITH_INTERCEPT_WEIGHTS_DF,
            'coeffs_no_intercept': LINREG_NO_INTERCEPT_WEIGHTS_DF,
            'TimeSeries_ensemble_models_info': DF_ENSMBLE_INFO
        },
        result_file_name
    )