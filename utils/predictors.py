import os
import shutil
import numpy as np
import pandas as pd
from functools import reduce

from datetime import datetime
from pandas.tseries.offsets import MonthEnd
from dateutil.relativedelta import relativedelta

from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from autogluon.tabular import TabularPredictor
from typing import List, Dict, Optional, Tuple

from utils.common import normalize_to_list


def generate_naive_forecast(
    df: pd.DataFrame,
    target_col: str,
    date_col: str,
) -> pd.DataFrame:
    """
    Генерирует наивный прогноз путем сдвига последнего известного значения на заданный период.
    
    Параметры:
    ----------
    df : pd.DataFrame
        Исходный DataFrame с данными
    target_col : str
        Название целевой колонки для прогнозирования
    date_col : str, optional
        Название колонки с датой (по умолчанию "Дата")
    
    Возвращает:
    ----------
    pd.DataFrame
        DataFrame с наивным прогнозом
    """
    # Выбираем и сортируем нужные колонки
    forecast_df = df.loc[:, [date_col, target_col]].sort_values(date_col)
    
    # Генерируем дату прогноза
    forecast_date = forecast_df[date_col].max() + MonthEnd(1)
    
    # Создаем строку с прогнозом
    forecast_row = pd.DataFrame({
        date_col: [forecast_date],
        target_col: [np.nan]
    })
    
    # Объединяем и делаем прогноз (сдвиг)
    result = (
        pd.concat([forecast_df, forecast_row])
        .set_index(date_col)
        .shift(1)
        .reset_index()
        .rename(columns={target_col: "predict_naive"})
    )
    
    return result


def prepare_time_series_data(df_item_id_train, item_id, target_column, prediction_date, date_column):
    """
    Prepare time series data for training by converting to TimeSeriesDataFrame format
    and filling missing values.
    
    Args:
        df_item_id_train: DataFrame containing training data for a specific item_id
        item_id: The identifier for the item
        target_column: The column to predict
        prediction_date: The date for which prediction is being made
        
    Returns:
        TimeSeriesDataFrame ready for model training
    """
    # Convert to TimeSeriesDataFrame format
    df_ts = TimeSeriesDataFrame.from_data_frame(
        df_item_id_train,
        id_column="item_id",
        timestamp_column=date_column
    )
    
    df_ts = df_ts.reset_index()
    
    # Fill gaps in time series with zeros
    last_month = prediction_date - MonthEnd(1)
    full_index = pd.MultiIndex.from_product([
        df_ts['item_id'].unique(),
        pd.date_range(df_ts['timestamp'].min(), last_month, freq='M')
    ], names=['item_id', 'timestamp'])
    
    df_ts = df_ts.set_index(['item_id', 'timestamp']).reindex(full_index, fill_value=0).convert_frequency(freq="M")
    df_ts = df_ts.fillna(0)
    
    return df_ts

def train_and_predict_model(df_ts, target_column, model_path, metric, models_config):
    """
    Train a time series model and make predictions.
    
    Args:
        df_ts: TimeSeriesDataFrame containing the training data
        target_column: The column to predict
        model_path: Path where the model will be saved
        metric: Evaluation metric to use
        models_config: Configuration for models to use
        
    Returns:
        tuple: (predictions DataFrame, model information)
    """
    # Initialize and train the predictor
    predictor = TimeSeriesPredictor(
        prediction_length=1,
        path=model_path,
        target=target_column,
        eval_metric=metric,
        freq="M"
    )
    
    predictor.fit(
        df_ts,
        presets="best_quality",
        hyperparameters=models_config,
    )
    
    # Extract model information

    if len(models_config) == 1:
        model_info = None
    else:
        model_info = predictor.info()['model_info'].copy()
        for model_data in model_info.values():
            del model_data['quantile_levels']
            if model_data['name'] != 'WeightedEnsemble':
                del model_data['info_per_val_window']
    
    # Make predictions
    try:
        predictions = predictor.predict(df_ts)
    except Exception as E:
        print(f'Не удалось сделать прогноз из-за ошибки: {E}')
        predictions = pd.DataFrame()
        model_info = None
    
    return predictions, model_info

def format_predictions(predictions, target_column, prediction_date, factor, date_column):
    """
    Format the predictions into a standardized DataFrame.
    
    Args:
        predictions: Raw predictions from the model
        target_column: The column that was predicted
        prediction_date: The date for which prediction was made
        factor: The factor used for prediction
        
    Returns:
        DataFrame with formatted predictions
    """
    result = predictions.reset_index().loc[:, ["item_id", "timestamp", "mean"]]
    result = result.rename(columns={"timestamp": date_column, "mean": f"{target_column}_predict"})
    result[date_column] = prediction_date
    result['factor'] = factor
    
    return result

def generate_timeseries_predictions(
    df, 
    months_to_predict, 
    metric, 
    factors, 
    targets,
    date_column,
    company, 
    drop_covariates_features,
    delete_previous_models=True, 
    show_prediction_status=True,
    models=None
):
    """
    Make time series predictions for multiple factors, months, and targets.
    
    Args:
        df: DataFrame containing the data
        months_to_predict: List of months for which to make predictions
        metric: Evaluation metric to use
        factors: List of factors to predict for
        targets: List of target columns to predict
        company: Company identifier for model path
        delete_previous_models: Whether to delete previous models before training
        show_prediction_status: Whether to show prediction status messages
        models: Configuration for models to use
        
    Returns:
        tuple: (DataFrame with all predictions, dictionary with model information)
    """
    all_models_info = {}
    result_dfs = []

    if not isinstance(targets, list):
        print(f"targets должен быть списком, получен {type(targets)}")
        targets = normalize_to_list(targets)
    
    for factor in factors:
        if show_prediction_status:
            print(f'Factor: {factor}')
        if show_prediction_status:
            print(f"[DEBUG] generate_timeseries_predictions: factor={factor}, months={months_to_predict}, targets={targets}")
        # Initialize factor info in the model info dictionary
        all_models_info.setdefault(factor, {})
        # Filter data for the current factor
        df_factor = df.loc[df['factor'] == factor]
        print(f"[DEBUG] df_factor shape: {df_factor.shape}, columns: {df_factor.columns.tolist()}")
        print(f"[DEBUG] df_factor head:\n{df_factor.head(2)}")
        if df_factor.empty:
            print(f"[ERROR] df_factor is empty for factor={factor}")
        for month in months_to_predict:
            print(f"[DEBUG] Predicting for month: {month}")
            if show_prediction_status:
                print(f'\tMonth: {month}')
            
            # Format date for model info dictionary
            date_str = month.strftime('%Y-%m-%d')
            all_models_info[factor].setdefault(date_str, {})
            
            # Define prediction period
            prediction_date = month + MonthEnd(0)
            
            # Filter training data
            df_train = df_factor.loc[df_factor[date_column] < month]
            # Drop covariates features (for BASE algorithm when we use only item_id, date and target)
            if drop_covariates_features:
                df_train = df_train.loc[:, ['item_id', date_column] + targets].reset_index(drop=True)

            for item_id in df_train['item_id'].unique():
                if show_prediction_status:
                    print(f'\t  Material: {item_id}')
                
                all_models_info[factor][date_str].setdefault(item_id, {})
                
                # Filter data for current item_id
                df_item_train = df_train.loc[df_train['item_id'] == item_id]
                
                for target_column in targets:
                    # Prepare time series data
                    df_ts = prepare_time_series_data(df_item_train, item_id, target_column, month, date_column)

                    if show_prediction_status:
                        print(f'\t\tTarget: {target_column}')
                    
                    # Define model path
                    model_path = f"models/{company}/base_{company}_{factor}_{target_column}_{prediction_date.strftime('%B%Y')}_{metric}"
                    
                    # Delete previous model if requested
                    if delete_previous_models and os.path.exists(model_path):
                        shutil.rmtree(model_path)
                    
                    # Train model and make predictions
                    predictions, model_info = train_and_predict_model(
                        df_ts, 
                        target_column, 
                        model_path, 
                        metric, 
                        models
                    )
                    
                    if not predictions.empty:
                        # Store model info
                        all_models_info[factor][date_str][item_id][target_column] = model_info

                        # Format and store predictions
                        result = format_predictions(predictions, target_column, prediction_date, factor, date_column)
                        result_dfs.append(result)
    
    # Combine all predictions
    if result_dfs:
        print(f"[DEBUG] result_dfs shapes: {[df.shape for df in result_dfs]}")
        return pd.concat(result_dfs), all_models_info
    else:
        print(f"[ERROR] No predictions generated in generate_timeseries_predictions!")
        return pd.DataFrame(), all_models_info


def generate_tabular_predictions(
    df_tabular: pd.DataFrame,
    target_column: str,
    date_column: str,
    months_to_predict: List[datetime],
    metric: str = None,
    models_to_use: str | dict = 'default'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Генерирует прогнозы с использованием AutoGluon TabularPredictor для каждого указанного месяца.
    
    Параметры:
        df_tabular: DataFrame с данными для обучения и прогнозирования
        target_column: Название целевой переменной
        date_column: Название столбца, отражающего дату в df_tabular
        months_to_predict: Список дат для прогнозирования
        metric: Метрика для оценки моделей (по умолчанию None)
        models_to_use: Словарь из моделей, которые будут использоваться в обучении. По умолчанию None, что приводит к использованию всех доступных моделей
    Возвращает:
        Кортеж из трех DataFrame:
        1. Прогнозы для каждого месяца
        2. Важность признаков для каждого месяца
        3. Информация об ансамблях моделей для каждого месяца
    """
    
    # Инициализация структур для хранения результатов
    all_predictions = []
    feature_importances = []
    ensembles_info = []
    
    for month in months_to_predict:
        month = month + MonthEnd(0)
        print(f'Прогнозируем для месяца: {month.strftime("%Y-%m-%d")}')
        
        # Подготовка данных
        train_data = df_tabular[df_tabular[date_column] < month].drop(columns=[date_column])
        test_data = df_tabular[df_tabular[date_column] == month].drop(columns=[target_column, date_column])
        
        # Обучение модели
        predictor = TabularPredictor(
            label=target_column,
            eval_metric=metric.lower(),
            problem_type='regression'
        )

        predictor = predictor.fit(
            train_data=train_data,
            hyperparameters=models_to_use,
            verbosity=2
        )
        
        # Получение важности признаков
        try:
            month_importance = predictor.feature_importance(train_data)
            month_importance = (
                month_importance.reset_index()
                .rename(columns={'index': 'feature'})
                .assign(Дата=month)
            )
            feature_importances.append(month_importance)
        except Exception as E:
            print(f'Ошибка при попытке получить feature importance для моделей: {models_to_use}')
        
        # Получение информации об ансамбле
        try:
            if len(models_to_use) == 1:
                ensemble_weights = list(models_to_use)[0]
            else:
                ensemble_weights = (
                    predictor.info()['model_info']['WeightedEnsemble_L2']
                    ['children_info']['S1F1']['model_weights']
                )

            ensemble_df = pd.DataFrame({
                date_column: [month],
                'Ансамбль': [ensemble_weights]
            })
            ensembles_info.append(ensemble_df)
        except Exception as e:
            print(f'Не удалось получить информацию об ансамбле для {month} для моделей: {models_to_use}: {str(e)}')
        
        # Прогнозирование
        prediction = predictor.predict(test_data)
        prediction_df = pd.DataFrame({
            date_column: [month],
            'predict': [prediction.iloc[0]]
        })
        all_predictions.append(prediction_df)
    
    # Объединение результатов
    predictions = pd.concat(all_predictions) if all_predictions else pd.DataFrame()
    importance_df = pd.concat(feature_importances) if feature_importances else pd.DataFrame()
    ensembles_df = pd.concat(ensembles_info) if ensembles_info else pd.DataFrame()
    
    return predictions, importance_df, ensembles_df


def generate_svr_predictions(
    all_models: pd.DataFrame,
    windows: List[int],
    months_to_predict: List[datetime],
    predicts_to_use_as_features: List[str],
    target_column: str,
    date_column: str,
    target_article: str,
    ts_prediction_column: str = 'predict_TS_ML'
) -> pd.DataFrame:
    """
    Обучает модели SVR для различных временных окон и делает прогнозы.
    
    Параметры:
        all_models: DataFrame с исходными данными и предыдущими прогнозами
        windows: Список размеров окон для обучения моделей
        months_to_predict: Список дат для прогнозирования
        predicts_to_use_as_features: Список колонок с прогнозами, используемых как признаки
        target_column: Название целевой переменной
        date_column: Название колонки с датой (по умолчанию 'Дата')
        ts_prediction_column: Название колонки с прогнозами временных рядов (по умолчанию 'predict_TS_ML')
    
    Возвращает:
        Модифицированный DataFrame с добавленными прогнозами для каждого окна
    
    Исключения:
        ValueError: Если размер данных не соответствует размеру окна
    """
    
    print(f"[DEBUG] generate_svr_predictions: all_models shape={all_models.shape}, columns={all_models.columns.tolist()}")
    print(f"[DEBUG] unique dates: {all_models[date_column].sort_values().unique()}")
    print(f"[DEBUG] head:\n{all_models.head(2)}")
    
    all_models = all_models.reset_index(drop=True)
    # ===== TEMPORARY FIX: Начало =====
    # Исключаем признаки с NaN только для стекера, не дропаем строки
    valid_features = [col for col in predicts_to_use_as_features if not all_models[col].isna().any()]
    if len(valid_features) < len(predicts_to_use_as_features):
        print(f"[TEMPORARY FIX] Некоторые признаки содержат NaN и будут исключены из стекера: {set(predicts_to_use_as_features) - set(valid_features)}")
    if not valid_features:
        raise ValueError("[TEMPORARY FIX] Нет ни одного признака без NaN для обучения SVR!")
    # ===== TEMPORARY FIX: Конец =====
    for window in windows:
        print(f"[DEBUG] SVR window: {window}")
        
        current_months_to_predict = months_to_predict[window:]
        train_set_df = all_models.iloc[:window].dropna(subset=[target_column])
        X_train = train_set_df[valid_features]
        y_train = train_set_df[target_column]
        model_win = SVR()
        model_win.fit(X_train, y_train)
        X_test = all_models.loc[:window-1, valid_features]
        y_pred = model_win.predict(X_test)
        all_models.loc[:window-1, f"predict_svm{window}"] = y_pred
        for pred_month in current_months_to_predict:
            pred_month_end = pred_month + MonthEnd(0)
            window_train_df = all_models.loc[all_models[date_column] < pred_month_end].iloc[-window:]
            window_pred_df = all_models.loc[all_models[date_column] == pred_month_end]
            data = window_train_df.drop(columns=[date_column]).dropna(subset=[ts_prediction_column])
            print(f"\n[DEBUG] SVR window: {window}")
            print(f"[DEBUG] pred_month_end: {pred_month_end}")
            print(f"[DEBUG] window_train_df.shape: {window_train_df.shape}")
            print(f"[DEBUG] window_train_df['{date_column}'] tail:")
            print(window_train_df[date_column].tail())
            print(f"[DEBUG] window_pred_df.shape: {window_pred_df.shape}")
            print(f"[DEBUG] data.shape: {data.shape}")
            print(f"[DEBUG] data index:")
            print(data.index)
            if len(data) != window:
                print(f"[ERROR] Data length: {len(data)}, Window: {window}, Date: {pred_month_end}")
                print(f"[ERROR] data head:")
                print(data.head())
                print(f"[ERROR] window_train_df head:")
                print(window_train_df.head())
                print(f"[ERROR] all_models[{date_column}] == {pred_month_end}:")
                print(all_models.loc[all_models[date_column] == pred_month_end])
                raise ValueError(
                    f'Размер данных ({len(data)}) не соответствует размеру окна ({window}) ' +
                    f'для даты {pred_month_end.strftime("%Y-%m-%d")}'
                )
            X = data[valid_features]
            y = data[target_column]
            model = SVR()
            model.fit(X, y)
            X_pred = window_pred_df[valid_features]
            y_pred = model.predict(X_pred)
            all_models.loc[
                (all_models[date_column] == pred_month_end) & 
                (all_models["Статья"] == target_article),
                f"predict_svm{window}"
            ] = y_pred
    return all_models

def _create_weights_dataframe(
    model: LinearRegression,
    date: datetime,
    window: int,
    target_article: str,
    features: List[str]
) -> pd.DataFrame:
    """
    Создает DataFrame с весами модели.
    
    Параметры:
        model: Обученная линейная модель
        date: Дата для записи
        window: Размер окна
        target_article: Название статьи
        features: Список признаков
    
    Возвращает:
        DataFrame с весами модели
    """
    weights = {
        'Дата': [date],
        'Статья': [target_article],
        'window': [window]
    }
    
    # Динамически добавляем веса для каждого признака
    for i, feature in enumerate(features):
        weights[f'w_{feature}'] = [model.coef_[i]]
    
    weights['bias'] = model.intercept_

    return pd.DataFrame(weights)


def train_linear_models_for_windows(
    all_models: pd.DataFrame,
    windows: List[int],
    months_to_predict: List[datetime],
    predicts_to_use_as_features: List[str],
    target_column: str,
    target_article: str,
    fit_intercept: bool = True,
    date_column: str = 'Дата',
    ts_prediction_column: str = 'predict_TS_ML',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Обучает линейные модели для различных временных окон и возвращает прогнозы и веса моделей.
    
    Параметры:
        all_models: DataFrame с исходными данными и прогнозами
        windows: Список размеров окон для обучения
        months_to_predict: Список дат для прогнозирования
        predicts_to_use_as_features: Список используемых признаков-прогнозов
        target_column: Название целевой переменной
        target_article: Название статьи для фильтрации
        date_column: Название колонки с датой (по умолчанию 'Дата')
        ts_prediction_column: Колонка с прогнозами временных рядов (по умолчанию 'predict_TS_ML')
    
    Возвращает:
        Кортеж из:
        1. Модифицированный DataFrame с добавленными прогнозами
        2. DataFrame с весами моделей для каждого окна и даты
    
    Исключения:
        ValueError: Если размер данных не соответствует размеру окна
    """

    weights_data = []
    all_models = all_models.reset_index(drop=True)
    
    for window in windows:
        intercept_suffix = ("no_bias", "with_bias")
        result_column_name = f"predict_linreg{window}_{intercept_suffix[fit_intercept]}"

        current_months_to_predict = months_to_predict[window:]
        
        # Обучаем начальную модель на первых window точках
        train_set = all_models.iloc[:window].dropna(subset=[target_column])
        X_train = train_set[predicts_to_use_as_features]
        y_train = train_set[target_column]
        
        model_win = LinearRegression(fit_intercept=fit_intercept)
        model_win.fit(X_train, y_train)
        
        # Сохраняем веса начальной модели
        weights_data.append(_create_weights_dataframe(
            model_win, 
            date=train_set.iloc[0][date_column],
            window=window,
            target_article=target_article,
            features=predicts_to_use_as_features
        ))
        
        # Прогнозируем для начального окна
        X_test = all_models.loc[:window-1, predicts_to_use_as_features]
        y_pred = model_win.predict(X_test)
        all_models.loc[:window-1, result_column_name] = y_pred
        
        # Прогнозируем для каждого последующего месяца
        for pred_month in current_months_to_predict:
            pred_month_end = pred_month + MonthEnd(0)
            
            # Подготавливаем данные для текущего окна
            window_train = all_models.loc[
                all_models[date_column] < pred_month_end
            ].iloc[-window:]
            
            window_data = window_train.drop(columns=[date_column]).dropna(subset=[ts_prediction_column])
            
            if len(window_data) != window:
                raise ValueError(
                    f'Несоответствие размера данных ({len(window_data)}) '
                    f'размеру окна ({window}) для {pred_month_end.strftime("%Y-%m-%d")}'
                )
            
            # Обучаем и прогнозируем
            X = window_data[predicts_to_use_as_features]
            y = window_data[target_column]
            
            model = LinearRegression(fit_intercept=fit_intercept)
            model.fit(X, y)
            
            # Сохраняем веса модели
            weights_data.append(_create_weights_dataframe(
                model,
                date=pred_month_end,
                window=window,
                target_article=target_article,
                features=predicts_to_use_as_features
            ))
            
            # Делаем прогноз
            window_pred = all_models.loc[all_models[date_column] == pred_month_end]
            X_pred = window_pred[predicts_to_use_as_features]
            y_pred = model.predict(X_pred)
            
            all_models.loc[
                (all_models[date_column] == pred_month_end) & 
                (all_models["Статья"] == target_article),
                result_column_name
            ] = y_pred
    
    weights_df = pd.concat(weights_data).reset_index(drop=True)
    return all_models, weights_df


def train_stacking_RFR_model(
    all_models: pd.DataFrame,
    prediction_date: datetime,
    features_for_stacking: List[str],
    target_article: str,
    target_column: str = 'Fact',
    date_column: str = 'Дата',
    ts_prediction_column: str = 'predict_TS_ML',
) -> pd.DataFrame:
    """
    Обучает модель стекинга (Random Forest) на предсказаниях базовых моделей и делает прогноз.
    
    Параметры:
        all_models: DataFrame с исходными данными и предсказаниями базовых моделей
        prediction_date: Дата, для которой нужно сделать прогноз
        features_for_stacking: Список признаков для стекинга
        target_article: Название статьи для фильтрации
        target_column: Название целевой переменной (по умолчанию 'Fact')
        date_column: Название колонки с датой (по умолчанию 'Дата')
        ts_prediction_column: Колонка с прогнозами временных рядов (по умолчанию 'predict_TS_ML')
    
    Возвращает:
        Модифицированный DataFrame с добавленным прогнозом модели стекинга
    """
    
    # Подготовка даты прогноза
    pred_month_end = prediction_date + MonthEnd(0)
    
    # Разделение на обучающую и тестовую выборки
    train_data = all_models[features_for_stacking].loc[all_models[date_column] < pred_month_end]
    test_data = all_models[features_for_stacking].loc[all_models[date_column] == pred_month_end]
    
    # Очистка и проверка данных
    X_train = train_data.drop(columns=[date_column, target_column]).dropna(subset=[ts_prediction_column])
    y_train = train_data.loc[X_train.index, target_column]
    
    if len(X_train) == 0:
        raise ValueError("Нет данных для обучения. Проверьте фильтры и даты.")
    
    # Обучение модели
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    
    # Прогнозирование
    X_test = test_data.drop(columns=[date_column, target_column])
    if not X_test.empty:
        y_pred = model.predict(X_test)
        all_models.loc[
            (all_models[date_column] == pred_month_end) &
            (all_models["Статья"] == target_article),
            "predict_stacking_RFR"
        ] = y_pred
    
    return all_models