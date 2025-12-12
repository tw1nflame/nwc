import os
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
import logging
from pandas.tseries.offsets import MonthEnd
from datetime import datetime
from functools import reduce
from utils.pipelines import predict_individual, get_naive_predict, get_svr_predict, get_linreg_with_bias_predict, get_linreg_without_bias_predict, get_RFR_predict
from utils.common import generate_monthly_period

# Get absolute path to pretrained_models directory
# This file is in backend/app/taxes/
# We want backend/app/pretrained_models/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETRAINED_MODELS_DIR = os.path.join(BASE_DIR, 'pretrained_models')

group_companies = {
    'НДС': [
        'E110000 - ПАО "ГМК "Норильский никель"',
        'E120200 - АО "Кольская ГМК"',
        'Прочие компании'
    ],
    'Налог на прибыль': [
        'E110000 - ПАО "ГМК "Норильский никель"',
        'E120200 - АО "Кольская ГМК"',
        'E133700 - ООО "ГРК "Быстринское"',
        'Прочие компании'
    ],
    'НДПИ': [
        'E110000 - ПАО "ГМК "Норильский никель"',
        'Прочие компании' 
    ],
    'НВОС': ['Прочие компании'],
    'НДФЛ': ['Прочие компании'], 
    'Прочее': ['Прочие компании'],
    'Страховые взносы': ['Прочие компании'], 
}

def forecast_taxes(CHOSEN_MONTH, group_companies, progress_callback=None):
    CHOSEN_MONTH_END = CHOSEN_MONTH + MonthEnd(0)
    MONTHES_TO_PREDICT = [CHOSEN_MONTH]


    for factor, item_ids in group_companies.items():
        for item_id in item_ids:
            if progress_callback:
                progress_callback(f"{factor} | {item_id}")
            
            standart_cols = [
                "Дата"
            ]
            
            tech_cols = [
                'item_id',
                'factor'
            ]
            
            FEATURES_FOR_LAG1 = {
                'Разница Активов и Пассивов': [
                    'Разница Активов и Пассивов'
                ]
            }
            
            TRANSFORM_DATE_COLUMN= True
            
            features_to_use = []
            
            TARGET_COLUMN = 'Разница Активов и Пассивов'

            models_to_use = {
                'NaiveModel': {},
                'SeasonalNaiveModel': {},
                'AverageModel': {},
                'SeasonalAverageModel': {},
                'ZeroModel': {},
                'AutoARIMAModel': {},
                'AutoETSModel': {},
                'ThetaModel': {},
                'CrostonModel': {},
                'NPTSModel': {},
                'DeepARModel': {},
                'PatchTSTModel': {},
                'TemporalFusionTransformerModel': {},
                'TiDEModel': {},
                'DirectTabularModel': {},
                'RecursiveTabularModel': {},
                'Chronos': [
                    {'model_path': os.path.join(PRETRAINED_MODELS_DIR, 'chronos-bolt-base'), 'ag_args': {'name_suffix': 'ZeroShot'}},
                    {'model_path': os.path.join(PRETRAINED_MODELS_DIR, 'chronos-bolt-small'), 'ag_args': {'name_suffix': 'ZeroShot'}},
                    {'model_path': os.path.join(PRETRAINED_MODELS_DIR, 'chronos-bolt-small'), 'fine_tune': True, 'ag_args': {'name_suffix': 'FineTuned'}}
                ]
            }

            # Путь к результирующему файлу
            path = f'results/{TARGET_COLUMN}'
            file = f'{factor}_{item_id}_predict_BASE.xlsx'
            prev_result_file_name = f'{path}/{file}'.replace('"', '')
            os.makedirs(path, exist_ok=True)
            
            METRIC = 'MAE'
            ALL_MODELS_INFO = dict()
            
            print(f'---> ПРОГНОЗИРОВАНИЕ ПАРЫ: {factor} | {item_id} <---')
            features_to_use = []

            #===================== DATA READ ========================
            df = pd.read_excel('data/Налоги_generated_flatten_file_companies.xlsx')
            df["item_id"] = df['Кластер']
            df["factor"] = df['Группа']
            
            df = df.loc[(df['Группа'] == factor) & (df['Компания'] == item_id)]
            
            # for custom cols selection
            df = df.loc[:, tech_cols + standart_cols + features_to_use + [TARGET_COLUMN]]
            df['Дата'] = df['Дата'] + MonthEnd(0)
            
            # корректировка для НДПИ и страховые взносы за апрель 2024
            if factor == 'НДПИ':
                march_value = df.loc[df['Дата'] == datetime(2024, 3, 1) + MonthEnd(0), 'Разница Активов и Пассивов'].values[0]
                df.loc[df['Дата'] == datetime(2024, 4, 1) + MonthEnd(0), 'Разница Активов и Пассивов'] -= march_value
            
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
                features_to_use.append('Дата_year')
                features_to_use.append('Дата_month')

            #===================== PREDICTS =======================#
            naive_predict = get_naive_predict(df, TARGET_COLUMN)

            # AutoML TS
            df_predict, model_info = predict_individual(
                df=df, 
                MONTHES_TO_PREDICT=MONTHES_TO_PREDICT, 
                METRIC=METRIC, 
                FACTORS=[factor], 
                TARGETS=[TARGET_COLUMN],
                COMPANY="ALL", 
                delete_previous_models=True, 
                show_prediction_status=True,
                models=models_to_use
            )
            
            predict_ML = (
                df_predict.copy()
                .reset_index(drop=True)
                .drop(columns=["item_id", "factor"])
                .rename(columns={f"{TARGET_COLUMN}_predict": f"{TARGET_COLUMN}_predict_ML"})
            )
            
            # AutoARIMA
            df_predict, _ = predict_individual(
                df=df, 
                MONTHES_TO_PREDICT=MONTHES_TO_PREDICT, 
                METRIC=METRIC, 
                FACTORS=[factor], 
                TARGETS=[TARGET_COLUMN],
                COMPANY="ALL", 
                delete_previous_models=True, 
                show_prediction_status=True,
                models={'AutoARIMA': {}}
            )
            predict_autoARIMA = (
                df_predict.copy()
                .reset_index(drop=True)
                .drop(columns=["item_id", "factor"])
                .rename(columns={f"{TARGET_COLUMN}_predict": f"{TARGET_COLUMN}_predict_autoARIMA"})
            )

            # TFT
            df_predict, _ = predict_individual(
                df=df, 
                MONTHES_TO_PREDICT=MONTHES_TO_PREDICT, 
                METRIC=METRIC, 
                FACTORS=[factor], 
                TARGETS=[TARGET_COLUMN],
                COMPANY="ALL", 
                delete_previous_models=True, 
                show_prediction_status=True,
                models={'TemporalFusionTransformerModel': {}}
            )
            predict_TFT = (
                df_predict.copy()
                .reset_index(drop=True)
                .drop(columns=["item_id", "factor"])
                .rename(columns={f"{TARGET_COLUMN}_predict": f"{TARGET_COLUMN}_predict_TFT"})
            )

            # Chronos
            df_predict, _ = predict_individual(
                df=df, 
                MONTHES_TO_PREDICT=MONTHES_TO_PREDICT, 
                METRIC=METRIC, 
                FACTORS=[factor], 
                TARGETS=[TARGET_COLUMN],
                COMPANY="ALL", 
                delete_previous_models=True, 
                show_prediction_status=True,
                models={'Chronos': {'model_path': 'pretrained_models/chronos-bolt-base', 'ag_args': {'name_suffix': 'ZeroShot'}}}
            )
            predict_Chronos_base = (
                df_predict.copy()
                .reset_index(drop=True)
                .drop(columns=["item_id", "factor"])
                .rename(columns={f"{TARGET_COLUMN}_predict": f"{TARGET_COLUMN}_predict_Chronos_base"})
            )

            #===================== FACT - PREDICT ASSEMBLE =======================#
            fact = df.loc[:, ["Дата", TARGET_COLUMN]].rename(columns={TARGET_COLUMN: f"{TARGET_COLUMN}_fact"})
            all_models = reduce(
                    lambda left, right: pd.merge(left, right, on=["Дата"], how="outer"),
                    [fact, naive_predict, predict_autoARIMA, predict_ML, predict_TFT, predict_Chronos_base]
                )
            
            prev_month = CHOSEN_MONTH - MonthEnd(1)
            prev_month_fact_value = all_models.loc[all_models['Дата'] == prev_month, f"{TARGET_COLUMN}_fact"].values
            
            all_models = all_models.dropna(subset=[f"{TARGET_COLUMN}_predict_ML"]).reset_index(drop=True)
            all_models = all_models.copy()

            #===================== Загрузка исторических прогнозов и добавление/замена к ним новых =======================#
            prev_predicts = pd.read_excel(prev_result_file_name, sheet_name='data')
            cols_to_use = [col for col in prev_predicts.columns if ('разница' not in col) and ('отклонение' not in col)]
            prev_predicts = prev_predicts.loc[:, cols_to_use]
            prev_predicts['Дата'] = pd.to_datetime(prev_predicts['Дата'])
            
            if len(prev_month_fact_value) > 0 and not pd.isna(prev_month_fact_value[0]):
                mask = prev_predicts['Дата'] == prev_month
                if mask.any():
                    if pd.isna(prev_predicts.loc[mask, f"{TARGET_COLUMN}_fact"].values[0]):
                        prev_predicts.loc[mask, f"{TARGET_COLUMN}_fact"] = prev_month_fact_value[0]

            
            if prev_predicts['Дата'].isin([CHOSEN_MONTH_END]).any():
                prev_predicts = prev_predicts.loc[prev_predicts['Дата'] != CHOSEN_MONTH_END]
            all_models = pd.concat([prev_predicts, all_models])
            all_models = all_models.sort_values(by='Дата')

            prev_linreg_w_intercept = pd.read_excel(prev_result_file_name, sheet_name='coeffs')
            prev_linreg_w_intercept['Дата'] = pd.to_datetime(prev_linreg_w_intercept['Дата'])
            if prev_linreg_w_intercept['Дата'].isin([CHOSEN_MONTH_END, CHOSEN_MONTH]).any():
                prev_linreg_w_intercept = prev_linreg_w_intercept.loc[~prev_linreg_w_intercept['Дата'].isin([CHOSEN_MONTH_END, CHOSEN_MONTH])]

            prev_linreg_no_intercept = pd.read_excel(prev_result_file_name, sheet_name='coeffs_no_intercept')
            prev_linreg_no_intercept['Дата'] = pd.to_datetime(prev_linreg_no_intercept['Дата'])
            if prev_linreg_no_intercept['Дата'].isin([CHOSEN_MONTH_END, CHOSEN_MONTH]).any():
                prev_linreg_no_intercept = prev_linreg_no_intercept.loc[~prev_linreg_no_intercept['Дата'].isin([CHOSEN_MONTH_END, CHOSEN_MONTH])]

            prev_ensemble_info = pd.read_excel(prev_result_file_name, sheet_name='TimeSeries_ensemble_models_info')
            prev_ensemble_info['Дата'] = pd.to_datetime(prev_ensemble_info['Дата'])
            if prev_ensemble_info['Дата'].isin([CHOSEN_MONTH_END, CHOSEN_MONTH]).any():
                prev_ensemble_info = prev_ensemble_info.loc[~prev_ensemble_info['Дата'].isin([CHOSEN_MONTH_END, CHOSEN_MONTH])]

            #===================== Stacking =======================#
            WINDOWS = [6, 9, 12]
            
            MONTHS_TO_PREDICT_FOR_STACKING = generate_monthly_period(CHOSEN_MONTH)
            # SVM (SVR)
            all_models = get_svr_predict(WINDOWS, all_models, TARGET_COLUMN, MONTHS_TO_PREDICT_FOR_STACKING)

            # Linreg
            all_models, LINREG_WEIGHTS_DF = get_linreg_with_bias_predict(WINDOWS, all_models, TARGET_COLUMN, MONTHS_TO_PREDICT_FOR_STACKING)

            # Linreg no bias
            all_models, LINREG_NO_INTERCEPT_WEIGHTS_DF = get_linreg_without_bias_predict(WINDOWS, all_models, TARGET_COLUMN, MONTHS_TO_PREDICT_FOR_STACKING)

            # RFR stacking
            all_models = get_RFR_predict(WINDOWS, all_models, TARGET_COLUMN, MONTHS_TO_PREDICT_FOR_STACKING)

            # result
            pred_cols = [col for col in all_models.columns if col not in ('Дата', f'{TARGET_COLUMN}_fact')]
            
            #===================== METRICS CALCULATION =======================#
            for column in pred_cols:
                all_models[f'{column} разница'] = all_models[f'{TARGET_COLUMN}_fact'] - all_models[column]

            for column in pred_cols:
                all_models[f'{column} отклонение %'] = (all_models[f'{TARGET_COLUMN}_fact'] - all_models[column]) / all_models[f'{TARGET_COLUMN}_fact']

            #===================== ENSEMBLE INFO DATAFRAME =======================#
            data = model_info
            df_ensmbl_infos = []
            for date in data[factor]:
                for target in data[factor][date][item_id]:
                    dct = data[factor][date][item_id][target]["WeightedEnsemble"]["model_weights"]
                    for key, value in dct.items():
                        dct[key] = round(value, 4)
                    month_pred_info = pd.DataFrame({'Дата': [date], 'Статья': [target], 'Ансамбль': [dct]})
                    df_ensmbl_infos.append(month_pred_info)
            
            DF_ENSMBLE_INFO = pd.concat(df_ensmbl_infos).reset_index(drop=True)
            DF_ENSMBLE_INFO['Дата'] = pd.to_datetime(DF_ENSMBLE_INFO['Дата'])

            #===================== RESUTLS SAVING =======================#
            save_dataframes_to_excel(
                {
                    'data': all_models,
                    'coeffs': pd.concat([prev_linreg_w_intercept, LINREG_WEIGHTS_DF]),
                    'coeffs_no_intercept': pd.concat([prev_linreg_no_intercept, LINREG_NO_INTERCEPT_WEIGHTS_DF]),
                    'TimeSeries_ensemble_models_info': pd.concat([prev_ensemble_info, DF_ENSMBLE_INFO])
                },
                prev_result_file_name
            )
