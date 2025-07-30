import os
import re
import json
import scipy
import shutil
import numpy as np
import pandas as pd
import streamlit as st

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
from utils.pipelines import *

st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

logger = setup_custom_logging("log.txt")

with st.sidebar:
    with st.expander("tech user:"):
        st.text_input("username:", key="tech_user")
        st.text_input("password:", key="tech_password", type="password")
    if (st.session_state.get("tech_user", False) == "admin") and (st.session_state.get("tech_password", False) == "admin"):
        load_and_save_config()

        if os.path.exists('log.txt'):
            with open('log.txt', 'rb') as f:
                st.sidebar.download_button(
                    label='Download Logs',
                    data=f,
                    file_name='log.txt',
                    mime='text/plain'
                )


prediction_pipeline = st.radio(
    label="Выберите алгоритм прогноза",
    options=["BASE", "BASE+"],
    index=1
)

with st.spinner('Подготовка приложения...'):
    if st.session_state.get('init_clear_models', True):
        folders_to_clear = ['AutogluonModels'] # 'models', 
        for folder in folders_to_clear:
            if os.path.exists(folder):
                shutil.rmtree(folder)

        st.session_state['init_clear_models'] = False

# Config initial read
if not st.session_state.get('config', False):
    st.session_state['config'] = load_config('config_refined.yaml')

config = st.session_state['config']
DATE_COLUMN = config['DATE_COLUMN']
RATE_COLUMN = config['RATE_COLUMN']
ITEM_ID = config['item_id']
FACTOR = config['factor']
models_to_use = config['models_to_use']
TABPFNMIX_model = config['TABPFNMIX_model']
METRIC = config['metric_for_training']

input_items_to_predict = st.multiselect(
    label='Выберите статьи для прогноза',
    options=list(config['Статья'].keys()),
    default=config['Статья'].keys()
)

ITEMS_TO_PREDICT = {key: config['Статья'][key] for key in input_items_to_predict}
with st.expander("Пары Статья: Фичи"):
    st.write(ITEMS_TO_PREDICT)

# Monthes to predict and Metric to use
input_date = st.date_input(
    "Выберите месяц и год предикта (число можно игнорировать, возьмётся 1-е число)",
    value=datetime(year=2025, month=1, day=1)
)

# Приведём выбранную дату к 1 числу месяца (учитывая год/месяц из input_date)
CHOSEN_MONTH = datetime(input_date.year, input_date.month, 1)
st.write(f"Предиктивный месяц: {CHOSEN_MONTH.strftime('%B %Y')}")

MONTHES_TO_PREDICT = generate_monthly_period(CHOSEN_MONTH)

data_file = st.file_uploader(
    "Загрузите файл с данными (ЧОК исторические)", 
    key="data_file",
    type=["xlsm", "xlsx"]
)

prev_results_file = st.file_uploader(
    "Загрузите файл с предыдущими прогнозами", 
    key="prev_results_file",
    type=["xlsm", "xlsx"]
)

if st.button("Запустить расчёт"):
    logger.info("Calculation started")
    # Проверяем, что все файлы загружены
    if not data_file:
        logger.error("Data file not uploaded")
        st.error("Не загружен файл с данными")
        st.stop()

    if not prev_results_file:
        logger.error("Previous predicts file not uploaded")
        st.error("Не загружен файл с предыдущими прогнозами")
        st.stop()

    if f'{prediction_pipeline}.' not in prev_results_file.name:
        logger.error(f"Загружен неправильный файл для выбранного пайплайна {prediction_pipeline} -- {prev_results_file.name}")
        st.error(f"Загружен неправильный файл для выбранного пайплайна {prediction_pipeline} -- {prev_results_file.name}")
        st.stop()

    df_all_items = load_and_transform_data(data_file, DATE_COLUMN, RATE_COLUMN)
    st.session_state['result_file_name'] = f'predict_{prediction_pipeline}.xlsx'
    time_start = datetime.now()

    if prediction_pipeline == "BASE+":
        logger.info("Running BASE+ pipeline")
        with st.spinner("Прогнозирование по базовому+ методу..."):
            run_base_plus_pipeline(
                df_all_items=df_all_items,
                ITEMS_TO_PREDICT=ITEMS_TO_PREDICT,
                config=config,
                DATE_COLUMN=DATE_COLUMN,
                RATE_COLUMN=RATE_COLUMN,
                ITEM_ID=ITEM_ID,
                FACTOR=FACTOR,
                models_to_use=models_to_use,
                TABPFNMIX_model=TABPFNMIX_model,
                METRIC=METRIC,
                CHOSEN_MONTH=CHOSEN_MONTH,
                MONTHES_TO_PREDICT=MONTHES_TO_PREDICT,
                result_file_name=st.session_state['result_file_name'],
                prev_predicts_file=st.session_state['prev_results_file']
            )
    elif prediction_pipeline == "BASE":
        logger.info("Running BASE pipeline")
        with st.spinner("Прогнозирование по базовому методу..."):
            run_base_pipeline(
                df_all_items=df_all_items,
                ITEMS_TO_PREDICT=ITEMS_TO_PREDICT,
                config=config,
                DATE_COLUMN=DATE_COLUMN,
                RATE_COLUMN=RATE_COLUMN,
                ITEM_ID=ITEM_ID,
                FACTOR=FACTOR,
                models_to_use=models_to_use,
                METRIC=METRIC,
                CHOSEN_MONTH=CHOSEN_MONTH,
                MONTHES_TO_PREDICT=MONTHES_TO_PREDICT,
                result_file_name=st.session_state['result_file_name'],
                prev_predicts_file=st.session_state['prev_results_file']
            )

    time_finish = datetime.now()
    runtime = time_finish - time_start
    st.session_state['runtime'] = runtime
    logger.info(f"Pipeline completed in {runtime}")
    st.session_state['result_ready'] = True
    logger.info("Results ready")

if st.session_state.get('result_ready', False):
    st.success(f"Прогнозирование выполнено успешно. Время выполнения: {st.session_state['runtime']}")
    with open(f"results/{st.session_state['result_file_name']}", "rb") as result_file:
        res = result_file.read()
        st.download_button(
            label='Скачать прогнозный файл',
            data=res,
            file_name=st.session_state['result_file_name']
        )