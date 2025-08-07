import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from typing import Sequence, Union, Any, List, Dict

import streamlit as st
import yaml
import os
from pathlib import Path

import logging
import sys

from functools import wraps

# Декоратор для перехвата ошибок
def catch_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(
                f"\nОШИБКА: {str(e)}\n",
                exc_info=True
            )
            st.error(f"Произошла ошибка: {str(e)}")
            st.stop()
    return wrapper

def setup_custom_logging(log_file="log.txt"):
    """Настройка логирования в файл и в консоль."""
    
    # Создаем папку logs если она не существует
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs'))
    os.makedirs(log_dir, exist_ok=True)
    
    # Полный путь к файлу логов
    log_file_path = os.path.join(log_dir, log_file)
    
    # Очищаем все предыдущие обработчики у корневого логгера
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Настраиваем форматтер
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Обработчик для записи в файл
    file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Обработчик для вывода в консоль (опционально)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Обработчик для stderr (перенаправляем sys.stderr в лог)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.WARNING)
    
    # Устанавливаем уровень логирования
    root_logger.setLevel(logging.DEBUG)
    
    # Добавляем обработчики
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(stderr_handler)

    # Ловим warnings (например, DeprecationWarning)
    logging.captureWarnings(True)

    # Перехватываем все необработанные исключения
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        root_logger.error(
            "НЕОБРАБОТАННОЕ ИСКЛЮЧЕНИЕ",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

    sys.excepthook = handle_uncaught_exception
    
    # Убираем лишние обработчики у логгеров AutoGluon
    ag_logger = logging.getLogger("autogluon")
    ag_logger.handlers = []  # Удаляем все обработчики AutoGluon
    ag_logger.propagate = True  # Разрешаем передачу в корневой логгер
    
    return root_logger

def setup_logging():
    # Создаем папку logs если она не существует
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs'))
    os.makedirs(log_dir, exist_ok=True)
    
    # Полный путь к файлу логов
    log_file_path = os.path.join(log_dir, 'log.txt')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path, mode='a', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.info("Logging initialized")

def normalize_to_list(targets: Union[Sequence[Any], str, int, float, bool]) -> List[Any]:
    """
    Нормализует входные данные к списку.
    
    Параметры:
        targets: Входные данные, которые нужно преобразовать в список.
                Может быть последовательностью (list, tuple и т.д.), 
                отдельной строкой или числом.
    
    Возвращает:
        List[Any]: Список с нормализованными данными.
        
    Примеры:
        >>> normalize_to_list("abc")
        ['abc']
        >>> normalize_to_list(123)
        [123]
        >>> normalize_to_list(["a", "b"])
        ['a', 'b']
        >>> normalize_to_list(("x", "y"))
        ['x', 'y']
        >>> normalize_to_list(None)
        [None]
    """
    if isinstance(targets, (str, int, float, bool)) or targets is None:
        return [targets]
    elif isinstance(targets, Sequence) and not isinstance(targets, str):
        return list(targets)
    else:
        return [targets]


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

def calculate_errors(all_models: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет абсолютные и относительные ошибки прогнозов.
    
    Параметры:
        all_models: DataFrame с фактическими значениями и прогнозами
    
    Возвращает:
        Модифицированный DataFrame с добавленными колонками ошибок
    """
    pred_cols = [col for col in all_models.columns if 'predict' in col]
    
    for column in pred_cols:
        # Абсолютная ошибка
        all_models[f'{column} разница'] = all_models['Fact'] - all_models[column]
    
    for column in pred_cols:
        # Относительная ошибка (в процентах)
        all_models[f'{column} отклонение %'] = (
            all_models[f'{column} разница'] / all_models['Fact']
        )
    
    return all_models

def extract_ensemble_info(data: Dict, 
                         factor: str, 
                         DATE_COLUMN: str) -> pd.DataFrame:
    """
    Извлекает информацию о весах ансамбля моделей.
    
    Параметры:
        data: Словарь с информацией о моделях
        factor: Фактор/ключ для доступа к данным
        DATE_COLUMN: Название столбца с датой
    
    Возвращает:
        DataFrame с информацией о весах ансамблей
    """
    ensemble_records = []
    
    for date in data.get(factor, {}):
        for target in data[factor][date].get(factor, {}):
            model_weights = data[factor][date][factor][target].get(
                "WeightedEnsemble", {}).get("model_weights", {})
            
            if model_weights:
                # Округляем веса и создаем запись
                rounded_weights = {k: round(v, 4) for k, v in model_weights.items()}
                record = {
                    DATE_COLUMN: date,
                    'Статья': target,
                    'Ансамбль': [rounded_weights]  # Сохраняем как список для DataFrame
                }
                ensemble_records.append(pd.DataFrame(record))
    
    if not ensemble_records:
        return pd.DataFrame(columns=[DATE_COLUMN, 'Статья', 'Ансамбль'])
    
    return pd.concat(ensemble_records).reset_index(drop=True)

def load_and_save_config():
    st.title("Конфигурация проекта")
    st.markdown("Загрузите новый конфигурационный файл (config_refined.yaml)")
    
    # Загрузка файла
    uploaded_file = st.file_uploader(
        "Выберите YAML файл конфигурации",
        type=["yaml"],
        key="config_uploader"
    )
    
    if uploaded_file is not None:
        try:
            # Чтение содержимого файла
            config_data = yaml.safe_load(uploaded_file)
            
            # Показать содержимое конфига (опционально)
            with st.expander("Просмотр конфигурации"):
                st.code(yaml.dump(config_data, allow_unicode=True), language='yaml')
            
            # Кнопка для сохранения
            if st.button("Сохранить конфигурацию"):
                # Определяем путь для сохранения
                config_path = Path("config_refined.yaml")
                
                # Сохраняем файл
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(config_data, f, allow_unicode=True)
                
                st.success(f"Конфигурация успешно сохранена в {config_path.resolve()}")
                
                # Показать обновленный файл
                if config_path.exists():
                    with st.expander("Просмотр сохраненного файла"):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            st.code(f.read(), language='yaml')
        
        except yaml.YAMLError as e:
            st.error(f"Ошибка чтения YAML файла: {str(e)}")
        except Exception as e:
            st.error(f"Произошла ошибка: {str(e)}")