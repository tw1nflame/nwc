import pandas as pd
import logging
from utils.common import catch_errors

logger = logging.getLogger(__name__)

@catch_errors
def load_and_transform_data(filepath, DATE_COLUMN, RATE_COLUMN):
    """
    Загружает и преобразует финансовые данные из Excel файла.
    
    Параметры:
    filepath (str):    Путь к Excel файлу с данными
    DATE_COLUMN (str): Название столбца, отражающего дату в Excel файле
    RATE_COLUMN (str): Название столбца, отражающего курс USD в Excel файле
    
    Возвращает:
    pd.DataFrame: Преобразованный DataFrame с финансовыми данными
    """
    logger.info(f"Loading and transforming data from {filepath}")
    # Загрузка данных
    df = pd.read_excel(filepath)
    logger.info("Data loaded successfully")

    # Транспонирование и обработка заголовков
    df = (
        df.T
        .reset_index()
        .pipe(lambda x: x.set_axis(x.iloc[0], axis=1))
        .iloc[1:]
        .reset_index(drop=True)
    )

    # Преобразование типов данных
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    numeric_cols = df.columns.drop(DATE_COLUMN)
    df[numeric_cols] = df[numeric_cols].astype(float)

    # Конвертация в рубли (исключая курс)
    items_to_convert = numeric_cols.drop(RATE_COLUMN)
    df[items_to_convert] = df[items_to_convert].multiply(df[RATE_COLUMN], axis='index')

    # Очистка названий столбцов
    df.columns = [
        col.strip()[1:] if col.strip().startswith('-') else col.strip() 
        for col in df.columns
    ]

    # Заполнение пропусков
    df = df.fillna(0)
    logger.info("Data transformation completed successfully")
    return df