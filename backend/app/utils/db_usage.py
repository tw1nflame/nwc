import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from column_mapping import REVERSE_COLUMN_MAPPING
from utils.config import load_config

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT', 5432))
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

DATA_TABLE = os.getenv('DATA_TABLE')
COEFFS_WITH_INTERCEPT_TABLE = os.getenv('COEFFS_WITH_INTERCEPT_TABLE')
COEFFS_NO_INTERCEPT_TABLE = os.getenv('COEFFS_NO_INTERCEPT_TABLE')
ENSEMBLE_INFO_TABLE = os.getenv('ENSEMBLE_INFO_TABLE')
BASEPLUS_TABULAR_ENSEMBLE_INFO_TABLE = os.getenv('BASEPLUS_TABULAR_ENSEMBLE_INFO_TABLE')
BASEPLUS_FEATURE_IMPORTANCE_TABLE = os.getenv('BASEPLUS_FEATURE_IMPORTANCE_TABLE')

SHEET_TO_TABLE = {
    'data': DATA_TABLE,
    'coeffs_with_intercept': COEFFS_WITH_INTERCEPT_TABLE,
    'coeffs_no_intercept': COEFFS_NO_INTERCEPT_TABLE,
    'TimeSeries_ensemble_models_info': ENSEMBLE_INFO_TABLE,
    'Tabular_ensemble_models_info': BASEPLUS_TABULAR_ENSEMBLE_INFO_TABLE,
    'Tabular_feature_importance': BASEPLUS_FEATURE_IMPORTANCE_TABLE,
}

def upload_pipeline_result_to_db(file_path: str, sheet_to_table: dict):
    """
    Загружает все листы результата pipeline (BASE или BASE+) в соответствующие таблицы PostgreSQL через psycopg2.
    """
    xls = pd.read_excel(file_path, sheet_name=None)
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )
    try:
        cur = conn.cursor()
        # Очищаем все таблицы, даже если листа нет в Excel
        for sheet, table in sheet_to_table.items():
            if table:
                cur.execute(f'TRUNCATE TABLE {table} RESTART IDENTITY CASCADE')
        # Загружаем данные из Excel
        for sheet, df in xls.items():
            table = sheet_to_table.get(sheet)
            if table is None:
                continue
            if df.empty:
                continue
            df.columns = [c.lower() for c in df.columns]
            columns = ', '.join(df.columns)
            values_template = ', '.join(['%s'] * len(df.columns))
            insert_query = f'INSERT INTO {table} ({columns}) VALUES ({values_template})'
            records = df.values.tolist()
            for row in records:
                cur.execute(insert_query, row)
        conn.commit()
        cur.close()
    finally:
        conn.close()

def set_pipeline_column(date_column: str, date_value: str, pipeline_value: str):
    """
    Устанавливает значение pipeline в DATA_TABLE для заданной даты.
    date_column: имя столбца с датой (например, 'date' или 'дата')
    date_value: значение даты (строка в формате 'YYYY-MM-DD')
    pipeline_value: 'BASE' или 'BASE+'
    """
    table = DATA_TABLE
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )
    try:
        cur = conn.cursor()
        query = f"""
        UPDATE {table}
        SET pipeline = %s
        WHERE {date_column} = %s
        """
        cur.execute(query, (pipeline_value, date_value))
        conn.commit()
        cur.close()
    finally:
        conn.close()

from io import BytesIO

def export_pipeline_tables_to_excel(sheet_to_table: dict, make_final_prediction: bool = False) -> BytesIO:
    """
    Экспортирует все таблицы из sheet_to_table в Excel-файл в памяти и возвращает BytesIO.
    """
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )
    output = BytesIO()
    CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config_refined.yaml'))
    config = load_config(CONFIG_PATH)
    model_article = config.get('model_article', {})
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            if make_final_prediction:
                # Получаем данные из DATA_TABLE
                df_data = pd.read_sql_query(f'SELECT * FROM {DATA_TABLE}', conn)
                # Применяем маппинг к колонкам
                df_data.columns = [REVERSE_COLUMN_MAPPING.get(col, col) for col in df_data.columns]
                # Формируем финальный DataFrame
                final_rows = []
                for _, row in df_data.iterrows():
                    article = row.get('Статья')
                    model_name = model_article.get(article)
                    if not model_name:
                        continue
                    pred_col = f'predict_{model_name}'
                    # Найти колонку с учетом маппинга
                    mapped_pred_col = None
                    for col in df_data.columns:
                        if col.lower() == pred_col.lower():
                            mapped_pred_col = col
                            break
                    if not mapped_pred_col:
                        continue
                    final_rows.append({
                        'Дата': row.get('Дата'),
                        'Статья': article,
                        'Fact': row.get('Fact'),
                        'Прогноз': row.get(mapped_pred_col),
                        'Модель': model_name
                    })
                df_final = pd.DataFrame(final_rows, columns=['Дата', 'Статья', 'Fact', 'Прогноз', 'Модель'])
                df_final.to_excel(writer, index=False, sheet_name='final_prediction')
            # Остальные листы
            for sheet, table in sheet_to_table.items():
                if table is None:
                    continue
                df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
                df.columns = [REVERSE_COLUMN_MAPPING.get(col, col) for col in df.columns]
                df.to_excel(writer, index=False, sheet_name=sheet)
        output.seek(0)
        return output
    finally:
        conn.close()
