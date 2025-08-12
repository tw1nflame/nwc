import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from column_mapping import COLUMN_MAPPING
import re

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

# Используем тот же маппинг, что и в upload.py (для BASE+)
SHEET_TO_TABLE = {
    'data': DATA_TABLE,
    'coeffs_with_intercept': COEFFS_WITH_INTERCEPT_TABLE,
    'coeffs_no_intercept': COEFFS_NO_INTERCEPT_TABLE,
    'TimeSeries_ensemble_models_info': ENSEMBLE_INFO_TABLE,
    'Tabular_ensemble_models_info': BASEPLUS_TABULAR_ENSEMBLE_INFO_TABLE,
    'Tabular_feature_importance': BASEPLUS_FEATURE_IMPORTANCE_TABLE,
}

# Путь к синтетическому смешанному файлу (в корневой папке проекта в каталоге results)
EXCEL_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../../../results/synthetic_mixed_pipeline.xlsx')
)

# Загрузка всех листов Excel-файла
xls = pd.read_excel(EXCEL_FILE, sheet_name=None)

conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    user=DB_USER,
    password=DB_PASSWORD,
    dbname=DB_NAME
)
try:
    cur = conn.cursor()
    for sheet, df in xls.items():
        table_name = SHEET_TO_TABLE.get(sheet)
        if not table_name:
            print(f"Лист '{sheet}' не найден в маппинге, пропущен.")
            continue
        if df.empty:
            print(f"Лист '{sheet}' пустой, пропущен.")
            continue
        # Нормализуем имена колонок и применяем COLUMN_MAPPING
        normalized_cols = []
        for col in df.columns:
            col_str = str(col)
            # Сжимаем повторные пробелы и приводим 'отклонение  %' → 'отклонение %'
            col_str = re.sub(r"\s+", " ", col_str).strip()
            col_str = col_str.replace("отклонение  %", "отклонение %")
            normalized_cols.append(col_str)
        columns = [COLUMN_MAPPING.get(col, col) for col in normalized_cols]
        df.columns = columns
        col_types = []
        for col in columns:
            dtype = df[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                sql_type = 'INTEGER'
            elif pd.api.types.is_float_dtype(dtype):
                sql_type = 'DOUBLE PRECISION'
            elif pd.api.types.is_bool_dtype(dtype):
                sql_type = 'BOOLEAN'
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                sql_type = 'TIMESTAMP'
            else:
                sql_type = 'TEXT'
            col_types.append(f'"{col}" {sql_type}')
        # Удаляем таблицу, если она уже есть, чтобы структура всегда соответствовала Excel
        cur.execute(f'DROP TABLE IF EXISTS {table_name} CASCADE')
        conn.commit()
        create_table_sql = f"""
        CREATE TABLE {table_name} (
            {', '.join(col_types)}
        );
        """
        cur.execute(create_table_sql)
        conn.commit()
        # Полная очистка таблицы перед загрузкой
        try:
            cur.execute(f'TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE')
            conn.commit()
            print(f"Таблица {table_name} очищена (TRUNCATE RESTART IDENTITY CASCADE)")
        except Exception as e:
            # Если таблицы ещё не было или нет прав/ссылок — попытаемся обычный DELETE
            conn.rollback()
            try:
                cur.execute(f'DELETE FROM {table_name}')
                conn.commit()
                print(f"Таблица {table_name} очищена (DELETE)")
            except Exception as e2:
                conn.rollback()
                print(f"Не удалось очистить таблицу {table_name}: {e2}")
        columns_str = ', '.join([f'"{col}"' for col in columns])
        values_template = ', '.join(['%s'] * len(columns))
        insert_sql = f'INSERT INTO {table_name} ({columns_str}) VALUES ({values_template})'
        # Удалить полностью пустые строки
        df = df.dropna(how='all')
        # Заменить NaN на None для корректной вставки
        records = df.where(pd.notnull(df), None).values.tolist()
        for row in records:
            if len(row) != len(columns):
                print(f"Строка пропущена из-за несоответствия количества колонок: {row}")
                continue
            cur.execute(insert_sql, row)
        conn.commit()
        print(f"Таблица {table_name} создана и данные из листа '{sheet}' загружены.")
    cur.close()
finally:
    conn.close()
