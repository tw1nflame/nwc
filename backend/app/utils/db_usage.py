import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from utils.column_mapping import REVERSE_COLUMN_MAPPING, COLUMN_MAPPING
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


CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../config_refined.yaml'))

def upload_pipeline_result_to_db(file_path: str, sheet_to_table: dict, date_value: str = None, date_column: str = None):
    """
    Загружает все листы результата pipeline (BASE или BASE+) в соответствующие таблицы PostgreSQL через psycopg2.
    Если листы Tabular_ensemble_models_info или Tabular_feature_importance отсутствуют, 
    удаляет данные по указанной дате из соответствующих таблиц.
    
    Args:
        file_path: путь к Excel файлу
        sheet_to_table: маппинг листов к таблицам
        date_value: дата для удаления данных из отсутствующих листов (формат 'YYYY-MM-DD')
        date_column: имя столбца с датой в Excel (например, 'Дата'), будет преобразовано в имя столбца БД
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
        # Очищаем только те таблицы, для которых есть соответствующие листы в Excel
        for sheet, df in xls.items():
            table = sheet_to_table.get(sheet)
            if table:
                if table == DATA_TABLE:
                    # Для основной таблицы сохраняем корректировки перед очисткой
                    cur.execute(f'CREATE TEMP TABLE temp_adjustments AS SELECT "дата", "статья", adjustments FROM {table} WHERE adjustments IS NOT NULL AND adjustments != 0')
                cur.execute(f'TRUNCATE TABLE {table} RESTART IDENTITY CASCADE')
        # Загружаем данные из Excel
        for sheet, df in xls.items():
            table = sheet_to_table.get(sheet)
            if table is None:
                continue
            if df.empty:
                continue
            df.columns = [c.lower() for c in df.columns]
            
            # Для основной таблицы добавляем столбец adjustments если его нет
            if table == DATA_TABLE and 'adjustments' not in df.columns:
                df['adjustments'] = 0
            
            columns = ', '.join(df.columns)
            values_template = ', '.join(['%s'] * len(df.columns))
            insert_query = f'INSERT INTO {table} ({columns}) VALUES ({values_template})'
            records = df.values.tolist()
            for row in records:
                cur.execute(insert_query, row)
            
            # После загрузки основной таблицы восстанавливаем корректировки
            if table == DATA_TABLE:
                try:
                    cur.execute(f'''
                        UPDATE {table} 
                        SET adjustments = temp_adjustments.adjustments 
                        FROM temp_adjustments 
                        WHERE {table}."дата" = temp_adjustments."дата" 
                        AND {table}."статья" = temp_adjustments."статья"
                    ''')
                    cur.execute('DROP TABLE IF EXISTS temp_adjustments')
                except:
                    # Игнорируем ошибки если временная таблица не создана
                    pass
        
        # Если листы Tabular_ensemble_models_info или Tabular_feature_importance отсутствуют,
        # удаляем данные по указанной дате из соответствующих таблиц
        if date_value and date_column:
            # Преобразуем имя столбца из Excel в имя столбца БД
            db_date_column = COLUMN_MAPPING.get(date_column, date_column.lower())
            tabular_sheets_to_check = ['Tabular_ensemble_models_info', 'Tabular_feature_importance']
            for sheet_name in tabular_sheets_to_check:
                if sheet_name not in xls:  # Лист отсутствует в Excel
                    table = sheet_to_table.get(sheet_name)
                    if table:
                        # Удаляем данные по дате из таблицы, используя правильное имя столбца БД
                        delete_query = f'DELETE FROM {table} WHERE {db_date_column} = %s'
                        cur.execute(delete_query, (date_value,))
        
        conn.commit()
        cur.close()
    finally:
        conn.close()

def set_pipeline_column(date_column: str, date_value: str, pipeline_value: str):
    """
    Устанавливает значение pipeline в DATA_TABLE для заданной даты.
    date_column: имя столбца с датой в Excel (например, 'Дата'), будет преобразовано в имя столбца БД
    date_value: значение даты (строка в формате 'YYYY-MM-DD')
    pipeline_value: 'BASE' или 'BASE+'
    """
    table = DATA_TABLE
    # Преобразуем имя столбца из Excel в имя столбца БД
    db_date_column = COLUMN_MAPPING.get(date_column, date_column.lower())
    
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
        WHERE {db_date_column} = %s
        """
        cur.execute(query, (pipeline_value, date_value))
        conn.commit()
        cur.close()
    finally:
        conn.close()

from io import BytesIO

def process_adjustments_file(file_path: str, date_column: str = 'Дата'):
    """
    Обрабатывает файл корректировок и обновляет столбец adjustments в DATA_TABLE.
    Полностью перезаписывает все корректировки.
    
    Args:
        file_path: путь к Excel файлу с корректировками
        date_column: имя столбца с датой в Excel (например, 'Дата')
    """
    # Читаем файл корректировок
    df_adjustments = pd.read_excel(file_path, sheet_name='Корректировки')
    
    # Проверяем обязательные столбцы
    required_columns = ['Статья', 'Год', 'Месяц', 'Корректировка, руб']
    missing_columns = [col for col in required_columns if col not in df_adjustments.columns]
    if missing_columns:
        raise ValueError(f"Отсутствуют обязательные столбцы в файле корректировок: {missing_columns}")
    
    # Формируем дату из года и месяца (последний день месяца)
    df_adjustments['Дата'] = pd.to_datetime(
        df_adjustments['Год'].astype(str) + '-' + 
        df_adjustments['Месяц'].astype(str).str.zfill(2) + '-01'
    ) + pd.offsets.MonthEnd(0)
    
    # Группируем корректировки по дате и статье, суммируем значения
    adjustments_grouped = df_adjustments.groupby(['Дата', 'Статья'])['Корректировка, руб'].sum().reset_index()
    
    # Преобразуем дату в формат для БД
    adjustments_grouped['Дата'] = adjustments_grouped['Дата'].dt.date
    
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )
    
    try:
        cur = conn.cursor()
        
        # Преобразуем имя столбца даты из Excel в имя столбца БД
        db_date_column = COLUMN_MAPPING.get(date_column, date_column.lower())
        
        # Сначала обнуляем все корректировки
        cur.execute(f'UPDATE {DATA_TABLE} SET adjustments = 0')
        
        # Обновляем корректировки для каждой записи
        for _, row in adjustments_grouped.iterrows():
            adjustment_value = row['Корректировка, руб']
            date_value = row['Дата']
            article_value = row['Статья']
            
            update_query = f"""
            UPDATE {DATA_TABLE} 
            SET adjustments = %s 
            WHERE {db_date_column} = %s AND "статья" = %s
            """
            cur.execute(update_query, (adjustment_value, date_value, article_value))
        
        conn.commit()
        cur.close()
        
        return {
            "status": "success", 
            "processed_adjustments": len(adjustments_grouped),
            "message": f"Обработано {len(adjustments_grouped)} корректировок"
        }
        
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

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
