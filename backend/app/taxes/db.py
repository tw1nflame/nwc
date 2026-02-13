import os
import psycopg2
import logging
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import io
import json
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
import tempfile
from ..utils.excel_formatter import save_dataframes_to_excel
from ..utils.column_mapping import COLUMN_MAPPING, REVERSE_COLUMN_MAPPING

load_dotenv()

EXCLUDED_FROM_DATA_TABLE = {
    'article', 'pipeline', 'ensemble', 'factor', 
    'feature', 'importance', 'stddev', 'p_value', 
    'n', 'p99_high', 'p99_low', 'adjustment_value', 
    'description', 'w_predict_ml_tabular', 'w_predict_tabpfnmix',
    
    # Exclude Tabular ML, PatchTST, and TabPFNMIX columns + their diffs/pcts
    'predict_ml_tabular', 'predict_ml_tabular_diff', 'predict_ml_tabular_pct',
    'predict_patchtst', 'predict_patchtst_diff', 'predict_patchtst_pct',
    'predict_tabpfnmix', 'predict_tabpfnmix_diff', 'predict_tabpfnmix_pct'
}

logger = logging.getLogger(__name__)

DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT', 5432))
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )

def get_db_engine():
    return create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

def init_db():
    """Creates the tax forecast tables if they don't exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1. Tax Forecast Unified Data Table
            # Base columns
            columns_sql = [
                "tax_item TEXT",
                "item_id TEXT",
                "date TIMESTAMP",
                "fact FLOAT",
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ]
            
            # Add metric columns from mapping
            metric_cols = set(COLUMN_MAPPING.values())
            if 'date' in metric_cols: metric_cols.remove('date')
            if 'fact' in metric_cols: metric_cols.remove('fact')
            
            # Remove excluded columns
            metric_cols = metric_cols - EXCLUDED_FROM_DATA_TABLE
            
            for col in sorted(metric_cols):
                if col:
                    columns_sql.append(f'"{col}" FLOAT')
            
            create_table_sql = f"""
                CREATE TABLE IF NOT EXISTS tax_forecast_data (
                    {', '.join(columns_sql)},
                    PRIMARY KEY (tax_item, item_id, date)
                );
            """
            cur.execute(create_table_sql)
            
            # 2. Coeffs
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tax_forecast_coeffs (
                    tax_item TEXT,
                    item_id TEXT,
                    date TIMESTAMP,
                    feature_name TEXT,
                    value FLOAT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tax_item, item_id, date, feature_name)
                );
            """)
            
            # 3. Coeffs no intercept
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tax_forecast_coeffs_no_intercept (
                    tax_item TEXT,
                    item_id TEXT,
                    date TIMESTAMP,
                    feature_name TEXT,
                    value FLOAT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tax_item, item_id, date, feature_name)
                );
            """)
            
            # 4. Ensemble info
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tax_forecast_ensemble_weights (
                    tax_item TEXT,
                    item_id TEXT,
                    date TIMESTAMP,
                    target TEXT,
                    model_name TEXT,
                    weight FLOAT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (tax_item, item_id, date, target, model_name)
                );
            """)
        conn.commit()
    except Exception as e:
        logger.error(f"Error initializing DB: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def parse_filename(filename: str):
    """
    Parses filename like 'Налог на прибыль_E110000 - ПАО ГМК Норильский никель_predict_BASE.xlsx'
    Returns (tax_item, item_id)
    """
    base = filename
    if '_predict_BASE.xlsx' in filename:
        base = filename.replace('_predict_BASE.xlsx', '')
    elif '_predict_BASE+.xlsx' in filename:
        base = filename.replace('_predict_BASE+.xlsx', '')
    elif '.xlsx' in filename:
        base = filename.replace('.xlsx', '')
    
    # Split by first underscore only
    parts = base.split('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

def save_excel_to_db(filename: str, file_content: bytes):
    logger.info(f"Начало сохранения файла в БД: {filename}")
    tax_item, item_id = parse_filename(filename)
    if not tax_item or not item_id:
        logger.error(f"Не удалось распарсить имя файла: {filename}")
        raise ValueError(f"Could not parse tax_item and item_id from filename: {filename}")
    
    logger.info(f"Распарсено: tax_item='{tax_item}', item_id='{item_id}'")

    # Read Excel
    try:
        xls = pd.ExcelFile(io.BytesIO(file_content))
        logger.info(f"Excel файл прочитан успешно. Листы: {xls.sheet_names}")
    except Exception as e:
        logger.error(f"Ошибка при чтении Excel файла: {e}")
        raise

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1. Process 'data' sheet -> tax_forecast_data
            if 'data' in xls.sheet_names:
                logger.info("Обработка листа 'data'...")
                df = pd.read_excel(xls, 'data')
                logger.info(f"Размер DataFrame 'data': {df.shape}")
                
                # 1. Strip prefix 'Разница Активов и Пассивов_'
                rename_map = {}
                for col in df.columns:
                    if col.startswith('Разница Активов и Пассивов_'):
                        rename_map[col] = col.replace('Разница Активов и Пассивов_', '')
                if rename_map:
                    df.rename(columns=rename_map, inplace=True)

                # 2. Map legacy/specific model names to standard names (keys in COLUMN_MAPPING)
                tax_model_mapping = {
                    'predict_linreg6': 'predict_linreg6_with_bias',
                    'predict_linreg9': 'predict_linreg9_with_bias',
                    'predict_linreg12': 'predict_linreg12_with_bias',
                    'predict_linreg_no_intercept6': 'predict_linreg6_no_bias',
                    'predict_linreg_no_intercept9': 'predict_linreg9_no_bias',
                    'predict_linreg_no_intercept12': 'predict_linreg12_no_bias',
                    'predict_ML': 'predict_TS_ML',
                    'predict_stacking': 'predict_stacking_RFR'
                }
                
                full_tax_mapping = {}
                for old, new in tax_model_mapping.items():
                    full_tax_mapping[old] = new
                    full_tax_mapping[f"{old} разница"] = f"{new} разница"
                    full_tax_mapping[f"{old} отклонение %"] = f"{new} отклонение %"
                
                df.rename(columns=full_tax_mapping, inplace=True)
                
                # 3. Apply standard mapping
                df.rename(columns=COLUMN_MAPPING, inplace=True)
                
                # Add metadata columns
                df['tax_item'] = tax_item
                df['item_id'] = item_id
                df['updated_at'] = datetime.now()
                
                # Convert NaNs to None
                df = df.where(pd.notnull(df), None)
                
                # Filter columns to ensure they exist in DB
                # Base known columns
                valid_columns = {'tax_item', 'item_id', 'date', 'fact', 'updated_at'}
                valid_columns.update(set(COLUMN_MAPPING.values()) - EXCLUDED_FROM_DATA_TABLE)
                
                # Only keep columns that are in valid_columns
                # Note: df.columns are already renamed using mapping
                columns_to_keep = [c for c in df.columns if c in valid_columns]
                
                if len(columns_to_keep) < len(df.columns):
                     skipped = set(df.columns) - set(columns_to_keep)
                     logger.warning(f"Пропущены колонки (нет в схеме БД): {skipped}")

                df = df[columns_to_keep]
                columns_to_insert = list(df.columns)
                
                cols_str = ', '.join([f'"{c}"' for c in columns_to_insert])
                
                # Exclude primary key columns from UPDATE
                update_set_list = [f'"{c}" = EXCLUDED."{c}"' for c in columns_to_insert if c not in ('tax_item', 'item_id', 'date')]
                update_set = ', '.join(update_set_list) if update_set_list else ""
                
                query = f"""
                    INSERT INTO tax_forecast_data ({cols_str})
                    VALUES %s
                    ON CONFLICT (tax_item, item_id, date) 
                    DO UPDATE SET {update_set}
                """
                
                data_values = [tuple(x) for x in df.to_numpy()]
                if data_values:
                    execute_values(cur, query, data_values)
                    logger.info(f"Вставлено/обновлено {len(data_values)} строк в tax_forecast_data")
            else:
                logger.warning("Лист 'data' не найден!")

            # 2. Process 'coeffs' sheet
            if 'coeffs' in xls.sheet_names:
                logger.info("Обработка листа 'coeffs'...")
                df = pd.read_excel(xls, 'coeffs')
                records = []
                for _, row in df.iterrows():
                    date = row['Дата']
                    for col in df.columns:
                        if col != 'Дата':
                            val = row[col]
                            if pd.notna(val):
                                records.append((tax_item, item_id, date, col, val, datetime.now()))
                
                if records:
                    unique_coeffs = {}
                    for rec in records:
                        key = (rec[0], rec[1], rec[2], rec[3])
                        unique_coeffs[key] = rec
                    records_unique = list(unique_coeffs.values())

                    execute_values(cur, """
                        INSERT INTO tax_forecast_coeffs (tax_item, item_id, date, feature_name, value, updated_at)
                        VALUES %s
                        ON CONFLICT (tax_item, item_id, date, feature_name) 
                        DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                    """, records_unique)
                    logger.info(f"Вставлено {len(records_unique)} записей в tax_forecast_coeffs")

            # 3. Process 'coeffs_no_intercept' sheet
            if 'coeffs_no_intercept' in xls.sheet_names:
                logger.info("Обработка листа 'coeffs_no_intercept'...")
                df = pd.read_excel(xls, 'coeffs_no_intercept')
                records = []
                for _, row in df.iterrows():
                    date = row['Дата']
                    for col in df.columns:
                        if col != 'Дата':
                            val = row[col]
                            if pd.notna(val):
                                records.append((tax_item, item_id, date, col, val, datetime.now()))
                
                if records:
                    unique_coeffs_ni = {}
                    for rec in records:
                        key = (rec[0], rec[1], rec[2], rec[3])
                        unique_coeffs_ni[key] = rec
                    records_unique = list(unique_coeffs_ni.values())

                    execute_values(cur, """
                        INSERT INTO tax_forecast_coeffs_no_intercept (tax_item, item_id, date, feature_name, value, updated_at)
                        VALUES %s
                        ON CONFLICT (tax_item, item_id, date, feature_name) 
                        DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                    """, records_unique)
                    logger.info(f"Вставлено {len(records_unique)} записей в tax_forecast_coeffs_no_intercept")

            # 4. Process 'TimeSeries_ensemble_models_info' sheet
            if 'TimeSeries_ensemble_models_info' in xls.sheet_names:
                logger.info("Обработка листа 'TimeSeries_ensemble_models_info'...")
                df = pd.read_excel(xls, 'TimeSeries_ensemble_models_info')
                records = []
                for _, row in df.iterrows():
                    date = row['Дата']
                    target = row['Статья']
                    info_val = row['Ансамбль']
                    
                    if isinstance(info_val, str):
                        try:
                            info_val = json.loads(info_val)
                        except json.JSONDecodeError:
                            try:
                                info_val = json.loads(info_val.replace("'", '"'))
                            except Exception as e:
                                pass 
                    
                    if isinstance(info_val, dict):
                        for model_name, weight in info_val.items():
                            records.append((tax_item, item_id, date, target, model_name, weight, datetime.now()))
                
                if records:
                    unique_weights = {}
                    for rec in records:
                        key = (rec[0], rec[1], rec[2], rec[3], rec[4])
                        unique_weights[key] = rec
                    records_unique = list(unique_weights.values())

                    execute_values(cur, """
                        INSERT INTO tax_forecast_ensemble_weights (tax_item, item_id, date, target, model_name, weight, updated_at)
                        VALUES %s
                        ON CONFLICT (tax_item, item_id, date, target, model_name) 
                        DO UPDATE SET weight = EXCLUDED.weight, updated_at = EXCLUDED.updated_at
                    """, records_unique)
                    logger.info(f"Вставлено {len(records_unique)} записей в tax_forecast_ensemble_weights")
                
        conn.commit()
        logger.info(f"Файл {filename} успешно сохранен в БД")
    except Exception as e:
        logger.error(f"Error saving excel to DB: {e}", exc_info=True)
        conn.rollback()
        raise
    finally:
        conn.close()

def restore_excel_from_db(tax_item: str, item_id: str) -> bytes:
    engine = get_db_engine()
    dataframes = {}
    
    try:
        with engine.connect() as conn:
            # 1. Data -> From tax_forecast_data
            df_data = pd.read_sql(text("""
                SELECT *
                FROM tax_forecast_data
                WHERE tax_item = :tax_item AND item_id = :item_id
            """), conn, params={"tax_item": tax_item, "item_id": item_id})
            
            if not df_data.empty:
                # 1. Drop metadata
                cols_to_drop = ['tax_item', 'item_id', 'updated_at']
                df_data.drop(columns=[c for c in cols_to_drop if c in df_data.columns], inplace=True)
                
                # 2. Reverse DB -> Standard Name
                df_data.rename(columns=REVERSE_COLUMN_MAPPING, inplace=True)
                
                # 3. Reverse Standard Name -> Legacy Tax Name
                tax_model_mapping = {
                    'predict_linreg6': 'predict_linreg6_with_bias',
                    'predict_linreg9': 'predict_linreg9_with_bias',
                    'predict_linreg12': 'predict_linreg12_with_bias',
                    'predict_linreg_no_intercept6': 'predict_linreg6_no_bias',
                    'predict_linreg_no_intercept9': 'predict_linreg9_no_bias',
                    'predict_linreg_no_intercept12': 'predict_linreg12_no_bias',
                    'predict_ML': 'predict_TS_ML',
                    'predict_stacking': 'predict_stacking_RFR'
                }
                
                tax_model_reverse = {}
                for old, new in tax_model_mapping.items():
                    tax_model_reverse[new] = old
                    tax_model_reverse[f"{new} разница"] = f"{old} разница"
                    tax_model_reverse[f"{new} отклонение %"] = f"{old} отклонение %"

                df_data.rename(columns=tax_model_reverse, inplace=True)
                
                # 4. Add Prefix 'Разница Активов и Пассивов_'
                prefix = 'Разница Активов и Пассивов_'
                final_rename = {}
                for col in df_data.columns:
                    if col == 'Дата': 
                        continue
                    if col == 'Fact': # 'Fact' comes from REVERSE_COLUMN_MAPPING
                         final_rename[col] = f'{prefix}fact' 
                    elif col == 'fact': # If it somehow stayed lowercase
                         final_rename[col] = f'{prefix}fact'
                    else:
                         final_rename[col] = f'{prefix}{col}'
                
                df_data.rename(columns=final_rename, inplace=True)
                
                if 'Дата' in df_data.columns:
                    df_data.sort_values('Дата', inplace=True)
                
                df_data.dropna(axis=1, how='all', inplace=True)
                
                # Упорядочиваем колонки: Дата, факт, все прогнозы (в порядке создания), все разницы, все отклонения
                ordered_cols = []
                
                # 1. Дата
                if 'Дата' in df_data.columns:
                    ordered_cols.append('Дата')
                
                # 2. Факт
                fact_cols = [col for col in df_data.columns if col.endswith('_fact')]
                ordered_cols.extend(fact_cols)
                
                # 3. Все predict колонки в порядке создания (как в forecast.py)
                # Порядок моделей: naive, autoARIMA, ML, TFT, Chronos_base, svm (6,9,12), linreg (6,9,12), linreg_no_intercept (6,9,12), stacking
                predict_order = [
                    'predict_naive',
                    'predict_autoARIMA',
                    'predict_ML',
                    'predict_TFT',
                    'predict_Chronos_base',
                    'predict_svm6',
                    'predict_svm9',
                    'predict_svm12',
                    'predict_linreg6',
                    'predict_linreg9',
                    'predict_linreg12',
                    'predict_linreg_no_intercept6',
                    'predict_linreg_no_intercept9',
                    'predict_linreg_no_intercept12',
                    'predict_stacking'
                ]
                
                # Собираем predict колонки в правильном порядке
                predict_cols = []
                for model_name in predict_order:
                    matching_cols = [col for col in df_data.columns 
                                    if model_name in col 
                                    and ' разница' not in col 
                                    and ' отклонение %' not in col]
                    predict_cols.extend(matching_cols)
                
                # Добавляем predict колонки которые не попали в список (на случай новых моделей)
                all_predict_cols = [col for col in df_data.columns 
                                   if 'predict' in col 
                                   and ' разница' not in col 
                                   and ' отклонение %' not in col
                                   and col not in predict_cols]
                predict_cols.extend(all_predict_cols)
                
                ordered_cols.extend(predict_cols)
                
                # 4. Все разницы в том же порядке что и предикты
                for pred_col in predict_cols:
                    diff_col = f"{pred_col} разница"
                    if diff_col in df_data.columns:
                        ordered_cols.append(diff_col)
                
                # 5. Все отклонения % в том же порядке что и предикты
                for pred_col in predict_cols:
                    pct_col = f"{pred_col} отклонение %"
                    if pct_col in df_data.columns:
                        ordered_cols.append(pct_col)
                
                # Добавляем оставшиеся колонки (если есть)
                remaining_cols = [col for col in df_data.columns if col not in ordered_cols]
                ordered_cols.extend(remaining_cols)
                
                # Применяем порядок
                df_data = df_data[ordered_cols]
                
                dataframes['data'] = df_data
            
            # 2. Coeffs
            df_coeffs_long = pd.read_sql(text("""
                SELECT date as "Дата", feature_name, value
                FROM tax_forecast_coeffs 
                WHERE tax_item = :tax_item AND item_id = :item_id
            """), conn, params={"tax_item": tax_item, "item_id": item_id})
            
            if not df_coeffs_long.empty:
                df_coeffs = df_coeffs_long.pivot(index='Дата', columns='feature_name', values='value').reset_index()
                df_coeffs = df_coeffs.sort_values('Дата')
                dataframes['coeffs'] = df_coeffs

            # 3. Coeffs no intercept
            df_coeffs_ni_long = pd.read_sql(text("""
                SELECT date as "Дата", feature_name, value
                FROM tax_forecast_coeffs_no_intercept 
                WHERE tax_item = :tax_item AND item_id = :item_id
            """), conn, params={"tax_item": tax_item, "item_id": item_id})
            
            if not df_coeffs_ni_long.empty:
                df_coeffs_ni = df_coeffs_ni_long.pivot(index='Дата', columns='feature_name', values='value').reset_index()
                df_coeffs_ni = df_coeffs_ni.sort_values('Дата')
                dataframes['coeffs_no_intercept'] = df_coeffs_ni

            # 4. Ensemble info
            df_weights_long = pd.read_sql(text("""
                SELECT date as "Дата", target as "Статья", model_name, weight
                FROM tax_forecast_ensemble_weights 
                WHERE tax_item = :tax_item AND item_id = :item_id
            """), conn, params={"tax_item": tax_item, "item_id": item_id})
            
            if not df_weights_long.empty:
                records = []
                grouped = df_weights_long.groupby(['Дата', 'Статья'])
                for (date, target), group in grouped:
                    weights = dict(zip(group['model_name'], group['weight']))
                    records.append({'Дата': date, 'Статья': target, 'Ансамбль': json.dumps(weights, ensure_ascii=False)})
                
                df_info = pd.DataFrame(records)
                df_info = df_info.sort_values('Дата')
                dataframes['TimeSeries_ensemble_models_info'] = df_info

        # Save using formatter
        with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
            tmp_path = tmp.name
            
        save_dataframes_to_excel(dataframes, tmp_path)
        
        with open(tmp_path, 'rb') as f:
            content = f.read()
            
        os.remove(tmp_path)
        return content

    except Exception as e:
        logger.error(f"Error restoring excel from DB: {e}")
        raise

def get_all_forecast_pairs():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT tax_item, item_id FROM tax_forecast_data")
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error getting forecast pairs: {e}")
        raise
    finally:
        conn.close()

