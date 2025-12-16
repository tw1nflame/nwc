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

load_dotenv()

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
            # 1. Data (Facts)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tax_forecast_facts (
                    factor TEXT,
                    item_id TEXT,
                    date TIMESTAMP,
                    fact_value FLOAT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (factor, item_id, date)
                );
            """)
            
            # 2. Predictions (Long format)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tax_forecast_predictions (
                    factor TEXT,
                    item_id TEXT,
                    date TIMESTAMP,
                    metric_name TEXT,
                    value FLOAT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (factor, item_id, date, metric_name)
                );
            """)
            
            # 3. Coeffs
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tax_forecast_coeffs (
                    factor TEXT,
                    item_id TEXT,
                    date TIMESTAMP,
                    feature_name TEXT,
                    value FLOAT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (factor, item_id, date, feature_name)
                );
            """)
            
            # 4. Coeffs no intercept
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tax_forecast_coeffs_no_intercept (
                    factor TEXT,
                    item_id TEXT,
                    date TIMESTAMP,
                    feature_name TEXT,
                    value FLOAT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (factor, item_id, date, feature_name)
                );
            """)
            
            # 5. Ensemble info
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tax_forecast_ensemble_weights (
                    factor TEXT,
                    item_id TEXT,
                    date TIMESTAMP,
                    target TEXT,
                    model_name TEXT,
                    weight FLOAT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (factor, item_id, date, target, model_name)
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
    Returns (factor, item_id)
    """
    base = filename.replace('_predict_BASE.xlsx', '').replace('_predict_BASE.xls', '')
    # Split by first underscore only
    parts = base.split('_', 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

def save_excel_to_db(filename: str, file_content: bytes):
    factor, item_id = parse_filename(filename)
    if not factor or not item_id:
        raise ValueError(f"Could not parse factor and item_id from filename: {filename}")
        
    # Read Excel
    xls = pd.ExcelFile(io.BytesIO(file_content))
    
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # 1. Process 'data' sheet
            if 'data' in xls.sheet_names:
                df = pd.read_excel(xls, 'data')
                
                # 1.1 Facts
                fact_records = []
                pred_records = []
                
                for _, row in df.iterrows():
                    date = row['Дата']
                    fact = row.get('Разница Активов и Пассивов_fact')
                    
                    if pd.notna(fact):
                        fact_records.append((factor, item_id, date, fact, datetime.now()))
                    
                    # 1.2 Predictions (all other columns)
                    for col in df.columns:
                        if col not in ['Дата', 'Разница Активов и Пассивов_fact']:
                            val = row[col]
                            if pd.notna(val):
                                pred_records.append((factor, item_id, date, col, val, datetime.now()))
                
                if fact_records:
                    # Remove duplicates from fact_records based on primary key (factor, item_id, date)
                    # Keep the last occurrence (most recent update)
                    unique_facts = {}
                    for rec in fact_records:
                        key = (rec[0], rec[1], rec[2]) # factor, item_id, date
                        unique_facts[key] = rec
                    fact_records_unique = list(unique_facts.values())

                    execute_values(cur, """
                        INSERT INTO tax_forecast_facts (factor, item_id, date, fact_value, updated_at)
                        VALUES %s
                        ON CONFLICT (factor, item_id, date) 
                        DO UPDATE SET fact_value = EXCLUDED.fact_value, updated_at = EXCLUDED.updated_at
                    """, fact_records_unique)
                
                if pred_records:
                    # Remove duplicates from pred_records based on primary key (factor, item_id, date, metric_name)
                    unique_preds = {}
                    for rec in pred_records:
                        key = (rec[0], rec[1], rec[2], rec[3]) # factor, item_id, date, metric_name
                        unique_preds[key] = rec
                    pred_records_unique = list(unique_preds.values())

                    execute_values(cur, """
                        INSERT INTO tax_forecast_predictions (factor, item_id, date, metric_name, value, updated_at)
                        VALUES %s
                        ON CONFLICT (factor, item_id, date, metric_name) 
                        DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                    """, pred_records_unique)

            # 2. Process 'coeffs' sheet
            if 'coeffs' in xls.sheet_names:
                df = pd.read_excel(xls, 'coeffs')
                records = []
                for _, row in df.iterrows():
                    date = row['Дата']
                    for col in df.columns:
                        if col != 'Дата':
                            val = row[col]
                            if pd.notna(val):
                                records.append((factor, item_id, date, col, val, datetime.now()))
                
                if records:
                    # Remove duplicates
                    unique_coeffs = {}
                    for rec in records:
                        key = (rec[0], rec[1], rec[2], rec[3]) # factor, item_id, date, feature_name
                        unique_coeffs[key] = rec
                    records_unique = list(unique_coeffs.values())

                    execute_values(cur, """
                        INSERT INTO tax_forecast_coeffs (factor, item_id, date, feature_name, value, updated_at)
                        VALUES %s
                        ON CONFLICT (factor, item_id, date, feature_name) 
                        DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                    """, records_unique)

            # 3. Process 'coeffs_no_intercept' sheet
            if 'coeffs_no_intercept' in xls.sheet_names:
                df = pd.read_excel(xls, 'coeffs_no_intercept')
                records = []
                for _, row in df.iterrows():
                    date = row['Дата']
                    for col in df.columns:
                        if col != 'Дата':
                            val = row[col]
                            if pd.notna(val):
                                records.append((factor, item_id, date, col, val, datetime.now()))
                
                if records:
                    # Remove duplicates
                    unique_coeffs_ni = {}
                    for rec in records:
                        key = (rec[0], rec[1], rec[2], rec[3]) # factor, item_id, date, feature_name
                        unique_coeffs_ni[key] = rec
                    records_unique = list(unique_coeffs_ni.values())

                    execute_values(cur, """
                        INSERT INTO tax_forecast_coeffs_no_intercept (factor, item_id, date, feature_name, value, updated_at)
                        VALUES %s
                        ON CONFLICT (factor, item_id, date, feature_name) 
                        DO UPDATE SET value = EXCLUDED.value, updated_at = EXCLUDED.updated_at
                    """, records_unique)

            # 4. Process 'TimeSeries_ensemble_models_info' sheet
            if 'TimeSeries_ensemble_models_info' in xls.sheet_names:
                df = pd.read_excel(xls, 'TimeSeries_ensemble_models_info')
                records = []
                for _, row in df.iterrows():
                    date = row['Дата']
                    target = row['Статья']
                    info_val = row['Ансамбль']
                    
                    if isinstance(info_val, str):
                        try:
                            # Try standard JSON first
                            info_val = json.loads(info_val)
                        except json.JSONDecodeError:
                            try:
                                # Fallback for single quotes (legacy data)
                                info_val = json.loads(info_val.replace("'", '"'))
                            except Exception as e:
                                logger.warning(f"Failed to parse ensemble info for {factor}|{item_id}|{date}: {e}")
                                pass 
                    
                    if isinstance(info_val, dict):
                        for model_name, weight in info_val.items():
                            records.append((factor, item_id, date, target, model_name, weight, datetime.now()))
                
                if records:
                    # Remove duplicates
                    unique_weights = {}
                    for rec in records:
                        key = (rec[0], rec[1], rec[2], rec[3], rec[4]) # factor, item_id, date, target, model_name
                        unique_weights[key] = rec
                    records_unique = list(unique_weights.values())

                    execute_values(cur, """
                        INSERT INTO tax_forecast_ensemble_weights (factor, item_id, date, target, model_name, weight, updated_at)
                        VALUES %s
                        ON CONFLICT (factor, item_id, date, target, model_name) 
                        DO UPDATE SET weight = EXCLUDED.weight, updated_at = EXCLUDED.updated_at
                    """, records_unique)
                
        conn.commit()
    except Exception as e:
        logger.error(f"Error saving excel to DB: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

def restore_excel_from_db(factor: str, item_id: str) -> bytes:
    engine = get_db_engine()
    output = io.BytesIO()
    try:
        with engine.connect() as conn:
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # 1. Data
                # Get facts
                df_facts = pd.read_sql(text("""
                    SELECT date as "Дата", fact_value as "Разница Активов и Пассивов_fact"
                    FROM tax_forecast_facts
                    WHERE factor = :factor AND item_id = :item_id
                """), conn, params={"factor": factor, "item_id": item_id})
                
                # Get predictions
                df_preds_long = pd.read_sql(text("""
                    SELECT date as "Дата", metric_name, value
                    FROM tax_forecast_predictions
                    WHERE factor = :factor AND item_id = :item_id
                """), conn, params={"factor": factor, "item_id": item_id})
                
                if not df_preds_long.empty:
                    df_preds = df_preds_long.pivot(index='Дата', columns='metric_name', values='value').reset_index()
                    # Merge
                    if not df_facts.empty:
                        df_data = pd.merge(df_facts, df_preds, on='Дата', how='outer')
                    else:
                        df_data = df_preds
                else:
                    df_data = df_facts
                    
                if not df_data.empty:
                    df_data = df_data.sort_values('Дата')
                    df_data.to_excel(writer, sheet_name='data', index=False)
                
                # 2. Coeffs
                df_coeffs_long = pd.read_sql(text("""
                    SELECT date as "Дата", feature_name, value
                    FROM tax_forecast_coeffs 
                    WHERE factor = :factor AND item_id = :item_id
                """), conn, params={"factor": factor, "item_id": item_id})
                
                if not df_coeffs_long.empty:
                    df_coeffs = df_coeffs_long.pivot(index='Дата', columns='feature_name', values='value').reset_index()
                    df_coeffs = df_coeffs.sort_values('Дата')
                    df_coeffs.to_excel(writer, sheet_name='coeffs', index=False)

                # 3. Coeffs no intercept
                df_coeffs_ni_long = pd.read_sql(text("""
                    SELECT date as "Дата", feature_name, value
                    FROM tax_forecast_coeffs_no_intercept 
                    WHERE factor = :factor AND item_id = :item_id
                """), conn, params={"factor": factor, "item_id": item_id})
                
                if not df_coeffs_ni_long.empty:
                    df_coeffs_ni = df_coeffs_ni_long.pivot(index='Дата', columns='feature_name', values='value').reset_index()
                    df_coeffs_ni = df_coeffs_ni.sort_values('Дата')
                    df_coeffs_ni.to_excel(writer, sheet_name='coeffs_no_intercept', index=False)

                # 4. Ensemble info
                df_weights_long = pd.read_sql(text("""
                    SELECT date as "Дата", target as "Статья", model_name, weight
                    FROM tax_forecast_ensemble_weights 
                    WHERE factor = :factor AND item_id = :item_id
                """), conn, params={"factor": factor, "item_id": item_id})
                
                if not df_weights_long.empty:
                    # Need to reconstruct the dict structure: {'model': weight, ...}
                    # Group by Date and Target
                    records = []
                    grouped = df_weights_long.groupby(['Дата', 'Статья'])
                    for (date, target), group in grouped:
                        weights = dict(zip(group['model_name'], group['weight']))
                        records.append({'Дата': date, 'Статья': target, 'Ансамбль': weights})
                    
                    df_info = pd.DataFrame(records)
                    df_info = df_info.sort_values('Дата')
                    df_info.to_excel(writer, sheet_name='TimeSeries_ensemble_models_info', index=False)

    except Exception as e:
        logger.error(f"Error restoring excel from DB: {e}")
        raise
        
    return output.getvalue()

def get_all_forecast_pairs():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT factor, item_id FROM tax_forecast_facts")
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Error getting forecast pairs: {e}")
        raise
    finally:
        conn.close()

