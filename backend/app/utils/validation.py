from typing import List
import pandas as pd
import psycopg2
import logging
from utils.db_usage import DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DATA_TABLE, COLUMN_MAPPING
from utils.datahandler import load_and_transform_data

logger = logging.getLogger(__name__)

def check_historical_data_consistency(file_path: str, date_column: str, rate_column: str, forecast_date: str) -> List[str]:
    """
    Validates that the historical data in the uploaded file matches the data in the database
    up to the forecast date.
    
    Args:
        file_path: Path to the uploaded Excel file.
        date_column: Name of the date column in the Excel file.
        rate_column: Name of the rate column in the Excel file.
        forecast_date: The start date of the forecast (YYYY-MM-DD). Data before this date is checked.
        
    Returns:
        List of warning messages describing the mismatches.
    """
    warnings = []
    conn = None
    try:
        logger.info(f"Starting historical data validation. File: {file_path}, Forecast Date: {forecast_date}")
        
        # Load and transform uploaded data
        # This returns a DataFrame where columns are article names and one 'date_column' with Date objects
        # Values are in RUB (converted using rate)
        # Note: load_and_transform_data applies RATE multiplication.
        df_new = load_and_transform_data(file_path, date_column, rate_column)
        
        # Melt to long format for comparison
        article_cols = [c for c in df_new.columns if c != date_column]
        df_new_long = df_new.melt(id_vars=[date_column], value_vars=article_cols, var_name='article', value_name='fact_new')
        
        # Filter strictly before forecast_date
        forecast_dt = pd.to_datetime(forecast_date)
        df_new_long = df_new_long[df_new_long[date_column] < forecast_dt]
        
        if df_new_long.empty:
            logger.info("No historical data found in uploaded file before forecast date.")
            return warnings

        # Get DB data
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASSWORD,
            dbname=DB_NAME
        )
        
        # Get column names from mapping
        db_date_col = COLUMN_MAPPING.get('Дата', 'date')
        db_fact_col = COLUMN_MAPPING.get('Fact', 'fact')
        
        # We need to query by date. 
        # Check if table exists first? Usually it assumes existence.
        
        query = f"""
            SELECT article, "{db_date_col}" as date, "{db_fact_col}" as fact_db
            FROM {DATA_TABLE}
            WHERE "{db_date_col}" < %s
        """
        
        # Read into dataframe
        df_db = pd.read_sql_query(query, conn, params=(forecast_date,))
        if df_db.empty:
            logger.info("No historical data found in DB before forecast date.")
            # Nothing to compare against means no mismatch (or maybe user wants to know?)
            # Usually strict consistency means if DB has data, it must match. If DB has no data, we assume initial load?
            # User said: "sver' s tem chto v bd... if facts were substituted". If DB empty, no substitution.
            return warnings

        df_db['date'] = pd.to_datetime(df_db['date'])
        
        # Rename date column in new data for merge
        df_new_long = df_new_long.rename(columns={date_column: 'date'})
        
        # Merge on article and date
        merged = pd.merge(df_new_long, df_db, on=['article', 'date'], how='inner')
        
        # Compare
        # Allow small floating point difference (tolerance 1.0 unit)
        epsilon = 1.0
        
        mismatches = merged[abs(merged['fact_new'] - merged['fact_db']) > epsilon]
        
        if not mismatches.empty:
            logger.warning(f"Found {len(mismatches)} mismatches in historical data.")
            
            # Group by article to summarize
            limit = 5
            count = 0
            
            bad_articles = mismatches['article'].unique()
            
            for article in bad_articles:
                article_mismatches = mismatches[mismatches['article'] == article]
                dates = article_mismatches['date'].dt.strftime('%Y-%m-%d').tolist()
                
                # Format warning using first date
                msg = f"Несовпадение факта для '{article}' (даты: {', '.join(dates[:3])}"
                if len(dates) > 3:
                    msg += "..."
                msg += ")"
                
                # Add sample values
                row = article_mismatches.iloc[0]
                msg += f" [Файл: {row['fact_new']:.2f} != БД: {row['fact_db']:.2f}]"
                
                warnings.append(msg)
                count += 1
                if count >= limit:
                    warnings.append(f"... и еще по {len(bad_articles) - limit} статьям.")
                    break
        else:
            logger.info("Validation passed: File data matches DB data.")
            
    except Exception as e:
        logger.error(f"Error validating historical data: {e}", exc_info=True)
        # We append error as warning to inform user
        warnings.append(f"Ошибка проверки данных: {str(e)}")
    finally:
        if conn:
            conn.close()
            
    return warnings
