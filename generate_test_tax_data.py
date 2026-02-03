import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import json

def generate_historical_data(filename="test_tax_history.xlsx"):
    print(f"Generating {filename}...")
    
    # Create dates (columns) - Month Year format (e.g. "январь 2020")
    dates_dt = pd.date_range(start='2020-01-01', periods=48+24, freq='M')
    month_names = {
        1: 'январь', 2: 'февраль', 3: 'март', 4: 'апрель', 5: 'май', 6: 'июнь',
        7: 'июль', 8: 'август', 9: 'сентябрь', 10: 'октябрь', 11: 'ноябрь', 12: 'декабрь'
    }
    dates = [f"{month_names[d.month]} {d.year}" for d in dates_dt]
    
    # Create rows
    # We need rows that match the filters in data_preparation.py
    # Columns: Счет, Классификация, Компания, [Dates...]
    
    data = []
    
    # Helper to add row
    def add_row(acc, cls, comp):
        row = [acc, cls, comp] + list(np.random.uniform(-1000000, 1000000, len(dates)))
        data.append(row)

    companies = [
        'E110000 - ПАО "ГМК "Норильский никель"',
        'E120200 - АО "Кольская ГМК"',
        'E133700 - ООО "ГРК "Быстринское"',
        'Other Company 1',
        'Other Company 2'
    ]

    # 1. НДС (Active)
    # AC180000 / LTTX101
    for comp in companies:
        add_row('AC180000 - Краткосрочная предоплата по налогам и сборам', 'LTTX101 - НДС по приобретенным основным средствам', comp)
        add_row('LC210100 - Основная сумма по налогам и сборам', 'LTTX301 - НДС - основная сумма (сч. 68)', comp)

    # 2. Налог на прибыль
    for comp in companies:
        add_row('AC180000 - Краткосрочная предоплата по налогам и сборам', 'LTTX201 - Налог на прибыль', comp)
        add_row('LC210100 - Основная сумма по налогам и сборам', 'LTTX201 - Налог на прибыль', comp)

    # 3. НДПИ
    for comp in companies:
        add_row('AC180000 - Краткосрочная предоплата по налогам и сборам', 'LTTX305 - Налог на добычу полезных ископаемых', comp)
        add_row('LC210100 - Основная сумма по налогам и сборам', 'LTTX305 - Налог на добычу полезных ископаемых', comp)

    # Create DataFrame
    columns = ['Счет', 'Классификация', 'Компания'] + dates
    df = pd.DataFrame(data, columns=columns)
    
    # Create Excel with specific structure (skiprows=9, header at row 13 (index 12))
    # We need to pad the top
    
    # Create a writer
    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        # Write empty rows
        # We need the header to be at Excel Row 12 (index 11)
        # So we write the dataframe starting at row 11
        # startrow=12 means Excel Row 13.
        df.to_excel(writer, sheet_name='Налоги_RUB_комп_value', startrow=12, index=False)


def generate_forecast_files(output_dir="generated_forecasts"):
    print(f"Generating forecast files in {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Define groups and companies (clusters)
    # Based on forecast.py and data_preparation.py logic
    clusters = [
        ('НДС', 'E110000 - ПАО "ГМК "Норильский никель"'),
        ('НДС', 'E120200 - АО "Кольская ГМК"'),
        ('НДС', 'Прочие компании'),
        
        ('Налог на прибыль', 'E110000 - ПАО "ГМК "Норильский никель"'),
        ('Налог на прибыль', 'E120200 - АО "Кольская ГМК"'),
        ('Налог на прибыль', 'E133700 - ООО "ГРК "Быстринское"'),
        ('Налог на прибыль', 'Прочие компании'),
        
        ('НДПИ', 'E110000 - ПАО "ГМК "Норильский никель"'),
        ('НДПИ', 'Прочие компании'),
        
        ('НВОС', 'Прочие компании'),
        ('НДФЛ', 'Прочие компании'),
        ('Страховые взносы', 'Прочие компании'),
        ('Налог на сверхприбыль', 'Прочие компании'),
    ]
    
    # Dates for forecast
    dates_dt = pd.date_range(start='2020-01-01', periods=48+24, freq='M')
    # Use datetime.date objects to ensure no time component in Excel
    dates = [d.date() for d in dates_dt]
    
    for factor, item_id in clusters:
        # Clean filename
        safe_item_id = item_id.replace('"', '').replace(':', '')
        filename = f"{factor}_{safe_item_id}_predict_BASE.xlsx"
        filepath = os.path.join(output_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='xlsxwriter') as writer:
            # 1. Data Sheet
            data_columns = {
                'Дата': dates,
                'Разница Активов и Пассивов_fact': np.random.uniform(-1000, 1000, len(dates)),
            }
            
            models = [
                'naive', 'autoARIMA', 'ML', 'TFT', 'Chronos_base',
                'svm6', 'svm9', 'svm12',
                'linreg6', 'linreg9', 'linreg12',
                'linreg_no_intercept6', 'linreg_no_intercept9', 'linreg_no_intercept12',
                'stacking'
            ]
            
            for model in models:
                data_columns[f'Разница Активов и Пассивов_predict_{model}'] = np.random.uniform(-1000, 1000, len(dates))
                data_columns[f'Разница Активов и Пассивов_predict_{model} разница'] = np.random.uniform(-100, 100, len(dates))
                data_columns[f'Разница Активов и Пассивов_predict_{model} отклонение %'] = np.random.uniform(0, 100, len(dates))

            df_data = pd.DataFrame(data_columns)
            df_data.to_excel(writer, sheet_name='data', index=False)
            
            # 2. Coeffs Sheet
            df_coeffs = pd.DataFrame({
                'Дата': dates,
                'feature_1': np.random.uniform(0, 1, len(dates)),
                'feature_2': np.random.uniform(0, 1, len(dates))
            })
            df_coeffs.to_excel(writer, sheet_name='coeffs', index=False)
            
            # 3. Coeffs No Intercept Sheet
            df_coeffs_ni = pd.DataFrame({
                'Дата': dates,
                'feature_1': np.random.uniform(0, 1, len(dates)),
                'feature_2': np.random.uniform(0, 1, len(dates))
            })
            df_coeffs_ni.to_excel(writer, sheet_name='coeffs_no_intercept', index=False)
            
            # 4. Ensemble Info Sheet
            ensemble_data = []
            for d in dates:
                ensemble_data.append({
                    'Дата': d,
                    'Статья': 'Target_A',
                    'Ансамбль': str({'ModelA': 0.6, 'ModelB': 0.4})
                })
            df_ensemble = pd.DataFrame(ensemble_data)
            df_ensemble.to_excel(writer, sheet_name='TimeSeries_ensemble_models_info', index=False)
            
        print(f"Created {filename}")

if __name__ == "__main__":
    generate_historical_data()
    generate_forecast_files()
