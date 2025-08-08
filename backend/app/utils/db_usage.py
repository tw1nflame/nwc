import os
import pandas as pd
import psycopg2
import logging
from dotenv import load_dotenv
from utils.column_mapping import REVERSE_COLUMN_MAPPING, COLUMN_MAPPING
from utils.config import load_config
from utils.common import setup_custom_logging

load_dotenv()

# Создаем директорию для логов
log_dir = os.path.join(os.path.dirname(__file__), '..', 'logs')
os.makedirs(log_dir, exist_ok=True)

# Инициализируем логгер
setup_custom_logging(os.path.join(log_dir, "db_operations.log"))
logger = logging.getLogger(__name__)

DB_HOST = os.getenv('DB_HOST')
DB_PORT = int(os.getenv('DB_PORT', 5432))
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')
DB_NAME = os.getenv('DB_NAME')

DATA_TABLE = os.getenv('DATA_TABLE')
ADJUSTMENTS_TABLE = os.getenv('ADJUSTMENTS_TABLE')
EXCHANGE_RATE_TABLE = os.getenv('EXCHANGE_RATE_TABLE')
EXCHANGE_RATE_FILE_PATH = os.getenv('EXCHANGE_RATE_FILE_PATH')
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
    logger.info(f"Начало загрузки результатов пайплайна в БД из файла: {file_path}")
    logger.info(f"Параметры: date_value={date_value}, date_column={date_column}")
    
    xls = pd.read_excel(file_path, sheet_name=None)
    logger.info(f"Загружены листы Excel: {list(xls.keys())}")
    
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )
    logger.info(f"Подключение к БД установлено: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    try:
        cur = conn.cursor()
        
        # Получаем список статей, которые были обработаны (из листа 'data')
        processed_articles = []
        if 'data' in xls and not xls['data'].empty:
            # Применяем маппинг к колонкам для определения статей
            data_df = xls['data'].copy()
            data_df.columns = [COLUMN_MAPPING.get(col, col) for col in data_df.columns]
            if 'article' in data_df.columns:
                processed_articles = data_df['article'].unique().tolist()
                logger.info(f"Обработанные статьи: {processed_articles}")
            else:
                logger.warning("Колонка 'article' не найдена в данных после применения маппинга")
        else:
            logger.warning("Лист 'data' отсутствует или пуст")
        
        # Селективно удаляем данные только по обработанным статьям
        deleted_rows_total = 0
        for sheet, df in xls.items():
            table = sheet_to_table.get(sheet)
            if table and processed_articles:
                # Проверяем, есть ли в таблице колонка article
                cur.execute(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}' AND column_name = 'article'")
                if cur.fetchone():
                    # Удаляем данные только по обработанным статьям
                    placeholders = ', '.join(['%s'] * len(processed_articles))
                    delete_query = f'DELETE FROM {table} WHERE "article" IN ({placeholders})'
                    cur.execute(delete_query, processed_articles)
                    deleted_rows = cur.rowcount
                    deleted_rows_total += deleted_rows
                    logger.info(f"Удалено {deleted_rows} строк из таблицы {table} для статей: {processed_articles}")
                else:
                    # Если нет колонки article, очищаем всю таблицу (для совместимости)
                    cur.execute(f'TRUNCATE TABLE {table} RESTART IDENTITY CASCADE')
                    logger.info(f"Таблица {table} полностью очищена (нет колонки article)")
            elif table:
                logger.info(f"Таблица {table} пропущена (нет обработанных статей)")
                
        logger.info(f"Всего удалено строк: {deleted_rows_total}")
        
        # Загружаем новые данные из Excel
        inserted_rows_total = 0
        for sheet, df in xls.items():
            table = sheet_to_table.get(sheet)
            if table is None:
                logger.warning(f"Лист '{sheet}' не найден в маппинге таблиц")
                continue
            if df.empty:
                logger.warning(f"Лист '{sheet}' пуст")
                continue
            
            logger.info(f"Обработка листа '{sheet}' -> таблица '{table}', строк: {len(df)}")
            logger.info(f"Форма DataFrame: {df.shape} (строки x колонки)")
            logger.info(f"Исходные колонки ({len(df.columns)}): {list(df.columns)}")
            
            # Применяем маппинг колонок из Excel в БД
            original_columns = df.columns.tolist()
            df.columns = [COLUMN_MAPPING.get(col, col) for col in df.columns]
            mapped_columns = df.columns.tolist()
            logger.debug(f"Маппинг колонок для {sheet}: {dict(zip(original_columns, mapped_columns))}")
            logger.info(f"Колонки после маппинга ({len(df.columns)}): {list(df.columns)}")
            
            columns = ', '.join([f'"{col}"' for col in df.columns])
            values_template = ', '.join(['%s'] * len(df.columns))
            insert_query = f'INSERT INTO {table} ({columns}) VALUES ({values_template})'
            
            logger.info(f"SQL запрос: {insert_query}")
            logger.info(f"Ожидается {len(df.columns)} значений, колонки: {list(df.columns)}")
            
            records = df.values.tolist()
            logger.info(f"Первые 3 строки данных:")
            for i, row in enumerate(records[:3]):
                logger.info(f"  Строка {i+1}: {len(row)} значений - {row}")
            
            inserted_rows = 0
            for i, row in enumerate(records):
                try:
                    if len(row) != len(df.columns):
                        logger.error(f"Несоответствие количества значений в строке {i+1}:")
                        logger.error(f"  Ожидается: {len(df.columns)} значений")
                        logger.error(f"  Получено: {len(row)} значений")
                        logger.error(f"  Колонки ({len(df.columns)}): {list(df.columns)}")
                        logger.error(f"  Значения ({len(row)}): {row}")
                        raise ValueError(f"Количество значений ({len(row)}) не соответствует количеству колонок ({len(df.columns)})")
                    
                    cur.execute(insert_query, row)
                    inserted_rows += 1
                except Exception as e:
                    logger.error(f"Ошибка вставки строки {i+1} в таблицу {table}: {e}")
                    logger.error(f"Данные строки: {row}")
                    raise
            
            inserted_rows_total += inserted_rows
            logger.info(f"Вставлено {inserted_rows} строк в таблицу {table}")
        
        logger.info(f"Всего вставлено строк: {inserted_rows_total}")
        conn.commit()
        logger.info("Транзакция успешно завершена")
        
        # Если листы Tabular_ensemble_models_info или Tabular_feature_importance отсутствуют,
        # удаляем данные по указанной дате из соответствующих таблиц
        if date_value and date_column:
            # Преобразуем имя столбца из Excel в имя столбца БД
            db_date_column = COLUMN_MAPPING.get(date_column, date_column)
            tabular_sheets_to_check = ['Tabular_ensemble_models_info', 'Tabular_feature_importance']
            for sheet_name in tabular_sheets_to_check:
                if sheet_name not in xls:  # Лист отсутствует в Excel
                    table = sheet_to_table.get(sheet_name)
                    if table:
                        # Удаляем данные по дате из таблицы, используя правильное имя столбца БД
                        delete_by_date_query = f'DELETE FROM {table} WHERE "{db_date_column}" = %s'
                        cur.execute(delete_by_date_query, (date_value,))
                        deleted_by_date = cur.rowcount
                        logger.info(f"Удалено {deleted_by_date} строк из таблицы {table} по дате {date_value}")
            conn.commit()
            logger.info("Дополнительная очистка по дате завершена")
        
        cur.close()
        logger.info("Загрузка результатов пайплайна в БД завершена успешно")
        
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных в БД: {e}", exc_info=True)
        conn.rollback()
        logger.info("Транзакция отменена")
        raise e
    finally:
        conn.close()
        logger.info("Соединение с БД закрыто")

def set_pipeline_column(date_column: str, date_value: str, pipeline_value: str):
    """
    Устанавливает значение pipeline в DATA_TABLE для заданной даты.
    date_column: имя столбца с датой в Excel (например, 'Дата'), будет преобразовано в имя столбца БД
    date_value: значение даты (строка в формате 'YYYY-MM-DD')
    pipeline_value: 'BASE' или 'BASE+'
    """
    logger = setup_custom_logging()
    logger.info(f"Установка pipeline колонки: {date_column}={date_value} -> {pipeline_value}")
    
    table = DATA_TABLE
    # Преобразуем имя столбца из Excel в имя столбца БД
    db_date_column = COLUMN_MAPPING.get(date_column, date_column)
    logger.info(f"Маппинг колонки: {date_column} -> {db_date_column}")
    
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )
    logger.info(f"Подключение к БД для установки pipeline: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    try:
        cur = conn.cursor()
        query = f"""
        UPDATE {table}
        SET pipeline = %s
        WHERE "{db_date_column}" = %s
        """
        
        cur.execute(query, (pipeline_value, date_value))
        rows_affected = cur.rowcount
        logger.info(f"Выполнен UPDATE в таблице {table}: {rows_affected} строк обновлено")
        
        conn.commit()
        logger.info("Изменения pipeline зафиксированы в БД")
        cur.close()
        
    except Exception as e:
        logger.error(f"Ошибка при установке pipeline колонки: {e}", exc_info=True)
        conn.rollback()
        logger.info("Транзакция отменена")
        raise e
    finally:
        conn.close()
        logger.info("Соединение с БД для pipeline закрыто")

from io import BytesIO

def process_exchange_rate_file(file_path: str, rate_column: str = 'Курс', date_column: str = 'Дата'):
    """
    Обрабатывает файл с курсами валют и сохраняет их в таблицу EXCHANGE_RATE_TABLE.
    Полностью дропает таблицу и создает заново при каждой загрузке.
    
    Args:
        file_path: путь к Excel файлу с курсами валют
        rate_column: название столбца с курсом (по умолчанию 'Курс')
        date_column: название столбца с датой (по умолчанию 'Дата')
    
    Returns:
        dict: результат операции с количеством обработанных записей
    """
    logger.info(f"Начало обработки файла курсов валют: {file_path}")
    
    # Читаем Excel файл
    df = pd.read_excel(file_path)
    logger.info(f"Загружен Excel файл с курсами: {df.shape} (строки x колонки)")
    
    # Транспонируем DataFrame (аналогично datahandler.py)
    df = (
        df.T
        .reset_index()
        .pipe(lambda x: x.set_axis(x.iloc[0], axis=1))
        .iloc[1:]
        .reset_index(drop=True)
    )
    logger.info(f"DataFrame после транспонирования: {df.shape}")
    
    # Проверяем наличие нужных колонок
    if rate_column not in df.columns:
        raise ValueError(f"Колонка '{rate_column}' не найдена в файле")
    if date_column not in df.columns:
        raise ValueError(f"Колонка '{date_column}' не найдена в файле")
    
    # Оставляем только нужные колонки
    exchange_data = df[[date_column, rate_column]].copy()
    logger.info(f"Выбраны колонки для курса: {list(exchange_data.columns)}")
    
    # Преобразуем типы данных
    exchange_data[date_column] = pd.to_datetime(exchange_data[date_column])
    exchange_data[rate_column] = pd.to_numeric(exchange_data[rate_column], errors='coerce')
    
    # Удаляем строки с некорректными данными
    initial_rows = len(exchange_data)
    exchange_data = exchange_data.dropna()
    final_rows = len(exchange_data)
    
    if initial_rows != final_rows:
        logger.warning(f"Удалено {initial_rows - final_rows} строк с некорректными данными")
    
    logger.info(f"Подготовлено {len(exchange_data)} записей курса для сохранения")
    
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )
    logger.info(f"Подключение к БД для курсов установлено: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    try:
        cur = conn.cursor()
        
        # Проверяем существование таблицы курсов
        cur.execute(f"""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = '{EXCHANGE_RATE_TABLE}'
            )
        """)
        table_exists = cur.fetchone()[0]
        
        if table_exists:
            # Если таблица существует, дропаем её
            cur.execute(f'DROP TABLE {EXCHANGE_RATE_TABLE}')
            logger.info(f"Таблица {EXCHANGE_RATE_TABLE} удалена")
        else:
            logger.info(f"Таблица {EXCHANGE_RATE_TABLE} не существовала")
        
        # Создаем таблицу курсов заново
        create_table_sql = f"""
        CREATE TABLE {EXCHANGE_RATE_TABLE} (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL UNIQUE,
            exchange_rate DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_sql)
        logger.info(f"Таблица {EXCHANGE_RATE_TABLE} создана заново")
        
        # Вставляем курсы в таблицу (без указания id и created_at)
        inserted_rates = 0
        for _, row in exchange_data.iterrows():
            rate_value = row[rate_column]
            date_value = row[date_column].date()  # Конвертируем в date
            
            insert_query = f"""
            INSERT INTO {EXCHANGE_RATE_TABLE} (date, exchange_rate)
            VALUES (%s, %s)
            """
            try:
                cur.execute(insert_query, (date_value, rate_value))
                inserted_rates += 1
            except Exception as e:
                logger.error(f"Ошибка вставки курса для даты {date_value}: {e}")
                continue
        
        logger.info(f"Вставлено {inserted_rates} курсов в таблицу {EXCHANGE_RATE_TABLE}")
        conn.commit()
        logger.info("Курсы валют успешно сохранены")
        
        cur.close()
        
        return {
            "status": "success", 
            "processed_rates": inserted_rates,
            "message": f"Обработано {inserted_rates} курсов валют"
        }
        
    except Exception as e:
        logger.error(f"Ошибка при обработке курсов валют: {e}", exc_info=True)
        conn.rollback()
        logger.info("Транзакция курсов отменена")
        raise e
    finally:
        conn.close()
        logger.info("Соединение с БД для курсов закрыто")

def process_adjustments_file(file_path: str, date_column: str = 'Дата'):
    """
    Обрабатывает файл корректировок и сохраняет их в отдельную таблицу ADJUSTMENTS_TABLE.
    Полностью дропает таблицу и создает заново при каждой загрузке.
    
    Args:
        file_path: путь к Excel файлу с корректировками
        date_column: имя столбца с датой в Excel (например, 'Дата')
    """
    logger.info(f"Начало обработки файла корректировок: {file_path}")
    
    # Читаем файл корректировок
    df_adjustments = pd.read_excel(file_path, sheet_name='Корректировки')
    logger.info(f"Загружен лист 'Корректировки', строк: {len(df_adjustments)}")
    
    # Проверяем обязательные столбцы
    required_columns = ['Статья', 'Год', 'Месяц', 'Корректировка, руб']
    missing_columns = [col for col in required_columns if col not in df_adjustments.columns]
    if missing_columns:
        logger.error(f"Отсутствуют обязательные столбцы: {missing_columns}")
        raise ValueError(f"Отсутствуют обязательные столбцы в файле корректировок: {missing_columns}")
    
    logger.info(f"Все обязательные столбцы присутствуют: {required_columns}")
    
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
    logger.info(f"Подключение к БД для корректировок установлено: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    try:
        cur = conn.cursor()
        
        # Дропаем таблицу корректировок, если она существует
        cur.execute(f'DROP TABLE IF EXISTS {ADJUSTMENTS_TABLE}')
        logger.info(f"Таблица {ADJUSTMENTS_TABLE} удалена (если существовала)")
        
        # Создаем таблицу корректировок заново
        create_table_sql = f"""
        CREATE TABLE {ADJUSTMENTS_TABLE} (
            id SERIAL PRIMARY KEY,
            "date" DATE NOT NULL,
            "article" TEXT NOT NULL,
            adjustment_value DOUBLE PRECISION NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        cur.execute(create_table_sql)
        logger.info(f"Таблица {ADJUSTMENTS_TABLE} создана заново")
        
        # Вставляем корректировки в таблицу
        inserted_adjustments = 0
        for _, row in adjustments_grouped.iterrows():
            adjustment_value = row['Корректировка, руб']
            date_value = row['Дата']
            article_value = row['Статья']
            
            insert_query = f"""
            INSERT INTO {ADJUSTMENTS_TABLE} ("date", "article", adjustment_value)
            VALUES (%s, %s, %s)
            """
            cur.execute(insert_query, (date_value, article_value, adjustment_value))
            inserted_adjustments += 1
        
        logger.info(f"Вставлено {inserted_adjustments} корректировок в таблицу {ADJUSTMENTS_TABLE}")
        conn.commit()
        logger.info("Корректировки успешно сохранены")
        
        cur.close()
        
        return {
            "status": "success", 
            "processed_adjustments": len(adjustments_grouped),
            "message": f"Обработано {len(adjustments_grouped)} корректировок в отдельной таблице"
        }
        
    except Exception as e:
        logger.error(f"Ошибка при обработке корректировок: {e}", exc_info=True)
        conn.rollback()
        logger.info("Транзакция корректировок отменена")
        raise e
    finally:
        conn.close()
        logger.info("Соединение с БД для корректировок закрыто")

def export_pipeline_tables_to_excel(sheet_to_table: dict, make_final_prediction: bool = False) -> BytesIO:
    """
    Экспортирует все таблицы из sheet_to_table в Excel-файл в памяти и возвращает BytesIO.
    """
    logger = setup_custom_logging()
    logger.info(f"Начало экспорта таблиц в Excel. Таблицы: {list(sheet_to_table.keys())}, make_final_prediction: {make_final_prediction}")
    
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        dbname=DB_NAME
    )
    logger.info(f"Подключение к БД для экспорта установлено: {DB_HOST}:{DB_PORT}/{DB_NAME}")
    
    output = BytesIO()
    config = load_config(CONFIG_PATH)
    model_article = config.get('model_article', {})
    logger.info(f"Загружена конфигурация модели: {len(model_article)} статей")
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            sheets_exported = 0
            total_rows_exported = 0
            
            if make_final_prediction:
                logger.info("Создание финального предсказания")
                
                # Получаем данные из DATA_TABLE
                df_data = pd.read_sql_query(f'SELECT * FROM {DATA_TABLE}', conn)
                logger.info(f"Загружены данные из {DATA_TABLE}: {len(df_data)} строк")
                
                # Получаем корректировки из отдельной таблицы
                try:
                    df_adjustments = pd.read_sql_query(f'SELECT "date", "article", adjustment_value FROM {ADJUSTMENTS_TABLE}', conn)
                    logger.info(f"Загружены корректировки из {ADJUSTMENTS_TABLE}: {len(df_adjustments)} строк")
                    
                    # Объединяем данные с корректировками
                    df_data = df_data.merge(
                        df_adjustments, 
                        left_on=['date', 'article'], 
                        right_on=['date', 'article'], 
                        how='left'
                    )
                    # Заполняем пропуски нулями для корректировок
                    df_data['adjustment_value'] = df_data['adjustment_value'].fillna(0)
                    logger.info(f"Данные объединены с корректировками: {len(df_data)} строк")
                except Exception as adj_error:
                    logger.warning(f"Не удалось загрузить корректировки: {adj_error}")
                    # Если таблица корректировок не существует, добавляем нулевые корректировки
                    df_data['adjustment_value'] = 0
                    logger.info("Добавлены нулевые корректировки")
                
                # Применяем маппинг к колонкам
                original_columns = list(df_data.columns)
                df_data.columns = [REVERSE_COLUMN_MAPPING.get(col, col) for col in df_data.columns]
                mapped_columns = list(df_data.columns)
                logger.info(f"Применено обратное маппинг колонок: {len(original_columns)} -> {len(mapped_columns)}")
                
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
                    
                    # Получаем значение корректировки
                    adjustment = row.get('Корректировка, руб', 0) or 0
                    
                    final_rows.append({
                        'Дата': row.get('Дата'),
                        'Статья': article,
                        'Fact': row.get('Fact'),
                        'Прогноз': row.get(mapped_pred_col),
                        'Корректировка': adjustment,
                        'Модель': model_name
                    })
                    
                df_final = pd.DataFrame(final_rows, columns=['Дата', 'Статья', 'Fact', 'Прогноз', 'Корректировка', 'Модель'])
                df_final.to_excel(writer, index=False, sheet_name='final_prediction')
                sheets_exported += 1
                total_rows_exported += len(df_final)
                logger.info(f"Создан лист final_prediction: {len(df_final)} строк")
                
            # Остальные листы
            for sheet, table in sheet_to_table.items():
                if table is None:
                    logger.warning(f"Пропуск листа {sheet}: таблица не указана")
                    continue
                    
                logger.info(f"Экспорт таблицы {table} в лист {sheet}")
                df = pd.read_sql_query(f'SELECT * FROM {table}', conn)
                
                original_columns = list(df.columns)
                df.columns = [REVERSE_COLUMN_MAPPING.get(col, col) for col in df.columns]
                mapped_columns = list(df.columns)
                
                df.to_excel(writer, index=False, sheet_name=sheet)
                sheets_exported += 1
                total_rows_exported += len(df)
                logger.info(f"Лист {sheet} экспортирован: {len(df)} строк, колонки: {len(original_columns)} -> {len(mapped_columns)}")
            
            # Добавляем лист с корректировками, если таблица существует
            try:
                df_adjustments = pd.read_sql_query(f'SELECT "date", "article", adjustment_value FROM {ADJUSTMENTS_TABLE}', conn)
                if not df_adjustments.empty:
                    logger.info(f"Экспорт корректировок: {len(df_adjustments)} строк")
                    # Применяем маппинг к колонкам
                    df_adjustments.columns = [REVERSE_COLUMN_MAPPING.get(col, col) for col in df_adjustments.columns]
                    df_adjustments.to_excel(writer, index=False, sheet_name='Корректировки')
                    sheets_exported += 1
                    total_rows_exported += len(df_adjustments)
                    logger.info("Лист Корректировки добавлен")
                else:
                    logger.info("Таблица корректировок пуста, лист не добавлен")
            except Exception as adj_error:
                logger.warning(f"Не удалось экспортировать корректировки: {adj_error}")
                # Таблица корректировок не существует или пуста
                pass
                
            logger.info(f"Экспорт завершен: {sheets_exported} листов, {total_rows_exported} общих строк")
        
        output.seek(0)
        logger.info("Excel файл создан в памяти и готов к возврату")
        return output
        
    except Exception as e:
        logger.error(f"Ошибка при экспорте таблиц в Excel: {e}", exc_info=True)
        raise e
    finally:
        conn.close()
        logger.info("Соединение с БД для экспорта закрыто")
