from celery_app import celery_app
from utils.config import load_config
from utils.datahandler import load_and_transform_data
from utils.pipelines import run_base_plus_pipeline, run_base_pipeline
from utils.common import generate_monthly_period, setup_custom_logging
from utils.db_usage import upload_pipeline_result_to_db, SHEET_TO_TABLE, set_pipeline_column, export_pipeline_tables_to_excel, process_exchange_rate_file
from utils.training_status import training_status_manager
import os
from datetime import datetime

# Устанавливаем флаг для Celery worker чтобы избежать дублирования логов
os.environ['CELERY_WORKER_PROCESS'] = '1'

# Создаем директорию для логов
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Инициализируем логирование для Celery worker
logger = setup_custom_logging(os.path.join(log_dir, "celery_worker.log"))

# Получаем путь к файлу курсов из переменных окружения
EXCHANGE_RATE_FILE_PATH = os.getenv('EXCHANGE_RATE_FILE_PATH')

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config_refined.yaml'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
TRAINING_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_files'))

@celery_app.task(bind=True, track_started=True)
def train_task(self, pipeline, items_list, date, data_path, result_file_name):
    logger.info(f"🚀 Начало задачи train_task: pipeline={pipeline}, items={len(items_list)}, date={date}")
    
    config = load_config(CONFIG_PATH)
    DATE_COLUMN = config['DATE_COLUMN']
    RATE_COLUMN = config['RATE_COLUMN']
    ITEM_ID = config['item_id']
    FACTOR = config['factor']
    models_to_use = config['models_to_use']
    TABPFNMIX_model = config.get('TABPFNMIX_model', {})
    METRIC = config['metric_for_training']
    ITEMS_TO_PREDICT = {key: config['Статья'][key] for key in items_list}
    CHOSEN_MONTH = datetime.strptime(date, "%Y-%m-%d")
    MONTHES_TO_PREDICT = generate_monthly_period(CHOSEN_MONTH)
    df_all_items = load_and_transform_data(data_path, DATE_COLUMN, RATE_COLUMN)
    
    logger.info(f"Конфигурация загружена: {len(ITEMS_TO_PREDICT)} статей для обработки")
    
    # Обрабатываем курсы валют перед запуском пайплайна
    try:
        if EXCHANGE_RATE_FILE_PATH and os.path.exists(EXCHANGE_RATE_FILE_PATH):
            logger.info(f"Обновление курсов валют из файла: {EXCHANGE_RATE_FILE_PATH}")
            exchange_result = process_exchange_rate_file(EXCHANGE_RATE_FILE_PATH, 'Курс', 'Дата')
            logger.info(f"Курсы валют обновлены: {exchange_result['message']}")
        else:
            logger.warning(f"Файл курсов не найден или не указан: {EXCHANGE_RATE_FILE_PATH}")
    except Exception as e:
        logger.error(f"Ошибка при обновлении курсов валют: {e}")
        # Продолжаем выполнение пайплайна, даже если курсы не обновились
    
    # Инициализация статуса обучения
    total_articles = len(ITEMS_TO_PREDICT)
    task_id = self.request.id
    training_status_manager.initialize_training(total_articles, task_id)
    logger.info(f"Статус обучения инициализирован для {total_articles} статей")
    
    # Создаем файл с данными из БД для использования в pipeline
    prev_path = os.path.join(TRAINING_FILES_DIR, f"prev_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")
    output = export_pipeline_tables_to_excel(SHEET_TO_TABLE)
    with open(prev_path, "wb") as f:
        f.write(output.getvalue())
    logger.info(f"Создан файл предыдущих данных: {prev_path}")
    
    try:
        if pipeline == "BASE+":
            logger.info("Запуск BASE+ пайплайна")
            run_base_plus_pipeline(
                df_all_items=df_all_items,
                ITEMS_TO_PREDICT=ITEMS_TO_PREDICT,
                config=config,
                DATE_COLUMN=DATE_COLUMN,
                RATE_COLUMN=RATE_COLUMN,
                ITEM_ID=ITEM_ID,
                FACTOR=FACTOR,
                models_to_use=models_to_use,
                TABPFNMIX_model=TABPFNMIX_model,
                METRIC=METRIC,
                CHOSEN_MONTH=CHOSEN_MONTH,
                MONTHES_TO_PREDICT=MONTHES_TO_PREDICT,
                result_file_name=result_file_name,
                prev_predicts_file=prev_path,
                status_manager=training_status_manager
            )
            logger.info("BASE+ пайплайн завершен, загрузка в БД")
            upload_pipeline_result_to_db(result_file_name, SHEET_TO_TABLE, date, DATE_COLUMN)
            set_pipeline_column(DATE_COLUMN, date, 'BASE+')
            logger.info("Данные BASE+ загружены в БД")
            
        elif pipeline == "BASE":
            logger.info("Запуск BASE пайплайна")
            run_base_pipeline(
                df_all_items=df_all_items,
                ITEMS_TO_PREDICT=ITEMS_TO_PREDICT,
                config=config,
                DATE_COLUMN=DATE_COLUMN,
                RATE_COLUMN=RATE_COLUMN,
                ITEM_ID=ITEM_ID,
                FACTOR=FACTOR,
                models_to_use=models_to_use,
                METRIC=METRIC,
                CHOSEN_MONTH=CHOSEN_MONTH,
                MONTHES_TO_PREDICT=MONTHES_TO_PREDICT,
                result_file_name=result_file_name,
                prev_predicts_file=prev_path,
                status_manager=training_status_manager
            )
            logger.info("BASE пайплайн завершен, загрузка в БД")
            upload_pipeline_result_to_db(result_file_name, SHEET_TO_TABLE, date, DATE_COLUMN)
            set_pipeline_column(DATE_COLUMN, date, 'BASE')
            logger.info("Данные BASE загружены в БД")
        
        # НЕ очищаем прогресс при успешном завершении - оставляем task_id
        # для отображения статуса "done" до нажатия кнопки ОК
        
        logger.info(f"✅ Задача train_task успешно завершена: {result_file_name}")
        return {"status": "done", "result_file": os.path.basename(result_file_name)}
        
    except Exception as e:
        logger.error(f"❌ Ошибка в задаче train_task: {e}", exc_info=True)
        # Очищаем прогресс при ошибке или отмене
        training_status_manager.clear_training_progress()
        
        # Проверяем, была ли задача отменена
        try:
            from celery.exceptions import Revoked
            if isinstance(e, Revoked):
                logger.warning("Задача была отменена (Revoked)")
                return {"status": "revoked", "error": "Task was revoked"}
        except ImportError:
            # Fallback для совместимости с разными версиями Celery
            if 'revoked' in str(e).lower():
                logger.warning("Задача была отменена (revoked в строке ошибки)")
                return {"status": "revoked", "error": "Task was revoked"}
        
        return {"status": "error", "error": str(e)}
