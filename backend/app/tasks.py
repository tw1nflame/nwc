from celery_app import celery_app
from utils.config import load_config
from utils.datahandler import load_and_transform_data
from utils.pipelines import run_base_plus_pipeline, run_base_pipeline
from utils.common import generate_monthly_period, setup_custom_logging, cleanup_temp_files, cleanup_model_folders
from utils.db_usage import upload_pipeline_result_to_db, SHEET_TO_TABLE, set_pipeline_column, export_pipeline_tables_to_excel, process_exchange_rate_file
from utils.training_status import training_status_manager
from taxes.data_preparation import prepare_tax_data
from taxes.forecast import forecast_taxes
from taxes.db import init_db, get_all_forecast_pairs, restore_excel_from_db, save_excel_to_db
import os
from datetime import datetime
import shutil
import base64

# Устанавливаем флаг для Celery worker чтобы избежать дублирования логов
os.environ['CELERY_WORKER_PROCESS'] = '1'

# Создаем директорию для логов
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

# Инициализируем логирование для Celery worker
logger = setup_custom_logging(os.path.join(log_dir, "celery_worker.log"))

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config_refined.yaml'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
TRAINING_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_files'))

@celery_app.task(bind=True, track_started=True)
def train_task(self, pipeline, items_list, date, data_path, result_file_name):
    logger.info(f"🚀 Начало задачи train_task: pipeline={pipeline}, items={len(items_list)}, date={date}")
    
    # Очистка старых временных файлов перед запуском (только для обучения)
    cleanup_temp_files(logger, task_type='training')
    
    prev_path = None
    
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
    
    # Обрабатываем курсы валют из загруженного файла данных
    try:
        if data_path and os.path.exists(data_path):
            logger.info(f"Обновление курсов валют из файла данных: {data_path}")
            exchange_result = process_exchange_rate_file(data_path, 'Курс', 'Дата')
            logger.info(f"Курсы валют обновлены: {exchange_result['message']}")
        else:
            logger.warning(f"Файл данных не найден: {data_path}")
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
    # Передаем pipeline ('base' или 'base+') в функцию экспорта
    pipeline_arg = None
    if pipeline == "BASE+":
        pipeline_arg = 'base+'
    elif pipeline == "BASE":
        pipeline_arg = 'base'
    output = export_pipeline_tables_to_excel(SHEET_TO_TABLE, pipeline=pipeline_arg)
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
            upload_pipeline_result_to_db(result_file_name, SHEET_TO_TABLE, date, DATE_COLUMN, pipeline='base+')
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
            upload_pipeline_result_to_db(result_file_name, SHEET_TO_TABLE, date, DATE_COLUMN, pipeline='base')
            logger.info("Данные BASE загружены в БД")
        
        # При успешном завершении:
        # 1. Очищаем current_article (больше не обрабатываем)
        training_status_manager.update_current_article('')
        
        # 2. Сохраняем ГЛОБАЛЬНЫЙ статус для ВСЕХ пользователей
        training_status_manager.save_completed_training({
            'status': 'done',
            'message': 'Прогноз готов. Данные сохранены в БД и готовы к выгрузке!',
            'pipeline': pipeline,
            'date': date,
            'articles': list(ITEMS_TO_PREDICT.keys()),  # Список названий статей для которых запущено обучение
            'completed_at': datetime.now().isoformat()
        })
        
        # 3. ВАЖНО: Очищаем current_task_id - разблокируем запуск нового обучения
        training_status_manager.redis_client.delete(training_status_manager.KEYS['current_task_id'])
        
        # Успешное завершение - удаляем файл результатов, так как данные уже в БД
        try:
            if os.path.exists(result_file_name):
                os.remove(result_file_name)
                logger.info(f"🗑️ Удален файл результатов (данные в БД): {result_file_name}")
        except Exception as e:
            logger.warning(f"Не удалось удалить файл результатов: {e}")

        logger.info(f"✅ Задача train_task успешно завершена")
        return {"status": "done"}
        
    except Exception as e:
        logger.error(f"❌ Ошибка в задаче train_task: {e}", exc_info=True)
        
        # При ошибке также сохраняем глобальный статус
        training_status_manager.save_completed_training({
            'status': 'error',
            'message': f'Ошибка при обучении: {str(e)}',
            'error': str(e),
            'articles': list(ITEMS_TO_PREDICT.keys()),  # Список названий статей, для которых была попытка обучения
            'completed_at': datetime.now().isoformat()
        })
        
        # Очищаем current_task_id и прогресс
        training_status_manager.redis_client.delete(training_status_manager.KEYS['current_task_id'])
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

    finally:
        # Всегда удаляем входные временные файлы (они больше не нужны)
        try:
            if prev_path and os.path.exists(prev_path):
                os.remove(prev_path)
                logger.info(f"🗑️ Удален временный файл prev: {prev_path}")
            
            if data_path and os.path.exists(data_path):
                os.remove(data_path)
                logger.info(f"🗑️ Удален загруженный файл данных: {data_path}")
            
            # Очищаем папки с моделями (AutogluonModels, models) чтобы не копились
            cleanup_model_folders(logger=logger)
            
        except Exception as cleanup_error:
            logger.error(f"⚠️ Ошибка при очистке входных файлов: {cleanup_error}")

@celery_app.task(bind=True)
def run_tax_forecast_task(self, file_content_b64: str, filename: str, forecast_date_str: str, selected_groups: dict):
    history_file_path = None
    try:
        # 1. Init DB
        init_db()
            
        # 2. Prepare Data
        self.update_state(state='PROGRESS', meta={'status': 'Preparing data...'})
        
        # Save content to temp file
        os.makedirs("temp_uploads", exist_ok=True)
        history_file_path = f"temp_uploads/{filename}"
        
        # Decode base64 and write to file
        file_content = base64.b64decode(file_content_b64)
        with open(history_file_path, "wb") as f:
            f.write(file_content)
            
        # Ensure data directory exists as prepare_tax_data might need it or use it
        os.makedirs('data', exist_ok=True)
        prepare_tax_data(history_file_path)
        
        # 3. Restore previous forecasts from DB
        self.update_state(state='PROGRESS', meta={'status': 'Restoring previous forecasts...'})
        target_dir = os.path.join('results', 'Разница Активов и Пассивов')
        os.makedirs(target_dir, exist_ok=True)
        
        forecast_pairs = get_all_forecast_pairs()
        for factor, item_id in forecast_pairs:
            file_data = restore_excel_from_db(factor, item_id)
            if file_data:
                # Reconstruct filename: {factor}_{item_id}_predict_BASE.xlsx
                filename = f"{factor}_{item_id}_predict_BASE.xlsx"
                with open(os.path.join(target_dir, filename), 'wb') as f:
                    f.write(file_data)
                    
        # 4. Run Forecast
        # forecast_date_str is expected to be YYYY-MM-DD
        forecast_date = datetime.strptime(forecast_date_str, "%Y-%m-%d")
        
        def progress_callback(msg, current, total):
            percent = round((current / total) * 100, 1) if total > 0 else 0
            self.update_state(state='PROGRESS', meta={
                'status': f'Forecasting: {msg}',
                'current': current,
                'total': total,
                'percent': percent,
                'current_item': msg
            })
            
        forecast_taxes(forecast_date, selected_groups, progress_callback=progress_callback)
        
        # 5. Save updated forecasts to DB
        self.update_state(state='PROGRESS', meta={'status': 'Saving results to DB...'})
        for filename in os.listdir(target_dir):
            if filename.endswith('.xlsx') or filename.endswith('.xls'):
                file_path = os.path.join(target_dir, filename)
                with open(file_path, 'rb') as f:
                    save_excel_to_db(filename, f.read())
                    
        
        # Очищаем статус активной задачи
        training_status_manager.clear_tax_task()
        
        # Сохраняем статус завершенного прогноза
        training_status_manager.save_completed_tax_forecast({
            'status': 'done',
            'message': 'Прогноз налогов готов. Данные сохранены в БД.',
            'completed_at': datetime.now().isoformat(),
            # 'zip_path': abs_zip_path
        })
        
        return {'status': 'Completed'}
        
    except Exception as e:
        print(f"ПРОИЗОШЛА ОШИБКА: {e}")
        logger.error(f"Task failed: {e}")
        self.update_state(
            state='FAILURE', 
            meta={
                'exc_type': type(e).__name__,
                'exc_message': str(e),
                'error': str(e)
            }
        )
        
        # Очищаем статус активной задачи при ошибке
        training_status_manager.clear_tax_task()
        
        # Сохраняем статус ошибки
        training_status_manager.save_completed_tax_forecast({
            'status': 'error',
            'message': f'Ошибка при прогнозе налогов: {str(e)}',
            'error': str(e),
            'completed_at': datetime.now().isoformat()
        })
        raise e
    finally:
        # Cleanup temp file
        if history_file_path and os.path.exists(history_file_path):
            try:
                os.remove(history_file_path)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {history_file_path}: {e}")
