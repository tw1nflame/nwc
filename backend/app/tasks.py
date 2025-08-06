from celery_app import celery_app
from utils.config import load_config
from utils.datahandler import load_and_transform_data
from utils.pipelines import run_base_plus_pipeline, run_base_pipeline
from utils.common import generate_monthly_period
from utils.db_usage import upload_pipeline_result_to_db, SHEET_TO_TABLE, set_pipeline_column, export_pipeline_tables_to_excel
from utils.training_status import training_status_manager
import os
from datetime import datetime

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config_refined.yaml'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../results'))
TRAINING_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_files'))

@celery_app.task(bind=True, track_started=True)
def train_task(self, pipeline, items_list, date, data_path, result_file_name):
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
    
    # Инициализация статуса обучения
    total_articles = len(ITEMS_TO_PREDICT)
    task_id = self.request.id
    training_status_manager.initialize_training(total_articles, task_id)
    
    # Создаем файл с данными из БД для использования в pipeline
    prev_path = os.path.join(TRAINING_FILES_DIR, f"prev_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")
    output = export_pipeline_tables_to_excel(SHEET_TO_TABLE)
    with open(prev_path, "wb") as f:
        f.write(output.getvalue())
    try:
        if pipeline == "BASE+":
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
            upload_pipeline_result_to_db(result_file_name, SHEET_TO_TABLE, date, DATE_COLUMN)
            set_pipeline_column(DATE_COLUMN, date, 'BASE+')
        elif pipeline == "BASE":
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
            upload_pipeline_result_to_db(result_file_name, SHEET_TO_TABLE, date, DATE_COLUMN)
            set_pipeline_column(DATE_COLUMN, date, 'BASE')
        
        # НЕ очищаем прогресс при успешном завершении - оставляем task_id
        # для отображения статуса "done" до нажатия кнопки ОК
        
        return {"status": "done", "result_file": os.path.basename(result_file_name)}
    except Exception as e:
        # Очищаем прогресс при ошибке или отмене
        training_status_manager.clear_training_progress()
        
        # Проверяем, была ли задача отменена
        try:
            from celery.exceptions import Revoked
            if isinstance(e, Revoked):
                return {"status": "revoked", "error": "Task was revoked"}
        except ImportError:
            # Fallback для совместимости с разными версиями Celery
            if 'revoked' in str(e).lower():
                return {"status": "revoked", "error": "Task was revoked"}
        
        return {"status": "error", "error": str(e)}
