from celery_app import celery_app
from utils.config import load_config
from utils.datahandler import load_and_transform_data
from utils.pipelines import run_base_plus_pipeline, run_base_pipeline
from utils.common import generate_monthly_period
import os
from datetime import datetime
import redis

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config_refined.yaml'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
redis_client = redis.Redis.from_url(REDIS_URL)

@celery_app.task(bind=True, track_started=True)
def train_task(self, pipeline, items_list, date, data_path, prev_path, result_file_name):
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
    try:
        task_id = self.request.id
        redis_client.set('current_train_task_id', task_id)
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
                prev_predicts_file=prev_path
            )
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
                prev_predicts_file=prev_path
            )
        return {"status": "done", "result_file": os.path.basename(result_file_name)}
    except Exception as e:
        return {"status": "error", "error": str(e)}
