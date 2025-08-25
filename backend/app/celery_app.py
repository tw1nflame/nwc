from celery import Celery
import os


def build_redis_url():
    url = os.environ.get('REDIS_URL')
    if url:
        return url
    host = os.environ.get('REDIS_HOST', 'localhost')
    port = os.environ.get('REDIS_PORT', '6379')
    user = os.environ.get('REDIS_USER', '')
    password = os.environ.get('REDIS_PASSWORD', '')
    db = os.environ.get('REDIS_DB', '0')
    auth = f"{user}:{password}@" if user or password else ""
    return f"redis://{auth}{host}:{port}/{db}"

CELERY_BROKER_URL = build_redis_url()
CELERY_RESULT_BACKEND = build_redis_url()

celery_app = Celery('tasks', broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

celery_app.conf.update(
    task_serializer='json',
    result_serializer='json',
    accept_content=['json'],
    timezone='UTC',
    enable_utc=True,
)
