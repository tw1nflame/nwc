from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
import json
from tasks import train_task
from celery_app import celery_app
import redis
from fastapi.responses import FileResponse, StreamingResponse
from upload import export_pipeline_tables_to_excel, SHEET_TO_TABLE, BASEPLUS_SHEET_TO_TABLE

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

TRAINING_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_files'))
os.makedirs(TRAINING_FILES_DIR, exist_ok=True)
REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
redis_client = redis.Redis.from_url(REDIS_URL)

@app.post("/train/")
async def train(
    pipeline: str = Form(...),
    items: str = Form(...),
    date: str = Form(...),
    data_file: UploadFile = File(...),
    prev_results_file: UploadFile = File(...)
):
    data_path = os.path.join(TRAINING_FILES_DIR, f"uploaded_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")
    prev_path = os.path.join(TRAINING_FILES_DIR, f"uploaded_prev_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")
    with open(data_path, "wb") as f:
        shutil.copyfileobj(data_file.file, f)
    with open(prev_path, "wb") as f:
        shutil.copyfileobj(prev_results_file.file, f)
    import json as _json
    items_list = _json.loads(items)
    result_file_name = os.path.join(TRAINING_FILES_DIR, f"predict_{pipeline}_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")
    celery_task = train_task.apply_async(args=[pipeline, items_list, date, data_path, prev_path, result_file_name])
    # Сохраняем актуальный task_id в Redis
    redis_client.set('current_train_task_id', celery_task.id)
    return {"status": "ok", "task_id": celery_task.id}

@app.get("/current_task_id")
async def get_current_task_id():
    task_id = redis_client.get('current_train_task_id')
    if not task_id:
        return {"task_id": None}
    return {"task_id": task_id.decode()}

@app.get("/train_status/")
async def train_status():
    task_id = redis_client.get('current_train_task_id')
    if not task_id:
        return {"status": "idle"}
    task_id = task_id.decode()
    res = AsyncResult(task_id)
    if res.state == 'PENDING':
        return {"status": "pending"}
    elif res.state == 'STARTED':
        return {"status": "running"}
    elif res.state == 'SUCCESS':
        result = res.result
        return {"status": "done", **(result if isinstance(result, dict) else {})}
    elif res.state == 'FAILURE':
        return {"status": "error", "error": str(res.result)}
    else:
        return {"status": res.state.lower()}

@app.post("/stop_train/")
async def stop_train():
    task_id = redis_client.get('current_train_task_id')
    if not task_id:
        raise HTTPException(status_code=404, detail="Task not found")
    task_id = task_id.decode()
    res = AsyncResult(task_id)
    import os
    if os.name == 'nt':  # Windows
        res.revoke(terminate=True, signal='SIGTERM')
    else:  # Linux/Unix
        res.revoke(terminate=True, signal='SIGKILL')
    # После остановки можно очистить ключ, если нужно:
    # redis_client.delete('current_train_task_id')
    return {"status": "revoked", "task_id": task_id}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/export_excel/")
async def export_excel():
    """
    Экспортирует таблицы BASEPLUS pipeline в Excel и возвращает файл из памяти.
    """
    sheet_to_table = BASEPLUS_SHEET_TO_TABLE
    excel_bytes = export_pipeline_tables_to_excel(sheet_to_table)
    filename = f"export_BASEPLUS_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
    return StreamingResponse(
        excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )