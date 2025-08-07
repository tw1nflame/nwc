from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import shutil
import os
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from celery.result import AsyncResult
from tasks import train_task
from fastapi.responses import FileResponse, StreamingResponse, Response

from utils.db_usage import export_pipeline_tables_to_excel, SHEET_TO_TABLE, process_adjustments_file
from utils.training_status import training_status_manager

app = FastAPI()

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../config_refined.yaml'))

@app.get("/config")
async def get_config():
    """
    Возвращает содержимое config_refined.yaml из корня проекта.
    """
    if not os.path.exists(CONFIG_PATH):
        raise HTTPException(status_code=404, detail="Config file not found")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content, media_type="text/yaml")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

TRAINING_FILES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_files'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), 'results'))
os.makedirs(TRAINING_FILES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

@app.post("/train/")
async def train(
    pipeline: str = Form(...),
    items: str = Form(...),
    date: str = Form(...),
    data_file: UploadFile = File(...)
):
    # Проверяем, не выполняется ли уже обучение
    task_id = training_status_manager.get_current_task_id()
    if task_id:
        res = AsyncResult(task_id)
        if res.state in ['PENDING', 'STARTED', 'RETRY']:
            raise HTTPException(
                status_code=409,
                detail="Обучение уже выполняется. Дождитесь завершения или остановите текущую задачу."
            )
    
    data_path = os.path.join(TRAINING_FILES_DIR, f"uploaded_data_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")
    with open(data_path, "wb") as f:
        shutil.copyfileobj(data_file.file, f)
    import json as _json
    items_list = _json.loads(items)
    result_file_name = os.path.join(RESULTS_DIR, f"predict_{pipeline}_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")
    celery_task = train_task.apply_async(args=[pipeline, items_list, date, data_path, result_file_name])
    # Сохраняем актуальный task_id через training_status_manager
    training_status_manager.set_current_task_id(celery_task.id)
    return {"status": "ok", "task_id": celery_task.id}

@app.get("/current_task_id")
async def get_current_task_id():
    task_id = training_status_manager.get_current_task_id()
    if not task_id:
        return {"task_id": None}
    return {"task_id": task_id}

@app.get("/train_status/")
async def train_status():
    task_id = training_status_manager.get_current_task_id()
    if not task_id:
        return {"status": "idle"}
    
    res = AsyncResult(task_id)
    progress = training_status_manager.get_training_progress()
    
    if res.state == 'PENDING':
        return {"status": "pending", **progress}
    elif res.state == 'STARTED':
        return {"status": "running", **progress}
    elif res.state == 'SUCCESS':
        result = res.result
        # НЕ очищаем task_id, чтобы статус "done" сохранялся
        return {"status": "done", **(result if isinstance(result, dict) else {})}
    elif res.state == 'FAILURE':
        # Очищаем прогресс после ошибки
        training_status_manager.clear_training_progress()
        return {"status": "error", "error": str(res.result)}
    elif res.state == 'REVOKED':
        # Очищаем прогресс после отмены
        training_status_manager.clear_training_progress()
        return {"status": "revoked", "message": "Training was stopped"}
    else:
        return {"status": res.state.lower(), **progress}

@app.post("/stop_train/")
async def stop_train():
    task_id = training_status_manager.get_current_task_id()
    if not task_id:
        raise HTTPException(status_code=404, detail="Task not found")
    
    res = AsyncResult(task_id)
    import os
    if os.name == 'nt':  # Windows
        res.revoke(terminate=True, signal='SIGTERM')
    else:  # Linux/Unix
        res.revoke(terminate=True, signal='SIGKILL')
    
    # Очищаем прогресс при остановке
    training_status_manager.clear_training_progress()
    
    return {"status": "revoked", "task_id": task_id}

@app.post("/clear_status/")
async def clear_status():
    """
    Очистка статуса завершенного обучения (кнопка ОК на фронтенде)
    """
    training_status_manager.clear_completed_status()
    return {"status": "cleared"}

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/export_excel/")
async def export_excel():
    """
    Экспортирует таблицы BASEPLUS pipeline в Excel и возвращает файл из памяти.
    """
    sheet_to_table = SHEET_TO_TABLE
    excel_bytes = export_pipeline_tables_to_excel(sheet_to_table, make_final_prediction=True)
    filename = f"export_BASEPLUS_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx"
    return StreamingResponse(
        excel_bytes,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@app.post("/upload_adjustments/")
async def upload_adjustments(
    adjustments_file: UploadFile = File(...),
    date_column: str = Form(default="Дата")
):
    """
    Загружает файл корректировок и обновляет столбец adjustments в основной таблице данных.
    Полностью перезаписывает все корректировки.
    
    Args:
        adjustments_file: Excel файл с листом 'Корректировки'
        date_column: имя столбца с датой в Excel (по умолчанию 'Дата')
    
    Returns:
        Статус обработки и количество обработанных корректировок
    """
    try:
        # Проверяем тип файла
        if not adjustments_file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(status_code=400, detail="Файл должен быть в формате Excel (.xlsx или .xls)")
        
        # Сохраняем временный файл
        temp_file_path = os.path.join(TRAINING_FILES_DIR, f"adjustments_{datetime.now().strftime('%Y%m%d%H%M%S')}.xlsx")
        with open(temp_file_path, "wb") as f:
            shutil.copyfileobj(adjustments_file.file, f)
        
        # Обрабатываем корректировки
        result = process_adjustments_file(temp_file_path, date_column)
        
        # Удаляем временный файл
        try:
            os.remove(temp_file_path)
        except:
            pass  # Игнорируем ошибки удаления временного файла
        
        return {
            "status": "success",
            "message": result["message"],
            "processed_adjustments": result["processed_adjustments"],
            "filename": adjustments_file.filename
        }
        
    except Exception as e:
        # Удаляем временный файл в случае ошибки
        try:
            if 'temp_file_path' in locals():
                os.remove(temp_file_path)
        except:
            pass
        
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла корректировок: {str(e)}")