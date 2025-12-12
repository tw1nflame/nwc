from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from tasks import run_tax_forecast_task
from celery.result import AsyncResult
from celery_app import celery_app
import os
import json
import shutil
from datetime import datetime
from utils.auth import require_authentication
from utils.training_status import training_status_manager

router = APIRouter(prefix="/taxes", tags=["taxes"])

@router.post("/forecast")
@require_authentication
async def start_forecast(
    request: Request,
    history_file: UploadFile = File(...),
    forecast_date: str = Form(...),
    selected_groups: str = Form(...) # JSON string of list of strings
):
    # Save history file
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = f"temp_uploads/{history_file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(history_file.file, buffer)
        
    # Parse groups
    try:
        selected_pairs = json.loads(selected_groups)
        if not isinstance(selected_pairs, list):
             raise ValueError("selected_groups must be a list")
             
        # Convert list of "Group | Company" to dict {Group: [Company, ...]}
        groups_dict = {}
        for item in selected_pairs:
            if " | " in item:
                group, company = item.split(' | ', 1)
                if group not in groups_dict:
                    groups_dict[group] = []
                groups_dict[group].append(company)
            else:
                # Handle unexpected format if necessary, or ignore
                pass
                
    except (json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON for selected_groups: {e}")

    # Start task
    # Use absolute path for file_path as Celery worker might be in different dir
    abs_file_path = os.path.abspath(file_path)
    task = run_tax_forecast_task.delay(abs_file_path, forecast_date, groups_dict)
    
    # Сохраняем ID задачи в Redis
    training_status_manager.set_tax_task(task.id)
    
    return {"task_id": task.id}

@router.get("/status/{task_id}")
@require_authentication
async def get_status(request: Request, task_id: str):
    task_result = AsyncResult(task_id)
    response = {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result if task_result.ready() else None
    }
    if task_result.state == 'PROGRESS':
        response['meta'] = task_result.info
    elif task_result.state == 'FAILURE':
        response['error'] = str(task_result.result)
        
    return response

@router.get("/download/{task_id}")
@require_authentication
async def download_result(request: Request, task_id: str):
    task_result = AsyncResult(task_id)
    if not task_result.ready():
        raise HTTPException(status_code=400, detail="Task not finished")
    
    result = task_result.result
    if isinstance(result, dict) and 'zip_path' in result:
        return FileResponse(result['zip_path'], media_type='application/zip', filename="forecast_results.zip")
    
    raise HTTPException(status_code=404, detail="Result file not found")

@router.post("/stop/{task_id}")
@require_authentication
async def stop_forecast(request: Request, task_id: str):
    celery_app.control.revoke(task_id, terminate=True)
    training_status_manager.clear_tax_task()
    return {"status": "Task revoked"}

@router.get("/active-task")
@require_authentication
async def get_active_task(request: Request):
    task_id = training_status_manager.get_current_tax_task()
    return {"task_id": task_id}
