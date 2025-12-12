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

from taxes.db import get_all_forecast_pairs, restore_excel_from_db
from starlette.background import BackgroundTask

def cleanup_file(path: str):
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as e:
        print(f"Error cleaning up file {path}: {e}")

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
    # Clear previous completed status
    training_status_manager.clear_last_completed_tax_forecast()

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
        file_path = result['zip_path']
        return FileResponse(
            file_path, 
            media_type='application/zip', 
            filename="forecast_results.zip",
            background=BackgroundTask(cleanup_file, file_path)
        )
    
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

@router.get("/last-completed")
@require_authentication
async def get_last_completed(request: Request):
    status = training_status_manager.get_last_completed_tax_forecast()
    return status or {"status": "idle"}

@router.post("/clear-status")
@require_authentication
async def clear_status(request: Request):
    training_status_manager.clear_last_completed_tax_forecast()
    return {"status": "cleared"}

@router.get("/export_excel")
@require_authentication
async def export_excel(request: Request):
    temp_dir = None
    zip_path = None
    try:
        # Create temp dir
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        temp_dir = f"temp_tax_export_{timestamp}"
        os.makedirs(temp_dir, exist_ok=True)
        
        forecast_pairs = get_all_forecast_pairs()
        if not forecast_pairs:
             raise HTTPException(status_code=404, detail="No forecast data found")

        for factor, item_id in forecast_pairs:
            file_data = restore_excel_from_db(factor, item_id)
            if file_data:
                filename = f"{factor}_{item_id}_predict.xlsx"
                with open(os.path.join(temp_dir, filename), 'wb') as f:
                    f.write(file_data)
        
        # Zip
        zip_name = f"tax_forecast_{timestamp}"
        zip_path = shutil.make_archive(zip_name, 'zip', temp_dir)
        
        # Clean up temp dir
        shutil.rmtree(temp_dir)
        
        return FileResponse(
            zip_path, 
            media_type='application/zip', 
            filename=f"{zip_name}.zip",
            background=BackgroundTask(cleanup_file, zip_path)
        )
        
    except Exception as e:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        if zip_path and os.path.exists(zip_path):
            os.remove(zip_path)
        raise HTTPException(status_code=500, detail=str(e))
