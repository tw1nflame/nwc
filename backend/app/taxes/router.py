from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse
from tasks import run_tax_forecast_task
from celery.result import AsyncResult
from celery_app import celery_app
import os
import json
import shutil
import base64
from datetime import datetime
from utils.auth import require_authentication
from utils.training_status import training_status_manager

from taxes.db import get_all_forecast_pairs, restore_excel_from_db
from starlette.background import BackgroundTask
import pandas as pd
import io
from utils.config import load_config
from utils.excel_formatter import save_dataframes_to_excel

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
    # Read file content and encode to base64
    file_content = await history_file.read()
    file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        
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

    # Pass file content instead of path
    task = run_tax_forecast_task.delay(file_content_b64, history_file.filename, forecast_date, groups_dict)
    
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
    res = AsyncResult(task_id)
    if os.name == 'nt':  # Windows
        res.revoke(terminate=True, signal='SIGTERM')
    else:  # Linux/Unix
        res.revoke(terminate=True, signal='SIGKILL')

    training_status_manager.clear_tax_task()
    
    # Save stopped status
    training_status_manager.save_completed_tax_forecast({
        'status': 'revoked',
        'message': 'Прогноз был остановлен пользователем',
        'completed_at': datetime.now().isoformat()
    })
    
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

        # Config loading
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        config_path = os.path.join(base_dir, 'config_refined.yaml')
        config = load_config(config_path)
        tax_models_config = config.get('tax_forecast_models', {'default': 'naive'})
        
        summary_rows = []
        target_col_base = 'Разница Активов и Пассивов'

        for factor, item_id in forecast_pairs:
            file_data = restore_excel_from_db(factor, item_id)
            if file_data:
                filename = f"{factor}_{item_id}_predict.xlsx"
                file_path = os.path.join(temp_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(file_data)
                
                # --- Summary Processing ---
                try:
                    df = pd.read_excel(file_path)
                    
                    # Determine model from config
                    # Default model if not found
                    model = 'naive'
                    
                    # Normalizing config keys to lower case for case-insensitive lookup
                    if isinstance(tax_models_config, dict):
                        models_config_lower = {k.lower(): v for k, v in tax_models_config.items()}
                        
                        key = f"{factor} | {item_id}".lower()
                        if key in models_config_lower:
                            model = models_config_lower[key]
                        elif 'default' in models_config_lower:
                             model = models_config_lower['default']

                    # Find prediction column dynamically
                    prediction_col = None
                    search_term = f"predict_{model}".lower()
                    
                    for col in df.columns:
                        if target_col_base in col and search_term in col.lower():
                            prediction_col = col
                            break
                    
                    fact_col = f"{target_col_base}_fact"
                    
                    if prediction_col and 'Дата' in df.columns:
                        df['Дата'] = pd.to_datetime(df['Дата'])
                        
                        for _, row in df.iterrows():
                            predict_val = row.get(prediction_col)
                            fact_val = row.get(fact_col)
                            
                            # Skip if both are missing? Or just keep going? 
                            # Usually we want rows where we have predictions.
                            if pd.isna(predict_val) and pd.isna(fact_val):
                                continue

                            diff = None
                            dev_pct = None
                            
                            if pd.notna(predict_val) and pd.notna(fact_val):
                                diff = fact_val - predict_val
                                if fact_val != 0:
                                    dev_pct = (fact_val - predict_val) / fact_val
                            
                            summary_rows.append({
                                'Дата': row['Дата'],
                                'Налог': factor,
                                'Компания': item_id,
                                'Модель': model,
                                'Прогноз': predict_val,
                                'Факт': fact_val,
                                'Разница': diff,
                                'Отклонение %': dev_pct
                            })
                except Exception as ex:
                    print(f"Error processing summary for {filename}: {ex}")

        # Save Summary File
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            cols_order = ['Дата', 'Налог', 'Компания', 'Модель', 'Прогноз', 'Факт', 'Разница', 'Отклонение %']
            summary_df = summary_df.reindex(columns=cols_order)
            
            summary_filename = "Результаты.xlsx"
            summary_path = os.path.join(temp_dir, summary_filename)
            save_dataframes_to_excel({'Результаты': summary_df}, summary_path)

        
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
