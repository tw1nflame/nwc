from fastapi import APIRouter, UploadFile, File, HTTPException, status, Form
from fastapi.responses import FileResponse, StreamingResponse
import zipfile
import tempfile
import os
from dotenv import load_dotenv
import pandas as pd
import psycopg2
import io
import re
from typing import Dict
from utils.column_mapping import COLUMN_MAPPING
from celery.result import AsyncResult
from utils.training_status import training_status_manager

router = APIRouter()
load_dotenv()

SECRET_WORD = os.getenv('SECRET_WORD')

# --- 1. Проверка секретного слова ---
@router.post('/check-secret-word')
async def check_secret_word(word: str = Form(...)):
	if not SECRET_WORD:
		raise HTTPException(status_code=500, detail="SECRET_WORD не задан в .env")
	if word == SECRET_WORD:
		return {"ok": True}
	return {"ok": False}

# --- 2. Загрузка Excel и запись в БД ---
@router.post('/upload-excel')
async def upload_excel(
	file: UploadFile = File(...),
	word: str = Form(...)
):
	if not SECRET_WORD:
		raise HTTPException(status_code=500, detail="SECRET_WORD не задан в .env")
	if word != SECRET_WORD:
		raise HTTPException(status_code=403, detail="Секретное слово неверно")
	# Импортируем маппинг и переменные окружения
	load_dotenv()
	DB_HOST = os.getenv('DB_HOST')
	DB_PORT = int(os.getenv('DB_PORT', 5432))
	DB_USER = os.getenv('DB_USER')
	DB_PASSWORD = os.getenv('DB_PASSWORD')
	DB_NAME = os.getenv('DB_NAME')
	DATA_TABLE = os.getenv('DATA_TABLE')
	COEFFS_WITH_INTERCEPT_TABLE = os.getenv('COEFFS_WITH_INTERCEPT_TABLE')
	COEFFS_NO_INTERCEPT_TABLE = os.getenv('COEFFS_NO_INTERCEPT_TABLE')
	ENSEMBLE_INFO_TABLE = os.getenv('ENSEMBLE_INFO_TABLE')
	BASEPLUS_TABULAR_ENSEMBLE_INFO_TABLE = os.getenv('BASEPLUS_TABULAR_ENSEMBLE_INFO_TABLE')
	BASEPLUS_FEATURE_IMPORTANCE_TABLE = os.getenv('BASEPLUS_FEATURE_IMPORTANCE_TABLE')
	SHEET_TO_TABLE = {
		'data': DATA_TABLE,
		'coeffs_with_intercept': COEFFS_WITH_INTERCEPT_TABLE,
		'coeffs_no_intercept': COEFFS_NO_INTERCEPT_TABLE,
		'TimeSeries_ensemble_models_info': ENSEMBLE_INFO_TABLE,
		'Tabular_ensemble_models_info': BASEPLUS_TABULAR_ENSEMBLE_INFO_TABLE,
		'Tabular_feature_importance': BASEPLUS_FEATURE_IMPORTANCE_TABLE,
	}
	# Читаем Excel из UploadFile
	try:
		contents = await file.read()
		xls = pd.read_excel(io.BytesIO(contents), sheet_name=None)
	except Exception as e:
		raise HTTPException(status_code=400, detail=f"Ошибка чтения Excel: {e}")
	# Подключение к БД
	try:
		conn = psycopg2.connect(
			host=DB_HOST,
			port=DB_PORT,
			user=DB_USER,
			password=DB_PASSWORD,
			dbname=DB_NAME
		)
	except Exception as e:
		raise HTTPException(status_code=500, detail=f"Ошибка подключения к БД: {e}")
	try:
		cur = conn.cursor()
		for sheet, df in xls.items():
			table_name = SHEET_TO_TABLE.get(sheet)
			if not table_name:
				continue
			if df.empty:
				continue
			# Нормализуем имена колонок и применяем COLUMN_MAPPING
			normalized_cols = []
			for col in df.columns:
				col_str = str(col)
				col_str = re.sub(r"\s+", " ", col_str).strip()
				col_str = col_str.replace("отклонение  %", "отклонение %")
				normalized_cols.append(col_str)
			columns = [COLUMN_MAPPING.get(col, col) for col in normalized_cols]
			df.columns = columns
			col_types = []
			for col in columns:
				dtype = df[col].dtype
				if pd.api.types.is_integer_dtype(dtype):
					sql_type = 'INTEGER'
				elif pd.api.types.is_float_dtype(dtype):
					sql_type = 'DOUBLE PRECISION'
				elif pd.api.types.is_bool_dtype(dtype):
					sql_type = 'BOOLEAN'
				elif pd.api.types.is_datetime64_any_dtype(dtype):
					sql_type = 'TIMESTAMP'
				else:
					sql_type = 'TEXT'
				col_types.append(f'"{col}" {sql_type}')
			# Удаляем таблицу, если она уже есть
			cur.execute(f'DROP TABLE IF EXISTS {table_name} CASCADE')
			conn.commit()
			create_table_sql = f"""
			CREATE TABLE {table_name} (
				{', '.join(col_types)}
			);
			"""
			cur.execute(create_table_sql)
			conn.commit()
			# Полная очистка таблицы
			try:
				cur.execute(f'TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE')
				conn.commit()
			except Exception:
				conn.rollback()
				try:
					cur.execute(f'DELETE FROM {table_name}')
					conn.commit()
				except Exception:
					conn.rollback()
			columns_str = ', '.join([f'"{col}"' for col in columns])
			values_template = ', '.join(['%s'] * len(columns))
			insert_sql = f'INSERT INTO {table_name} ({columns_str}) VALUES ({values_template})'
			df = df.dropna(how='all')
			records = df.where(pd.notnull(df), None).values.tolist()
			for row in records:
				if len(row) != len(columns):
					continue
				cur.execute(insert_sql, row)
			conn.commit()
		cur.close()
	finally:
		conn.close()
	return {"ok": True, "message": "Excel успешно загружен в БД"}


# --- 4. Скачать все логи архивом (требует секретное слово) ---
@router.post('/download-logs-archive')
async def download_logs_archive(word: str = Form(...)):
	if not SECRET_WORD:
		raise HTTPException(status_code=500, detail="SECRET_WORD не задан в .env")
	if word != SECRET_WORD:
		raise HTTPException(status_code=403, detail="Секретное слово неверно")
	base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
	files = [
		os.path.join(base_dir, 'db_operations.log'),
		os.path.join(base_dir, 'celery_worker.log'),
		os.path.join(base_dir, 'log.txt')
	]
	files_to_zip = [(f, os.path.basename(f)) for f in files if os.path.exists(f)]
	if not files_to_zip:
		raise HTTPException(status_code=404, detail="Нет ни одного лог-файла для скачивания")
	with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
		with zipfile.ZipFile(tmp_zip, 'w') as archive:
			for abs_path, arc_name in files_to_zip:
				archive.write(abs_path, arc_name)
		tmp_zip_path = tmp_zip.name
	return FileResponse(tmp_zip_path, filename="logs.zip", media_type="application/zip")


# --- 5. Остановить текущую задачу и очистить Redis ---
@router.post('/stop-and-clear')
async def stop_and_clear_training(word: str = Form(...)):
	"""
	Останавливает текущую задачу обучения и полностью очищает Redis
	"""
	if not SECRET_WORD:
		raise HTTPException(status_code=500, detail="SECRET_WORD не задан в .env")
	if word != SECRET_WORD:
		raise HTTPException(status_code=403, detail="Секретное слово неверно")
	
	# Получаем ID текущей задачи
	task_id = training_status_manager.get_current_task_id()
	
	stopped_task = None
	if task_id:
		# Останавливаем задачу
		res = AsyncResult(task_id)
		if res.state in ['PENDING', 'STARTED', 'RETRY']:
			# Останавливаем с терминацией
			if os.name == 'nt':  # Windows
				res.revoke(terminate=True, signal='SIGTERM')
			else:  # Linux/Unix
				res.revoke(terminate=True, signal='SIGKILL')
			stopped_task = task_id
	
	# Полностью очищаем Redis
	training_status_manager.clear_training_progress()
	
	return {
		"ok": True,
		"message": "Задача остановлена и Redis очищен",
		"stopped_task_id": stopped_task
	}