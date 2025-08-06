#!/usr/bin/env python3
"""
Тестовый скрипт для проверки путей и импортов
"""
import os
import sys

# Добавляем путь к backend/app в PYTHONPATH
backend_app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'backend', 'app'))
sys.path.insert(0, backend_app_path)

def test_imports():
    """Тестирование импортов"""
    print("Тестирование импортов...")
    
    try:
        from celery.result import AsyncResult
        print("✅ AsyncResult импортирован успешно")
    except ImportError as e:
        print(f"❌ Ошибка импорта AsyncResult: {e}")
    
    try:
        from celery.exceptions import Revoked
        print("✅ Revoked импортирован успешно")
    except ImportError as e:
        print(f"⚠️  Revoked не импортирован (это нормально для некоторых версий Celery): {e}")
    
    try:
        from tasks import train_task
        print("✅ train_task импортирован успешно")
    except ImportError as e:
        print(f"❌ Ошибка импорта train_task: {e}")
    
    try:
        from utils.training_status import training_status_manager
        print("✅ training_status_manager импортирован успешно")
    except ImportError as e:
        print(f"❌ Ошибка импорта training_status_manager: {e}")

def test_paths():
    """Тестирование путей"""
    print("\nТестирование путей...")
    
    # Проверяем пути
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'config_refined.yaml'))
    print(f"Config path: {config_path}")
    print(f"Config exists: {'✅' if os.path.exists(config_path) else '❌'}")
    
    # Проверяем директории
    backend_dir = os.path.join(os.path.dirname(__file__), 'backend', 'app')
    training_files_dir = os.path.join(backend_dir, 'training_files')
    models_dir = os.path.join(backend_dir, 'models')
    results_dir = os.path.join(backend_dir, 'results')
    
    print(f"Training files dir: {training_files_dir}")
    print(f"Training files exists: {'✅' if os.path.exists(training_files_dir) else '❌'}")
    
    print(f"Models dir: {models_dir}")
    print(f"Models exists: {'✅' if os.path.exists(models_dir) else '❌'}")
    
    print(f"Results dir: {results_dir}")
    print(f"Results exists: {'✅' if os.path.exists(results_dir) else '❌'}")
    
    # Проверяем предобученные модели
    pretrained_models_dir = os.path.join(os.path.dirname(__file__), 'pretrained_models', 'chronos-bolt-base')
    print(f"Pretrained models dir: {pretrained_models_dir}")
    print(f"Pretrained models exists: {'✅' if os.path.exists(pretrained_models_dir) else '❌'}")

if __name__ == "__main__":
    print("🔍 Проверка исправлений путей и импортов...")
    test_imports()
    test_paths()
    print("\n🎉 Проверка завершена!")
