#!/usr/bin/env python3
"""
Генерация синтетического файла корректировок
Создает Excel файл с листом 'Корректировки' для тестирования функционала корректировок
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import random

# Установка seed для воспроизводимости
np.random.seed(42)
random.seed(42)

# Параметры временного диапазона (аналогично generate_synthetic_base.py)
START_YEAR = 2023
END_YEAR = 2025
START_MONTH = 1
END_MONTH = 6  # До июня 2025

# Список статей из generate_synthetic_base.py
ARTICLES = [
    'Торговая ДЗ_USD',
    'Прочая ДЗ',
    'Авансы выданные и расходы будущих периодов',
    'Прочие налоги к возмещению ST',
    'Прочие налоговые обязательства',
    'Задолженность перед персоналом',
    'Резерв по неиспользованным отпускам',
    'Краткосрочный резерв по премиям',
    'Прочее',
    'Кредиторская задлолженность по ОС',
    'Авансы полученные',
    'Авансы полученные(металлы)',
    'Торговая КЗ',
    'Авансовые платежи по налогу на прибыль',
    'Обязательства по налогу на прибыль',
    'ЧОК (нормализовано на расчеты с акционерами)'
]

# Типы корректировок
CORRECTION_TYPES = ['прогноз', 'факт']

# Примеры описаний корректировок
DESCRIPTIONS = [
    'Корректировка по результатам аудита',
    'Пересчет по новой методологии',
    'Учет сезонных факторов',
    'Корректировка валютных операций',
    'Пересмотр бюджетных показателей',
    'Учет инфляционных факторов',
    'Корректировка по результатам инвентаризации',
    'Пересчет амортизации',
    'Учет курсовых разниц',
    'Корректировка налоговых обязательств',
    'Пересмотр резервов',
    'Учет изменений в законодательстве',
    'Корректировка по решению руководства',
    'Пересчет по фактическим данным',
    'Учет экономических изменений'
]

def generate_corrections_data(num_rows=50):
    """
    Генерирует синтетические данные корректировок
    
    Args:
        num_rows: количество строк для генерации
    
    Returns:
        pd.DataFrame: датафрейм с корректировками
    """
    corrections = []
    
    for i in range(num_rows):
        # Случайная статья
        article = random.choice(ARTICLES)
        
        # Случайный год и месяц в диапазоне
        if START_YEAR == END_YEAR:
            year = START_YEAR
            month = random.randint(START_MONTH, END_MONTH)
        else:
            year = random.randint(START_YEAR, END_YEAR)
            if year == START_YEAR:
                month = random.randint(START_MONTH, 12)
            elif year == END_YEAR:
                month = random.randint(1, END_MONTH)
            else:
                month = random.randint(1, 12)
        
        # Генерация корректировки в рублях (от -50000 до +50000)
        correction_rub = round(np.random.normal(0, 15000), 2)
        
        # Случайный тип корректировки
        correction_type = random.choice(CORRECTION_TYPES)
        
        # Случайное описание
        description = random.choice(DESCRIPTIONS)
        
        correction = {
            'Статья': article,
            'Год': year,
            'Месяц': month,
            'Корректировка, руб': correction_rub,
            'Тип': correction_type,
            'Описание': description
        }
        
        corrections.append(correction)
    
    return pd.DataFrame(corrections)

def main():
    """
    Основная функция для создания файла корректировок
    """
    print("Генерация синтетического файла корректировок...")
    
    # Генерируем данные
    df_corrections = generate_corrections_data(50)
    
    # Сортируем по году, месяцу и статье для удобства
    df_corrections = df_corrections.sort_values(['Год', 'Месяц', 'Статья']).reset_index(drop=True)
    
    # Создаем директорию для результатов
    os.makedirs('results', exist_ok=True)
    
    # Имя файла с текущей датой
    current_date = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'results/synthetic_corrections_{current_date}.xlsx'
    
    # Сохраняем в Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_corrections.to_excel(writer, index=False, sheet_name='Корректировки')
    
    print(f"Файл корректировок создан: {filename}")
    print(f"Количество записей: {len(df_corrections)}")
    print("\nПримеры первых 5 записей:")
    print(df_corrections.head().to_string(index=False))
    
    # Статистика по типам корректировок
    print(f"\nСтатистика по типам:")
    type_stats = df_corrections['Тип'].value_counts()
    for correction_type, count in type_stats.items():
        print(f"  {correction_type}: {count} записей")
    
    # Статистика по годам
    print(f"\nСтатистика по годам:")
    year_stats = df_corrections['Год'].value_counts().sort_index()
    for year, count in year_stats.items():
        print(f"  {year}: {count} записей")
    
    return filename

if __name__ == "__main__":
    main()
