"""
Модуль для управления статусом обучения через Redis
"""
import redis
import os
from typing import Optional, Dict, Any


class TrainingStatusManager:
    """Управление статусом обучения через Redis"""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Инициализация менеджера статуса
        
        Args:
            redis_url: URL для подключения к Redis
        """
        self.redis_url = redis_url or os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
        self.redis_client = redis.Redis.from_url(self.redis_url)
        
        # Ключи для хранения информации о прогрессе
        self.KEYS = {
            'current_article': 'training_progress:current_article',
            'total_articles': 'training_progress:total_articles', 
            'processed_articles': 'training_progress:processed_articles',
            'percentage': 'training_progress:percentage',
            'current_task_id': 'current_train_task_id'
        }
    
    def initialize_training(self, total_articles: int, task_id: str) -> None:
        """
        Инициализация нового процесса обучения
        
        Args:
            total_articles: Общее количество статей для обучения
            task_id: ID задачи Celery
        """
        try:
            self.redis_client.set(self.KEYS['total_articles'], total_articles)
            self.redis_client.set(self.KEYS['processed_articles'], 0)
            self.redis_client.set(self.KEYS['current_article'], '')
            self.redis_client.set(self.KEYS['percentage'], 0)
            self.redis_client.set(self.KEYS['current_task_id'], task_id)
        except Exception as e:
            print(f"Error initializing training: {e}")
    
    def update_current_article(self, article_name: str) -> None:
        """
        Обновление текущей обрабатываемой статьи
        
        Args:
            article_name: Название текущей статьи
        """
        try:
            self.redis_client.set(self.KEYS['current_article'], article_name)
        except Exception as e:
            print(f"Error updating current article: {e}")
    
    def increment_processed_articles(self) -> None:
        """
        Увеличение счетчика обработанных статей и обновление процента
        """
        try:
            processed = self.redis_client.incr(self.KEYS['processed_articles'])
            total = int(self.redis_client.get(self.KEYS['total_articles']) or 0)
            
            if total > 0:
                percentage = (processed / total) * 100
                self.redis_client.set(self.KEYS['percentage'], round(percentage, 1))
        except Exception as e:
            print(f"Error incrementing processed articles: {e}")
    
    def get_training_progress(self) -> Dict[str, Any]:
        """
        Получение текущего статуса обучения
        
        Returns:
            Словарь с информацией о прогрессе
        """
        try:
            current_article = self.redis_client.get(self.KEYS['current_article'])
            total_articles = self.redis_client.get(self.KEYS['total_articles'])
            processed_articles = self.redis_client.get(self.KEYS['processed_articles'])
            percentage = self.redis_client.get(self.KEYS['percentage'])
            
            return {
                'current_article': current_article.decode() if current_article else '',
                'total_articles': int(total_articles) if total_articles else 0,
                'processed_articles': int(processed_articles) if processed_articles else 0,
                'percentage': float(percentage) if percentage else 0.0
            }
        except Exception as e:
            # В случае ошибки возвращаем пустые значения
            return {
                'current_article': '',
                'total_articles': 0,
                'processed_articles': 0,
                'percentage': 0.0
            }
    
    def clear_training_progress(self) -> None:
        """
        Очистка информации о прогрессе обучения
        """
        try:
            for key in self.KEYS.values():
                self.redis_client.delete(key)
        except Exception as e:
            print(f"Error clearing training progress: {e}")
    
    def set_current_task_id(self, task_id: str) -> None:
        """
        Установка ID текущей задачи
        
        Args:
            task_id: ID задачи Celery
        """
        try:
            self.redis_client.set(self.KEYS['current_task_id'], task_id)
        except Exception as e:
            print(f"Error setting task ID: {e}")
    
    def get_current_task_id(self) -> Optional[str]:
        """
        Получение ID текущей задачи
        
        Returns:
            ID задачи или None если задача не запущена
        """
        try:
            task_id = self.redis_client.get(self.KEYS['current_task_id'])
            return task_id.decode() if task_id else None
        except Exception as e:
            print(f"Error getting task ID: {e}")
            return None
    
    def is_training_active(self) -> bool:
        """
        Проверка, активно ли обучение
        
        Returns:
            True если обучение активно, False иначе
        """
        return self.get_current_task_id() is not None


# Singleton instance для использования в приложении
training_status_manager = TrainingStatusManager()
