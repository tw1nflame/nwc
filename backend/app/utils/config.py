import yaml
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """Загружает конфигурацию из YAML-файла"""
    try:
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info("Config loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise