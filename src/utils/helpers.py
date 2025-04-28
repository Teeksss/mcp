from typing import Dict, List, Any, Optional
import hashlib
import json
import yaml
from datetime import datetime, timezone
import numpy as np
import logging

logger = logging.getLogger(__name__)

def generate_id(prefix: str, **kwargs) -> str:
    """Generate unique ID for resources"""
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    hash_input = f"{prefix}_{timestamp}_{json.dumps(kwargs, sort_keys=True)}"
    hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
    return f"{prefix}_{timestamp}_{hash_value}"

def load_yaml_file(file_path: str) -> Dict:
    """Load and parse YAML file"""
    try:
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load YAML file {file_path}: {e}")
        raise

def deep_merge(dict1: Dict, dict2: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if (
            key in result and 
            isinstance(result[key], dict) and 
            isinstance(value, dict)
        ):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def format_metrics(metrics: Dict) -> Dict:
    """Format metrics for storage and display"""
    formatted = {}
    
    for key, value in metrics.items():
        if isinstance(value, (np.float32, np.float64)):
            formatted[key] = float(value)
        elif isinstance(value, (np.int32, np.int64)):
            formatted[key] = int(value)
        elif isinstance(value, np.ndarray):
            formatted[key] = value.tolist()
        else:
            formatted[key] = value
    
    return formatted

def validate_config(config: Dict, required_fields: List[str]) -> bool:
    """Validate configuration dictionary"""
    return all(field in config for field in required_fields)

def parse_size_string(size_str: str) -> int:
    """Parse size string (e.g., '1Gi') to bytes"""
    units = {
        'K': 1024,
        'M': 1024 ** 2,
        'G': 1024 ** 3,
        'T': 1024 ** 4
    }
    
    size = size_str.strip()
    unit = size[-2:] if size[-2:].startswith(tuple(units.keys())) else size[-1:]
    value = float(size[:-len(unit)])
    
    if unit.endswith('i'):
        multiplier = units[unit[0]]
    else:
        multiplier = 1000 ** len(unit)
    
    return int(value * multiplier)