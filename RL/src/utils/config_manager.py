from pathlib import Path
from utils.custom_paths import PROJECT_ROOT
import yaml
from functools import lru_cache

@lru_cache(maxsize=None)
def load_full_config(cfg_name):
    """Wczytuje i parsuje plik raz, a potem trzyma go w pamięci."""
    config_path = PROJECT_ROOT / "configs" / cfg_name
    
    if not config_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}

def cfg_get(key, cfg_name, default=None):
    try:
        config = load_full_config(cfg_name)
        
        for k in key.split('.'):
            if isinstance(config, dict) and k in config:
                config = config[k]
            else:
                print(f"Key not found: {key}")
                return default 
        return config
        
    except (yaml.YAMLError, FileNotFoundError) as exc:
        print(f"Configuration error: {exc}")
        return default
