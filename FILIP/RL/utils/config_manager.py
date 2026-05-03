from pathlib import Path
import yaml

p=Path(__file__).resolve().parents[1]/"config.yaml"
CONFIG=yaml.safe_load(p.read_text())
    
def get(key,default=None):
    d=CONFIG
    for k in key.split('.'):
        if isinstance(d,dict) and k in d:
            d=d[k]
        else:
            return default
    return d
