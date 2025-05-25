import json
import yaml

def load_documents(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_settings(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

