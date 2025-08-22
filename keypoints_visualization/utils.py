import json
from pathlib import Path
from typing import Dict, Union

def load_json(json_path: Union[str, Path]) -> Dict:
    """Load JSON data from a file."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    with open(json_path, 'r') as f:
        return json.load(f)
