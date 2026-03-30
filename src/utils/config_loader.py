import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigLoader:
    """Load and access YAML configuration files."""

    def __init__(self, config_path: str = "configs/config.yaml"):
        self._config: Dict = {}
        self.load(config_path)

    def load(self, path: str):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config not found: {path}")
        with open(p) as f:
            self._config = yaml.safe_load(f) or {}

    def get(self, key: str, default: Any = None) -> Any:
        """Access nested keys with dot notation: 'detection.yolo_confidence'."""
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    def __getitem__(self, key: str) -> Any:
        return self._config[key]

    @property
    def raw(self) -> Dict:
        return self._config
