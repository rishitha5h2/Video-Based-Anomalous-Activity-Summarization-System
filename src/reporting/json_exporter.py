import json
from pathlib import Path
from datetime import datetime
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class JSONExporter:
    def export(self, results: Dict, output_path: str) -> str:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now().isoformat(),
            "vigil_version": "1.0.0",
            **results,
        }
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2, default=str)
        logger.info(f"JSON exported: {output_path}")
        return output_path
