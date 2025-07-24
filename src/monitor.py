from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import pandas as pd



LOG_PATH = Path(__file__).resolve().parents[1] / "predictions.log"


def log_prediction(features: Dict[str, Any], prediction: Dict[str, Any]) -> None:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "features": features,
        "prediction": prediction,
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


