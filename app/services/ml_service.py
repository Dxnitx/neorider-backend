"""Compatibility imports for callers of the former ML service."""

import logging
from typing import Optional

from app.services.mmc1_model_service import (  # noqa: F401
    MODEL_USED,
    load_model as _load_artifacts,
    model_status as model_health_check,
    predict_safety,
)

MODEL_LABEL_TO_RISK_LABEL = {
    "SAFE": "safe_riding",
    "RISK": "high_risk",
    "ACCIDENT": "possible_accident",
}

logger = logging.getLogger("neorider.ml")


def map_model_label_to_risk_label(model_label: str) -> Optional[str]:
    """Map known model classes without turning missing/unknown data into risk."""
    normalized = str(model_label).upper() if model_label is not None else ""
    mapped = MODEL_LABEL_TO_RISK_LABEL.get(normalized)
    if mapped is None:
        logger.warning("Unknown model label %r; leaving classification unknown", model_label)
    return mapped
