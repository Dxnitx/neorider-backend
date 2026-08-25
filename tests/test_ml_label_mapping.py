from app.services.ml_service import map_model_label_to_risk_label


def test_known_model_labels_map_explicitly():
    assert map_model_label_to_risk_label("SAFE") == "safe_riding"
    assert map_model_label_to_risk_label("RISK") == "high_risk"
    assert map_model_label_to_risk_label("ACCIDENT") == "possible_accident"


def test_unknown_or_missing_label_never_defaults_to_high_risk():
    assert map_model_label_to_risk_label(None) is None
    assert map_model_label_to_risk_label("collecting") is None
    assert map_model_label_to_risk_label("unknown") is None
