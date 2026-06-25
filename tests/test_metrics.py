from scripts.metrics import category_match_rate, compute_metrics, field_completeness, valid_json_rate


def test_valid_json_rate():
    outputs = ['{"category":"A"}', "not json"]
    assert valid_json_rate(outputs) == 0.5


def test_field_completeness():
    outputs = ['{"category":"A","priority":"High","suggested_reply":"Hi","next_action":"Do"}']
    assert field_completeness(outputs) == 1.0


def test_category_match_rate():
    outputs = ['{"category":"Damaged Item","priority":"High","suggested_reply":"Hi","next_action":"Replace"}']
    expected = ['{"category":"Damaged Item","priority":"High","suggested_reply":"Hi","next_action":"Replace"}']
    assert category_match_rate(outputs, expected) == 1.0


def test_compute_metrics_keys():
    metrics = compute_metrics(['{"category":"A","priority":"High","suggested_reply":"Hi","next_action":"Do"}'])
    assert "valid_json_rate" in metrics
    assert "field_completeness" in metrics
