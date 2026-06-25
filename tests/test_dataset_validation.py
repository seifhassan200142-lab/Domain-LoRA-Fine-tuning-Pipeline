from scripts.validate_dataset import validate_record


def test_validate_record_accepts_valid_row():
    row = {
        "instruction": "Return JSON.",
        "input": "My product is broken.",
        "output": '{"category":"Defective Product","priority":"High","suggested_reply":"Sorry.","next_action":"Open case"}',
    }
    assert validate_record(row, 1) == []


def test_validate_record_rejects_missing_field():
    row = {"instruction": "Return JSON.", "input": "Hello"}
    errors = validate_record(row, 1)
    assert any("output" in error for error in errors)
