from app.inference.formatting import format_prompt, missing_required_fields, parse_json_output


def test_format_prompt_contains_customer_message():
    prompt = format_prompt("My order is late.")
    assert "### Instruction:" in prompt
    assert "### Customer Message:" in prompt
    assert "My order is late." in prompt
    assert prompt.endswith("### Response:\n")


def test_parse_json_output_with_extra_text():
    parsed = parse_json_output('Here is the result: {"category":"Shipping Delay","priority":"Medium","suggested_reply":"Sorry.","next_action":"Check tracking"}')
    assert parsed["category"] == "Shipping Delay"


def test_missing_required_fields():
    missing = missing_required_fields({"category": "Billing Issue"})
    assert "priority" in missing
    assert "suggested_reply" in missing
    assert "next_action" in missing
