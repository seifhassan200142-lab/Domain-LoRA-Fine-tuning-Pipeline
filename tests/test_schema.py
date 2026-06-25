from app.schemas.models import PredictionRequest, StructuredSupportResponse


def test_prediction_request_schema():
    request = PredictionRequest(text="Where is my order?")
    assert request.text == "Where is my order?"


def test_structured_support_response_schema():
    response = StructuredSupportResponse(
        category="Shipping Delay",
        priority="Medium",
        suggested_reply="I can help check tracking.",
        next_action="Check carrier tracking",
    )
    assert response.category == "Shipping Delay"
