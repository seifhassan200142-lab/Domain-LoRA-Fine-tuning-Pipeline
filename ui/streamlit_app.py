"""Streamlit UI for the structured customer support model."""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000/api/predict")

st.set_page_config(page_title="Support Response Formatter", page_icon="🧩")
st.title("Customer Support Response Formatter")
st.write(
    "Enter a customer complaint or question. The model returns a structured JSON-like support response."
)

example = "My order arrived damaged and I need a replacement."
complaint = st.text_area("Customer message", value=example, height=140)

if st.button("Generate response"):
    if not complaint.strip():
        st.warning("Please enter a customer message.")
    else:
        with st.spinner("Generating..."):
            try:
                response = requests.post(API_URL, json={"text": complaint}, timeout=120)
                response.raise_for_status()
                payload = response.json()
                st.subheader("Structured response")
                st.json(payload["prediction"])
                with st.expander("Raw model output"):
                    st.code(payload.get("raw_output", ""), language="text")
            except requests.RequestException as exc:
                st.error(f"API request failed: {exc}")
                st.info("Start the API with: uvicorn app.api.main:app --reload")
