"""
Streamlit application for querying a trained classifier deployed as a CML Model.

Set the following environment variables on the CML Application to pre-populate
the connection settings in the sidebar:

    CML_MODEL_PREDICT_URL   Full predict endpoint URL from the CML Models page.
    CML_MODEL_ACCESS_KEY    Access key shown on the model detail page.

Both values can also be entered or overridden directly in the sidebar at runtime.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_FEATURES = 20


def _random_features() -> List[float]:
    """Return a random 20-element feature vector similar to the training data."""
    rng = np.random.default_rng()
    return rng.standard_normal(_N_FEATURES).round(4).tolist()


def _call_model(features: List[float]) -> Dict[str, Any]:
    """POST the feature vector to the deployed CML Model endpoint."""
    url = (st.session_state.get("model_url") or "").strip()
    access_key = (st.session_state.get("access_key") or "").strip()

    if not url:
        return {"error": "No model URL configured. Enter it in the sidebar."}

    payload = {
        "accessKey": access_key,
        "request": {"features": features},
    }

    try:
        r = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        r.raise_for_status()
        body = r.json()
        # CML wraps the handler return value under a "response" key.
        return body.get("response", body)
    except requests.RequestException as exc:
        return {
            "error": str(exc),
            "hint": "Check CML_MODEL_PREDICT_URL and CML_MODEL_ACCESS_KEY.",
        }


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="MLflow Classifier — CML Model Demo",
        page_icon="🤖",
        layout="wide",
    )

    # ---- Sidebar: connection settings -----
    with st.sidebar:
        st.header("CML Model connection")
        st.caption(
            "Pre-populate these by setting `CML_MODEL_PREDICT_URL` and "
            "`CML_MODEL_ACCESS_KEY` as environment variables on this CML Application."
        )

        if "model_url" not in st.session_state:
            st.session_state.model_url = os.getenv("CML_MODEL_PREDICT_URL", "")
        if "access_key" not in st.session_state:
            st.session_state.access_key = os.getenv("CML_MODEL_ACCESS_KEY", "")

        st.session_state.model_url = st.text_input(
            "Predict URL",
            value=st.session_state.model_url,
            placeholder="https://<cml-host>/api/v1/projects/.../models/.../predict",
        )
        st.session_state.access_key = st.text_input(
            "Access key",
            value=st.session_state.access_key,
            type="password",
            placeholder="Paste the model access key here",
        )

        st.divider()
        connected = bool(
            st.session_state.model_url.strip() and st.session_state.access_key.strip()
        )
        if connected:
            st.success("Endpoint configured", icon="✅")
        else:
            st.warning("Enter a URL and access key to enable predictions.", icon="⚠️")

    # ---- Main area -----
    st.title("MLflow Classifier — CML Model Demo")
    st.markdown(
        "This app sends a 20-feature vector to a **CML Model** endpoint backed by "
        "a scikit-learn pipeline trained with MLflow (KNN or Random Forest). "
        "Enter feature values manually, or click **Randomize** to generate a sample."
    )

    st.subheader("Feature vector")

    col_btn, col_spacer = st.columns([1, 5])
    with col_btn:
        if st.button("🎲 Randomize", use_container_width=True):
            st.session_state.features_json = json.dumps(_random_features(), indent=2)

    if "features_json" not in st.session_state:
        st.session_state.features_json = json.dumps(_random_features(), indent=2)

    features_raw = st.text_area(
        "Features (JSON array of 20 numbers)",
        value=st.session_state.features_json,
        height=160,
    )

    # Parse and validate before enabling the predict button.
    parse_error: str | None = None
    features: List[float] = []
    try:
        features = json.loads(features_raw)
        if not isinstance(features, list) or len(features) != _N_FEATURES:
            parse_error = f"Expected a JSON array of exactly {_N_FEATURES} numbers."
        elif not all(isinstance(v, (int, float)) for v in features):
            parse_error = "All values must be numbers."
    except json.JSONDecodeError as exc:
        parse_error = f"Invalid JSON: {exc}"

    if parse_error:
        st.error(parse_error)

    st.divider()

    predict_clicked = st.button(
        "▶ Predict",
        disabled=bool(parse_error) or not connected,
        type="primary",
        use_container_width=False,
    )

    if predict_clicked:
        with st.spinner("Calling model endpoint…"):
            result = _call_model(features)

        if "error" in result:
            st.error(result["error"])
            if "hint" in result:
                st.caption(result["hint"])
        else:
            prediction: int = result.get("prediction", -1)
            probabilities: List[float] = result.get("probabilities", [])
            run_id: str = result.get("run_id", "unknown")

            st.subheader("Result")
            label = "Class 1 ✅" if prediction == 1 else "Class 0 ❌"
            st.metric("Prediction", label)

            if probabilities and len(probabilities) == 2:
                col_a, col_b = st.columns(2)
                col_a.metric("P(Class 0)", f"{probabilities[0]:.1%}")
                col_b.metric("P(Class 1)", f"{probabilities[1]:.1%}")

                st.bar_chart(
                    {"Class 0": probabilities[0], "Class 1": probabilities[1]},
                    x_label="Class",
                    y_label="Probability",
                    color=["#6baed6", "#2171b5"],
                )

            with st.expander("Raw response"):
                st.json(result)

            st.caption(f"Served by MLflow run `{run_id}`")


if __name__ == "__main__":
    main()
