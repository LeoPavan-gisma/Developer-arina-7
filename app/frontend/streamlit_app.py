import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")
REQUEST_HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

st.set_page_config(page_title="Real Estate Price Predictor", page_icon="🏠", layout="wide")
st.title("🏢 Real Estate Price Prediction System")
st.caption("Explore estimated property values and prediction activity. Estimates are based on the training data.")

predict_tab, dashboard_tab = st.tabs(["Price prediction", "Business dashboard"])

with predict_tab:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            area = st.number_input("Area (sqft)", min_value=200.0, max_value=10000.0, value=1600.0, step=25.0)
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            bathrooms = st.number_input("Bathrooms", min_value=1, max_value=8, value=2)
            age = st.number_input("Age (years)", min_value=0, max_value=60, value=12)
            floor = st.number_input("Floor", min_value=1, max_value=40, value=5)
        with col2:
            location = st.selectbox("Location", ["Bengaluru", "Hyderabad", "Pune", "Chennai", "Delhi", "Mumbai"])
            property_type = st.selectbox("Property type", ["Apartment", "Villa", "Studio", "Townhouse"])
            facing = st.selectbox("Facing", ["East", "West", "North", "South"])
            amenities_score = st.slider("Amenities score", min_value=0, max_value=100, value=75)
            distance_to_center = st.number_input("Distance to city center (km)", min_value=1.0, max_value=30.0, value=8.0, step=0.5)

        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        payload = {
            "area": float(area),
            "bedrooms": int(bedrooms),
            "bathrooms": int(bathrooms),
            "age": int(age),
            "location": location,
            "property_type": property_type,
            "floor": int(floor),
            "facing": facing,
            "amenities_score": int(amenities_score),
            "distance_to_center": float(distance_to_center),
        }

        try:
            response = requests.post(
                f"{API_URL}/api/v1/predict",
                json=payload,
                headers=REQUEST_HEADERS,
                timeout=30,
            )
            response.raise_for_status()
            result = response.json()
            st.success("Prediction complete")
            st.metric("Estimated Price", f"₹{result['predicted_price']:,.2f}")
            st.write(
                "Approximate 90% validation-error range: "
                f"₹{result['confidence_interval']['lower_bound']:,.2f} - "
                f"₹{result['confidence_interval']['upper_bound']:,.2f}"
            )
            st.caption(f"Model version: {result['model_version']} | Timestamp: {result['timestamp']}")
        except requests.RequestException as exc:
            st.error(f"Unable to get a prediction from the backend: {exc}")

with dashboard_tab:
    if st.button("Refresh analytics"):
        st.rerun()

    try:
        analytics_response = requests.get(f"{API_URL}/api/v1/dashboard", timeout=10)
        analytics_response.raise_for_status()
        analytics = analytics_response.json()
        metric_cols = st.columns(4)
        metric_cols[0].metric("Predictions", analytics["prediction_count"])
        metric_cols[1].metric("Average estimate", f"₹{analytics['average_price']:,.0f}")
        metric_cols[2].metric("Lowest estimate", f"₹{analytics['minimum_price']:,.0f}")
        metric_cols[3].metric("Highest estimate", f"₹{analytics['maximum_price']:,.0f}")

        if analytics["by_location"]:
            st.subheader("Predictions by location")
            location_rows = analytics["by_location"]
            locations = pd.DataFrame(location_rows).set_index("location")
            st.bar_chart(locations["prediction_count"])
            st.dataframe(location_rows, use_container_width=True, hide_index=True)

        if analytics["daily_activity"]:
            st.subheader("Daily prediction trend (last 30 active days)")
            daily_activity = pd.DataFrame(analytics["daily_activity"]).set_index("date")
            st.line_chart(daily_activity["prediction_count"])

        model_response = requests.get(f"{API_URL}/api/v1/model", timeout=10)
        model_response.raise_for_status()
        model = model_response.json()
        st.subheader("Active model")
        model_cols = st.columns(3)
        model_cols[0].metric("Version", model["version"])
        model_cols[1].metric("Selected model", model["model_type"])
        model_cols[2].metric("Validation R²", f"{model['metrics'].get('r2_score', 0):.3f}")
        if model["versions"]:
            st.caption(f"Registered model versions: {len(model['versions'])}")

        st.subheader("Recent predictions")
        if analytics["recent_predictions"]:
            st.dataframe(analytics["recent_predictions"], use_container_width=True, hide_index=True)
        else:
            st.info("No predictions yet. Make a prediction to populate the dashboard.")
    except requests.RequestException as exc:
        st.error(f"Unable to load dashboard analytics: {exc}")
