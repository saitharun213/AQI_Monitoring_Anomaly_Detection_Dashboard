# Dashboard.py - Streamlit Frontend (Corrected: Extract 'data' from API Response)
# Run with: pipenv run streamlit run Dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta

# Config
API_BASE = "http://localhost:8000"
st.set_page_config(page_title="AQI Sentinel Dashboard", layout="wide")

@st.cache_data(ttl=300)
def fetch_data(endpoint, params=None):
    try:
        response = requests.get(f"{API_BASE}/{endpoint}", params=params, timeout=10)
        response.raise_for_status()
        json_data = response.json()
        # Extract 'data' key for /data endpoint; return as-is for others like /cities
        return json_data["data"] if "data" in json_data else json_data
    except Exception as e:
        st.error(f"API Error: {e}")
        return []

@st.cache_data(ttl=300)
def fetch_anomalies(city):
    try:
        response = requests.get(f"{API_BASE}/anomalies/{city}", timeout=10)
        response.raise_for_status()
        return response.json()["anomalies"]
    except Exception as e:
        st.error(f"API Error: {e}")
        return []

def post_anomaly_detection(city, date=None):
    payload = {"city": city}
    if date:
        payload["date"] = date.isoformat()
    try:
        response = requests.post(f"{API_BASE}/anomaly", json=payload, timeout=10)
        response.raise_for_status()
        return response.json()["anomalies"]
    except Exception as e:
        st.error(f"API Error: {e}")
        return []

st.title("🌫️ AQI Sentinel: Air Quality Monitoring & Anomaly Detection")

# Sidebar
st.sidebar.header("Filters")
cities = fetch_data("cities")
if cities:
    selected_city = st.sidebar.selectbox("City", cities, index=0)
else:
    selected_city = "Delhi"  # Fallback

col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start Date", value=datetime(2024, 1, 1))  # Default to data range
end_date = col2.date_input("End Date", value=datetime(2024, 12, 31))

if st.sidebar.button("Detect Anomalies for Date"):
    selected_date = st.sidebar.date_input("Specific Date for Detection")
    anomalies = post_anomaly_detection(selected_city, datetime(selected_date.year, selected_date.month, selected_date.day))
    if anomalies:
        st.sidebar.success(f"Found {len(anomalies)} anomalies!")
        st.sidebar.json(anomalies)

# Main Dashboard
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends", "🚨 Anomalies"])

with tab1:
    st.header("AQI Overview")
    data = fetch_data("data", {
        "city": selected_city,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d')
    })
    if data and len(data) > 0:
        df_display = pd.DataFrame(data)
        # Safe column selection: Check if columns exist
        required_cols = ['Datetime', 'AQI', 'AQI_Bucket', 'PM2.5', 'Anomaly']
        available_cols = [col for col in required_cols if col in df_display.columns]
        if len(available_cols) == len(required_cols):
            st.dataframe(df_display[required_cols].tail(50), use_container_width=True)
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                avg_aqi = df_display['AQI'].mean()
                st.metric("Avg AQI", f"{avg_aqi:.1f}")
            with col2:
                severe_days = len(df_display[df_display['AQI'] > 400])
                st.metric("Severe Days", severe_days)
            with col3:
                anomalies_count = len(df_display[df_display['Anomaly'] == -1])
                st.metric("Anomalies", anomalies_count)
        else:
            st.warning(f"Missing columns: {set(required_cols) - set(df_display.columns)}. Showing all data.")
            st.dataframe(df_display.tail(50), use_container_width=True)
    else:
        st.info("No data loaded. Ensure backend is running and filters are valid.")

with tab2:
    st.header("AQI Trends")
    data = fetch_data("data", {
        "city": selected_city,
        "start_date": start_date.strftime('%Y-%m-%d'),
        "end_date": end_date.strftime('%Y-%m-%d')
    })
    if data and len(data) > 0:
        df_plot = pd.DataFrame(data)
        # Safe hover_data
        hover_cols = ['PM2.5']
        if 'Anomaly' in df_plot.columns:
            hover_cols.append('Anomaly')
        fig = px.line(df_plot, x='Datetime', y='AQI', color='AQI_Bucket',
                      title=f"AQI Trend for {selected_city}",
                      hover_data=hover_cols)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data for trends. Check filters.")

with tab3:
    st.header("Anomalies")
    anomalies = fetch_anomalies(selected_city)
    if anomalies and len(anomalies) > 0:
        df_anom = pd.DataFrame(anomalies)
        st.dataframe(df_anom, use_container_width=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_anom['Datetime'], y=df_anom['AQI'],
                                 mode='markers', marker=dict(color='red', size=10),
                                 name='Anomalies'))
        fig.update_layout(title=f"Anomaly Spikes in {selected_city}", xaxis_title="Date", yaxis_title="AQI")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No anomalies detected for this city.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit + FastAPI | Data: Processed Air Quality CSV | Model: Isolation Forest")