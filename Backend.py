from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import uvicorn

app = FastAPI(title="AQI Monitoring & Anomaly Detection API", version="1.0.0")

# CORS for Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data at module level (runs on import)
print("Loading CSV...")
df = pd.read_csv("final_processed_air_quality.csv", parse_dates=['Datetime'], dayfirst=True)
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y', errors='coerce')
print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

# Features for anomaly detection
FEATURES = ['PM2.5', 'PM10', 'NO2', 'O3', 'AQI', 'Pollutant_Index']

# Train models per city
models = {}
scalers = {}
for city in df['City'].unique():
    city_df = df[df['City'] == city][FEATURES].fillna(df[FEATURES].mean())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(city_df)
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(scaled_data)
    models[city] = model
    scalers[city] = scaler
print("Models loaded for all cities.")

# Pydantic models
class AnomalyRequest(BaseModel):
    city: str
    date: Optional[datetime] = None

class DataResponse(BaseModel):
    data: List[dict]

class AnomalyResponse(BaseModel):
    anomalies: List[dict]

@app.get("/health")
def health_check():
    return {"status": "healthy", "rows": len(df) if df is not None else 0}

@app.get("/cities", response_model=List[str])
def get_cities():
    return df['City'].unique().tolist()

@app.get("/data", response_model=DataResponse)
def get_data(
    city: Optional[str] = Query(None),
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None)
):
    filtered_df = df.copy()
    if city:
        filtered_df = filtered_df[filtered_df['City'] == city]
    if start_date:
        filtered_df = filtered_df[filtered_df['Datetime'] >= pd.to_datetime(start_date)]
    if end_date:
        filtered_df = filtered_df[filtered_df['Datetime'] <= pd.to_datetime(end_date)]
    data_list = filtered_df.to_dict(orient='records')[:100]
    print(f"/data returned {len(data_list)} rows for city={city}")  # Debug
    return {"data": data_list}

@app.post("/anomaly", response_model=AnomalyResponse)
def detect_anomaly(request: AnomalyRequest):
    if request.city not in models:
        raise HTTPException(status_code=400, detail="City not found")
    
    city_df = df[df['City'] == request.city]
    if request.date:
        city_df = city_df[city_df['Datetime'].dt.date == request.date.date()]
    
    if city_df.empty:
        raise HTTPException(status_code=404, detail="No data for criteria")
    
    scaler = scalers[request.city]
    model = models[request.city]
    features_data = city_df[['PM2.5', 'PM10', 'NO2', 'O3', 'AQI', 'Pollutant_Index']].fillna(df[['PM2.5', 'PM10', 'NO2', 'O3', 'AQI', 'Pollutant_Index']].mean())
    scaled = scaler.transform(features_data)
    predictions = model.predict(scaled)
    
    anomalies = city_df[predictions == -1].to_dict(orient='records')
    return {"anomalies": anomalies}

@app.get("/anomalies/{city}")
def get_anomalies_city(city: str):
    if city not in models:
        raise HTTPException(status_code=400, detail="City not found")
    
    # Pre-compute anomalies from loaded data
    city_df = df[df['City'] == city][['PM2.5', 'PM10', 'NO2', 'O3', 'AQI', 'Pollutant_Index']].fillna(df[['PM2.5', 'PM10', 'NO2', 'O3', 'AQI', 'Pollutant_Index']].mean())
    scaler = scalers[city]
    model = models[city]
    scaled = scaler.transform(city_df)
    predictions = model.predict(scaled)
    full_city_df = df[df['City'] == city].copy()
    full_city_df['Anomaly'] = predictions  # Use existing 'Anomaly' key for consistency
    anomalies = full_city_df[predictions == -1].to_dict(orient='records')
    return {"anomalies": anomalies[:50]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)