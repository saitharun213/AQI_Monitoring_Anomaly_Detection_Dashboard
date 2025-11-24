# model_saver.py - Optional: Save models (run once)
# pipenv run python model_saver.py

import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("final_processed_air_quality.csv", parse_dates=['Datetime'], dayfirst=True)
df['Datetime'] = pd.to_datetime(df['Datetime'], format='%d-%m-%Y', errors='coerce')

FEATURES = ['PM2.5', 'PM10', 'NO2', 'O3', 'AQI', 'Pollutant_Index']

for city in df['City'].unique():
    city_df = df[df['City'] == city][FEATURES].fillna(df[FEATURES].mean())
    scaler = StandardScaler().fit(city_df)
    model = IsolationForest(contamination=0.05, random_state=42).fit(scaler.transform(city_df))
    
    joblib.dump(model, f"{city.lower()}_model.pkl")
    joblib.dump(scaler, f"{city.lower()}_scaler.pkl")
    print(f"Saved model and scaler for {city}")