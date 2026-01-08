from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import requests
import os

from dotenv import load_dotenv
# =========================================================
# LOAD ENV
# =========================================================
load_dotenv()
ORS_API_KEY = os.getenv("ORS_API_KEY")

if ORS_API_KEY is None:
    raise RuntimeError("ORS_API_KEY not found in environment variables")

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="Smart Traffic Route Optimization (OpenRouteService)",
    version="5.0"
)

# =========================================================
# LOAD ML ARTIFACTS
# =========================================================
model = joblib.load("traffic_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================================================
# TRAFFIC MULTIPLIERS
# =========================================================
TRAFFIC_WEIGHT = {
    0: 1.0,   # Low
    1: 1.3,   # Medium
    2: 1.6,   # High
    3: 2.0    # Heavy
}

# =========================================================
# INPUT SCHEMA
# =========================================================
class RawSensorInput(BaseModel):
    speed: float
    Ax: float
    Ay: float
    Az: float
    Gx: float
    Gy: float
    Gz: float
    front_distance: float
    back_distance: float
    jerk: float
    temperature: float
    humidity: float

    source_lat: float
    source_lon: float
    dest_lat: float
    dest_lon: float

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def engineer_features(raw: RawSensorInput):

    df = pd.DataFrame([{
        "speed(km/h)": raw.speed,
        "Ax(m/s2)": raw.Ax,
        "Ay(m/s2)": raw.Ay,
        "Az(m/s2)": raw.Az,
        "Gx(rad/s)": raw.Gx,
        "Gy(rad/s)": raw.Gy,
        "Gz(rad/s)": raw.Gz,
        "front_distance(m)": raw.front_distance,
        "back_distance(m)": raw.back_distance,
        "jerk": raw.jerk,
        "temperature(°C)": raw.temperature,
        "humidity(%)": raw.humidity,
        "latitude": raw.source_lat,
        "longitude": raw.source_lon
    }])

    df["acc_magnitude"] = np.sqrt(
        df["Ax(m/s2)"]**2 +
        df["Ay(m/s2)"]**2 +
        df["Az(m/s2)"]**2
    )

    df["gyro_magnitude"] = np.sqrt(
        df["Gx(rad/s)"]**2 +
        df["Gy(rad/s)"]**2 +
        df["Gz(rad/s)"]**2
    )

    df["speed_distance_ratio"] = (
        df["speed(km/h)"] / (df["front_distance(m)"] + 1)
    )

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]

# =========================================================
# ORS ROUTING
# =========================================================
import polyline
import requests

def get_route_from_ors(src_lat, src_lon, dst_lat, dst_lon):
    url = "https://api.openrouteservice.org/v2/directions/driving-car"

    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "coordinates": [
            [src_lon, src_lat],
            [dst_lon, dst_lat]
        ]
    }

    response = requests.post(url, json=payload, headers=headers, timeout=20)

    print("ORS STATUS:", response.status_code)

    data = response.json()
    print("ORS RAW RESPONSE:", data)

    route = data["routes"][0]

    coords = polyline.decode(route["geometry"])
    distance_km = route["summary"]["distance"] / 1000
    duration_min = route["summary"]["duration"] / 60

    return coords, distance_km, duration_min

# =========================================================
# API ENDPOINT
# =========================================================
@app.post("/optimized_route")
def optimized_route(data: RawSensorInput):
    try:
        # ---- ML Traffic Prediction ----
        features = engineer_features(data)
        scaled = scaler.transform(features)

        probs = model.predict_proba(scaled)[0]
        traffic_status = int(np.argmax(probs))
        confidence = round(float(np.max(probs)) * 100, 2)

        traffic_multiplier = TRAFFIC_WEIGHT[traffic_status]

        # ---- ORS Route ----
        polyline_coords, distance_km, base_eta = get_route_from_ors(
            data.source_lat,
            data.source_lon,
            data.dest_lat,
            data.dest_lon
        )

        eta_minutes = round(base_eta * traffic_multiplier, 2)

        return {
            "traffic_status": traffic_status,
            "confidence_percent": confidence,
            "total_distance_km": round(distance_km, 2),
            "eta_minutes": eta_minutes,
            "route_polyline": polyline_coords
        }

    except Exception as e:
        print("🔥 BACKEND CRASH:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
