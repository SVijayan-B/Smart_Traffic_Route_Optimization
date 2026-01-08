from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import heapq

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="Smart Traffic Management System",
    version="3.0"
)

# =========================================================
# LOAD ML ARTIFACTS
# =========================================================
model = joblib.load("traffic_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================================================
# GPS NODES (REAL LOCATIONS)
# =========================================================
NODES = {
    "A": (13.0827, 80.2707),
    "B": (13.0850, 80.2750),
    "C": (13.0900, 80.2700),
    "D": (13.0950, 80.2650),
    "E": (13.1000, 80.2600)
}

# =========================================================
# ROAD GRAPH (DISTANCE IN KM)
# =========================================================
GRAPH = {
    "A": {"B": 2.0, "C": 3.5},
    "B": {"A": 2.0, "C": 1.5, "D": 4.0},
    "C": {"A": 3.5, "B": 1.5, "D": 2.5},
    "D": {"B": 4.0, "C": 2.5, "E": 3.0},
    "E": {"D": 3.0}
}

TRAFFIC_WEIGHT = {
    0: 1.0,
    1: 1.3,
    2: 1.6,
    3: 2.0
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
    latitude: float
    longitude: float
    source: str
    destination: str

# =========================================================
# FEATURE ENGINEERING
# =========================================================
def engineer_features(raw):
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
        "latitude": raw.latitude,
        "longitude": raw.longitude
    }])

    df["acc_magnitude"] = np.sqrt(
        df["Ax(m/s2)"]**2 + df["Ay(m/s2)"]**2 + df["Az(m/s2)"]**2
    )

    df["gyro_magnitude"] = np.sqrt(
        df["Gx(rad/s)"]**2 + df["Gy(rad/s)"]**2 + df["Gz(rad/s)"]**2
    )

    df["speed_distance_ratio"] = df["speed(km/h)"] / (df["front_distance(m)"] + 1)

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]

# =========================================================
# DIJKSTRA (TRAFFIC-AWARE)
# =========================================================
def dijkstra(start, end, traffic_status):
    pq = [(0, start, [])]
    visited = set()

    while pq:
        cost, node, path = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        path = path + [node]

        if node == end:
            return cost, path

        for neigh, dist in GRAPH[node].items():
            if neigh not in visited:
                weighted = dist * TRAFFIC_WEIGHT[traffic_status]
                heapq.heappush(pq, (cost + weighted, neigh, path))

    return float("inf"), []

# =========================================================
# API ENDPOINT
# =========================================================
@app.post("/optimized_route")
def optimized_route(data: RawSensorInput):

    features = engineer_features(data)
    scaled = scaler.transform(features)

    probs = model.predict_proba(scaled)[0]
    traffic_status = int(np.argmax(probs))
    confidence = round(float(np.max(probs)) * 100, 2)

    total_cost, path = dijkstra(data.source, data.destination, traffic_status)

    # Total distance
    total_distance = sum(
        GRAPH[path[i]][path[i+1]] for i in range(len(path)-1)
    )

    eta_minutes = round(
        (total_distance / max(data.speed, 5)) *
        TRAFFIC_WEIGHT[traffic_status] * 60,
        2
    )

    polyline = [NODES[node] for node in path]

    return {
        "traffic_status": traffic_status,
        "confidence_percent": confidence,
        "optimized_path": path,
        "polyline": polyline,
        "total_distance_km": round(total_distance, 2),
        "eta_minutes": eta_minutes
    }
