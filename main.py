from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

import osmnx as ox
import networkx as nx

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="Smart Traffic Route Optimization (Real Map)",
    version="4.1"
)

# =========================================================
# LOAD ML ARTIFACTS
# =========================================================
model = joblib.load("traffic_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================================================
# GLOBAL GRAPH (LAZY LOADED)
# =========================================================
ROAD_GRAPH = None
CITY_NAME = "Chennai, India"

# =========================================================
# TRAFFIC MULTIPLIERS
# =========================================================
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

    source_lat: float
    source_lon: float
    dest_lat: float
    dest_lon: float

# =========================================================
# LOAD GRAPH (ON DEMAND)
# =========================================================
def get_graph():
    global ROAD_GRAPH
    if ROAD_GRAPH is None:
        print("Loading real city road network...")
        ROAD_GRAPH = ox.graph_from_place(
            CITY_NAME,
            network_type="drive",
            simplify=True
        )
        print("Road network loaded")
    return ROAD_GRAPH

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
        df["Ax(m/s2)"]**2 + df["Ay(m/s2)"]**2 + df["Az(m/s2)"]**2
    )

    df["gyro_magnitude"] = np.sqrt(
        df["Gx(rad/s)"]**2 + df["Gy(rad/s)"]**2 + df["Gz(rad/s)"]**2
    )

    df["speed_distance_ratio"] = (
        df["speed(km/h)"] / (df["front_distance(m)"] + 1)
    )

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    return df[feature_names]

# =========================================================
# API ENDPOINT
# =========================================================
@app.post("/optimized_route")
def optimized_route(data: RawSensorInput):

    # ---- ML Traffic Prediction ----
    features = engineer_features(data)
    scaled = scaler.transform(features)

    probs = model.predict_proba(scaled)[0]
    traffic_status = int(np.argmax(probs))
    confidence = round(float(np.max(probs)) * 100, 2)

    traffic_multiplier = TRAFFIC_WEIGHT[traffic_status]

    # ---- Load Graph ----
    graph = get_graph()

    # ---- Nearest Nodes ----
    source_node = ox.nearest_nodes(
        graph, data.source_lon, data.source_lat
    )
    dest_node = ox.nearest_nodes(
        graph, data.dest_lon, data.dest_lat
    )

    # ---- Shortest Path (Dijkstra) ----
    path_nodes = nx.shortest_path(
        graph,
        source_node,
        dest_node,
        weight="length"
    )

    # ---- Path Polyline ----
    polyline = [
        (graph.nodes[n]["y"], graph.nodes[n]["x"])
        for n in path_nodes
    ]

    # ---- Distance ----
    total_distance_km = sum(
        graph.edges[path_nodes[i], path_nodes[i+1], 0]["length"]
        for i in range(len(path_nodes) - 1)
    ) / 1000

    # ---- ETA ----
    eta_minutes = round(
        (total_distance_km / max(data.speed, 5)) *
        traffic_multiplier * 60,
        2
    )

    return {
        "traffic_status": traffic_status,
        "confidence_percent": confidence,
        "total_distance_km": round(total_distance_km, 2),
        "eta_minutes": eta_minutes,
        "route_polyline": polyline
    }
