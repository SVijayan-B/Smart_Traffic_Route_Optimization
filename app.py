import streamlit as st
import requests
import folium
from streamlit_folium import st_folium
import time

# =========================================================
# CONFIG
# =========================================================
API_URL = "http://127.0.0.1:8000/optimized_route"

st.set_page_config(
    page_title="Smart Traffic Management System",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================
st.title("🚦 Smart Traffic Management System")
st.write(
    "Traffic-aware route optimization using Machine Learning + Dijkstra’s Algorithm"
)

st.divider()

# =========================================================
# SIDEBAR – SIMULATION SETTINGS
# =========================================================
st.sidebar.header("⏱️ Simulation Settings")

simulate = st.sidebar.toggle("Enable Simulation", value=False)

refresh_rate = st.sidebar.slider(
    "Refresh interval (seconds)",
    min_value=3,
    max_value=10,
    value=5
)

traffic_mode = st.sidebar.selectbox(
    "Traffic Scenario",
    ["Low Traffic", "Medium Traffic", "Heavy Traffic"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("🛣️ **Route Selection**")

source = st.sidebar.selectbox("Source Node", ["A", "B", "C", "D", "E"])
destination = st.sidebar.selectbox("Destination Node", ["A", "B", "C", "D", "E"])

# =========================================================
# SIMULATION PRESETS
# =========================================================
if traffic_mode == "Low Traffic":
    preset_speed = 60.0
    preset_front_distance = 35.0
    preset_jerk = 0.05
elif traffic_mode == "Medium Traffic":
    preset_speed = 30.0
    preset_front_distance = 12.0
    preset_jerk = 0.25
else:
    preset_speed = 5.0
    preset_front_distance = 2.0
    preset_jerk = 0.9

# =========================================================
# INPUTS
# =========================================================
st.subheader("📡 Raw IoT Sensor Data")

col1, col2 = st.columns(2)

with col1:
    speed = st.slider("Speed (km/h)", 5.0, 100.0, preset_speed)
    front_distance = st.slider("Front Distance (m)", 0.5, 50.0, preset_front_distance)
    back_distance = st.slider("Back Distance (m)", 0.5, 50.0, 8.0)
    jerk = st.slider("Jerk", 0.0, 1.5, preset_jerk)

with col2:
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 30.0)
    humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
    latitude = st.number_input("Vehicle Latitude", value=13.0827)
    longitude = st.number_input("Vehicle Longitude", value=80.2707)

# Fixed sensor defaults
Ax, Ay, Az = 0.3, 0.4, 0.5
Gx, Gy, Gz = 0.1, 0.1, 0.1

# =========================================================
# API CALL
# =========================================================
def call_api():
    payload = {
        "speed": speed,
        "Ax": Ax,
        "Ay": Ay,
        "Az": Az,
        "Gx": Gx,
        "Gy": Gy,
        "Gz": Gz,
        "front_distance": front_distance,
        "back_distance": back_distance,
        "jerk": jerk,
        "temperature": temperature,
        "humidity": humidity,
        "latitude": latitude,
        "longitude": longitude,
        "source": source,
        "destination": destination
    }

    response = requests.post(API_URL, json=payload, timeout=5)
    response.raise_for_status()
    return response.json()

# =========================================================
# BUTTON CONTROLS
# =========================================================
predict_clicked = st.button("🚀 Find Optimized Route")

# Session state init
if "last_run" not in st.session_state:
    st.session_state.last_run = 0
if "result" not in st.session_state:
    st.session_state.result = None

# =========================================================
# SAFE AUTO-REFRESH (NO WHILE LOOP)
# =========================================================
current_time = time.time()

if simulate and source != destination:
    if current_time - st.session_state.last_run >= refresh_rate:
        st.session_state.result = call_api()
        st.session_state.last_run = current_time
        st.experimental_rerun()

elif predict_clicked and source != destination:
    st.session_state.result = call_api()

# =========================================================
# DISPLAY RESULTS
# =========================================================
if st.session_state.result:

    result = st.session_state.result

    st.divider()
    st.subheader("📊 Optimized Route Result")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Traffic Level", result["traffic_status"])
    c2.metric("Confidence", f"{result['confidence_percent']} %")
    c3.metric("Total Distance", f"{result['total_distance_km']} km")
    c4.metric("ETA", f"{result['eta_minutes']} min")

    # =====================================================
    # MAP WITH ROUTE POLYLINE
    # =====================================================
    st.subheader("🗺️ Optimized Route Map")

    polyline = result["polyline"]

    m = folium.Map(
        location=polyline[0],
        zoom_start=14
    )

    # Draw route polyline
    folium.PolyLine(
        polyline,
        color="blue",
        weight=6,
        tooltip="Optimized Route"
    ).add_to(m)

    # Start & End markers
    folium.Marker(
        polyline[0],
        icon=folium.Icon(color="green"),
        tooltip="Source"
    ).add_to(m)

    folium.Marker(
        polyline[-1],
        icon=folium.Icon(color="red"),
        tooltip="Destination"
    ).add_to(m)

    # Vehicle marker
    folium.Marker(
        [latitude, longitude],
        icon=folium.Icon(color="orange"),
        tooltip="Vehicle"
    ).add_to(m)

    st_folium(m, width=900, height=500)

    # =====================================================
    # PATH DETAILS
    # =====================================================
    st.subheader("🧭 Path Nodes")
    st.write(" → ".join(result["optimized_path"]))

elif source == destination:
    st.warning("Source and destination must be different.")
