import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

# =========================================================
# CONFIG
# =========================================================
API_URL = "http://127.0.0.1:8000/optimized_route"

st.set_page_config(
    page_title="Smart Traffic Route Optimization",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================
st.title("🚦 Smart Traffic Route Optimization")
st.caption(
    "Real-time traffic-aware routing using Machine Learning + OpenStreetMap"
)

st.divider()

# =========================================================
# SIDEBAR – ROUTE INPUT
# =========================================================
st.sidebar.header("🛣️ Route Input")

source_lat = st.sidebar.number_input(
    "Source Latitude", value=13.0827, format="%.6f"
)
source_lon = st.sidebar.number_input(
    "Source Longitude", value=80.2707, format="%.6f"
)

dest_lat = st.sidebar.number_input(
    "Destination Latitude", value=13.0950, format="%.6f"
)
dest_lon = st.sidebar.number_input(
    "Destination Longitude", value=80.2650, format="%.6f"
)

st.sidebar.divider()

st.sidebar.header("🚗 Vehicle & Environment")

speed = st.sidebar.slider("Speed (km/h)", 5, 100, 30)
front_distance = st.sidebar.slider("Front Distance (m)", 1, 50, 10)
back_distance = st.sidebar.slider("Back Distance (m)", 1, 50, 8)
jerk = st.sidebar.slider("Jerk", 0.0, 1.5, 0.3)

temperature = st.sidebar.slider("Temperature (°C)", 0, 50, 30)
humidity = st.sidebar.slider("Humidity (%)", 0, 100, 60)

# Fixed sensor defaults (hidden)
Ax, Ay, Az = 0.4, 0.5, 0.6
Gx, Gy, Gz = 0.2, 0.2, 0.1

# =========================================================
# API CALL FUNCTION
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
        "source_lat": source_lat,
        "source_lon": source_lon,
        "dest_lat": dest_lat,
        "dest_lon": dest_lon
    }

    response = requests.post(API_URL, json=payload, timeout=10)
    response.raise_for_status()
    return response.json()

# =========================================================
# ACTION BUTTON
# =========================================================
if st.button("🚀 Find Optimized Route"):

    with st.spinner("Computing optimal route on real city map..."):
        result = call_api()

    st.divider()

    # =====================================================
    # METRICS
    # =====================================================
    c1, c2, c3 = st.columns(3)

    c1.metric("Traffic Level", result["traffic_status"])
    c2.metric("Confidence", f"{result['confidence_percent']} %")
    c3.metric("ETA", f"{result['eta_minutes']} min")

    st.metric(
        "Total Distance",
        f"{result['total_distance_km']} km"
    )

    # =====================================================
    # MAP VISUALIZATION
    # =====================================================
    st.subheader("🗺️ Optimized Route (Real Roads)")

    polyline = result["route_polyline"]

    # Map centered at source
    m = folium.Map(
        location=polyline[0],
        zoom_start=14,
        control_scale=True
    )

    # Route polyline
    folium.PolyLine(
        polyline,
        color="blue",
        weight=6,
        opacity=0.9,
        tooltip="Optimized Route"
    ).add_to(m)

    # Source marker
    folium.Marker(
        polyline[0],
        tooltip="Source",
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)

    # Destination marker
    folium.Marker(
        polyline[-1],
        tooltip="Destination",
        icon=folium.Icon(color="red", icon="stop")
    ).add_to(m)

    # Vehicle marker
    folium.Marker(
        [source_lat, source_lon],
        tooltip="Vehicle Location",
        icon=folium.Icon(color="orange", icon="car")
    ).add_to(m)

    st_folium(m, width=1000, height=550)

    # =====================================================
    # RAW OUTPUT (OPTIONAL – FOR DEBUG / VIVA)
    # =====================================================
    with st.expander("🔍 API Response (Debug / Viva)"):
        st.json(result)

# =========================================================
# FOOTER
# =========================================================
st.caption(
    "Built using FastAPI, Streamlit, OpenStreetMap (OSMnx), and Machine Learning"
)
