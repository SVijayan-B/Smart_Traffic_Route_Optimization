# 🚦 Smart Traffic Route Optimization System

A real-time traffic-aware route optimization system that combines **Machine Learning**, **IoT-style sensor data**, and **OpenRouteService (OpenStreetMap-based routing API)** to predict traffic conditions and recommend optimized driving routes with ETA estimation.

---

## 📌 Project Overview

Urban traffic congestion is a major challenge in modern cities. This project predicts traffic conditions using a trained ML model and dynamically adjusts route ETA using real-world road data obtained from OpenRouteService.

The system is built using a **FastAPI backend** and a **Streamlit frontend**.

---

## ✨ Features

- 🚗 Traffic condition prediction (Low / Medium / High / Heavy)
- 🗺️ Real-world route optimization using OpenRouteService
- ⏱️ Dynamic ETA calculation based on traffic severity
- 📊 Interactive Streamlit dashboard with live map visualization
- 🛡️ Robust error handling for invalid routes

## 🏗️ Architecture
```bash
Streamlit Frontend
|
| JSON over HTTP
|
FastAPI Backend
|
|-- Machine Learning Model
|-- OpenRouteService API
```
---

---

## 📁 Project Structure
```bash
Smart_Traffic_Route_Optimization/
├── app.py
├── main.py
├── traffic_model.pkl
├── scaler.pkl
├── feature_names.pkl
├── requirements.txt
├── .env
└── README.md
```

---

## ⚙️ Installation

### 1. Clone Repository
```bash
  git clone https://github.com/your-username/Smart_Traffic_Route_Optimization.git
  cd Smart_Traffic_Route_Optimization

  python -m venv .venv
  .venv\Scripts\activate

  pip install -r requirements.txt
```

### 2.🔑 OpenRouteService API Key Setup
1. Visit https://openrouteservice.org
2. Create an account and generate an API key
3. Create a .env file in the project root:

```bash
  ORS_API_KEY=your_openrouteservice_api_key_here
  ⚠️ Do not commit the .env file to GitHub.
```

### 3.Running the application

```bash
  uvicorn main:app --reload //seperate terminal
  streamlit run app.py
```
