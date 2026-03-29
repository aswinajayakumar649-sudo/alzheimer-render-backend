# -*- coding: utf-8 -*-
"""
Render-ready backend for Alzheimer tracker
"""

import time
import joblib
import pandas as pd
import firebase_admin
from firebase_admin import credentials, db

# =====================================
# 1. FIREBASE SETUP
# =====================================
SERVICE_ACCOUNT_FILE = "serviceAccountKey.json"
DATABASE_URL = "https://alzheimertracker-ed473-default-rtdb.asia-southeast1.firebasedatabase.app"

if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_FILE)
    firebase_admin.initialize_app(cred, {
        'databaseURL': DATABASE_URL
    })

print("[OK] Firebase connected")

# =====================================
# 2. LOAD TRAINED MODEL
# =====================================
MODEL_FILE = "wandering_detector_model.pkl"
model = joblib.load(MODEL_FILE)

print("[OK] ML model loaded")

# =====================================
# 3. FIREBASE REFERENCES
# =====================================
live_ref = db.reference("/alzheimer_project/live")
status_ref = db.reference("/alzheimer_project/status")

print("[INFO] Starting real-time inference...\n")

# =====================================
# 4. CONTINUOUS LOOP
# =====================================
while True:
    try:
        live_data = live_ref.get()

        if live_data is None:
            print("[WARN] No live data found in /alzheimer_project/live")
            time.sleep(3)
            continue

        ax = float(live_data.get("ax", 0))
        ay = float(live_data.get("ay", 0))
        az = float(live_data.get("az", 0))
        gx = float(live_data.get("gx", 0))
        gy = float(live_data.get("gy", 0))
        gz = float(live_data.get("gz", 0))
        accelMag = float(live_data.get("accelMag", 0))
        gyroMag = float(live_data.get("gyroMag", 0))
        distanceFromHome = float(live_data.get("distanceFromHome", 0))
        geofenceStatus = int(live_data.get("geofenceStatus", 0))

        features = pd.DataFrame([{
            'ax': ax,
            'ay': ay,
            'az': az,
            'gx': gx,
            'gy': gy,
            'gz': gz,
            'accelMag': accelMag,
            'gyroMag': gyroMag,
            'distanceFromHome': distanceFromHome,
            'geofenceStatus': geofenceStatus
        }])

        prediction = int(model.predict(features)[0])

        # Stationary filter
        if gyroMag < 20 and accelMag < 500:
            routeAnomaly = 0
        else:
            routeAnomaly = prediction

        # Lost logic
        if geofenceStatus == 1 and routeAnomaly == 1:
            lostStatus = 1
        else:
            lostStatus = 0

        # Write result to Firebase
        status_ref.update({
            "routeAnomaly": routeAnomaly,
            "lostStatus": lostStatus
        })

        # Debug output
        print("===================================")
        print(f"AX:{ax} AY:{ay} AZ:{az}")
        print(f"GX:{gx} GY:{gy} GZ:{gz}")
        print(f"AccelMag: {accelMag}")
        print(f"GyroMag: {gyroMag}")
        print(f"DistanceFromHome: {distanceFromHome}")
        print(f"GeofenceStatus: {geofenceStatus}")
        print(f"Raw ML Prediction: {prediction}")
        print(f"Filtered RouteAnomaly: {routeAnomaly}")
        print(f"Lost Status: {lostStatus}")
        print("===================================\n")

        time.sleep(3)

    except Exception as e:
        print("[ERROR]", e)
        time.sleep(5)