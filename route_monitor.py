import os
import time
import threading
import joblib
import pandas as pd
import firebase_admin
from flask import Flask
from firebase_admin import credentials, db

# =====================================
# FLASK APP
# =====================================
app = Flask(__name__)

# =====================================
# FIREBASE SETUP
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
# LOAD MODEL
# =====================================
MODEL_FILE = "wandering_detector_model.pkl"
model = joblib.load(MODEL_FILE)

print("[OK] ML model loaded")

# =====================================
# FIREBASE REFERENCES
# =====================================
live_ref = db.reference("/alzheimer_project/live")
status_ref = db.reference("/alzheimer_project/status")

# =====================================
# BACKGROUND ML LOOP
# =====================================
def monitor_loop():
    print("[INFO] Background monitoring thread started")

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

# =====================================
# START BACKGROUND THREAD ONLY ONCE
# =====================================
monitor_started = False

def start_monitor():
    global monitor_started
    if not monitor_started:
        monitor_started = True
        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()
        print("[OK] Monitoring thread launched")

start_monitor()

# =====================================
# FLASK ROUTES
# =====================================
@app.route("/")
def home():
    return "Alzheimer Route Monitor is running on Render!", 200

@app.route('/health')
def health():
    return {
        "status": "running",
        "backend": "render",
        "ml": "active"
    }, 200

# =====================================
# LOCAL RUN (optional)
# =====================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

