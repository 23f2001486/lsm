from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import os
from collections import deque

app = Flask(__name__)
CORS(app)

# -----------------------------
# SENSOR BUFFER (noise filter)
# -----------------------------
history_buffer = deque(maxlen=5)

# -----------------------------
# SoH TREND BUFFER
# -----------------------------
soh_history = deque(maxlen=5)

# -----------------------------
# LOAD MODEL
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "abh_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    print("ABHONet-X Neural Engine Online")
except Exception as e:
    print("Error loading assets:", e)

# --------------------------------
# MODEL INFERENCE FUNCTION
# --------------------------------
def get_model_output(c, t, v):

    try:

        arr_term = np.exp(-1 / (float(t) + 273.15))
        v_t_inter = float(v) * float(t)

        feature_names = [
            "cycle",
            "temperature",
            "voltage",
            "arrhenius_term",
            "v_t_interaction"
        ]

        df = pd.DataFrame(
            [[float(c), float(t), float(v), arr_term, v_t_inter]],
            columns=feature_names
        )

        scaled = scaler.transform(df)

        preds = model.predict(scaled, verbose=0)

        soh = float(preds[0][0][0])
        soc = float(preds[1][0][0])

        soh = np.clip(soh, 0, 100)
        soc = np.clip(soc, 0, 100)

        return soh, soc

    except Exception as e:
        print("Inference Error:", e)
        return None, None


# --------------------------------
# RUL ESTIMATION FUNCTION
# --------------------------------
def estimate_rul(current_soh, history, failure_threshold=70):

    if len(history) < 3:
        return None

    drops = []

    for i in range(1, len(history)):
        drop = history[i-1] - history[i]
        drops.append(drop)

    avg_drop = sum(drops) / len(drops)

    if avg_drop <= 0:
        return None

    rul = (current_soh - failure_threshold) / avg_drop

    return max(int(rul), 0)


# --------------------------------
# HOME PAGE
# --------------------------------
@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------
# PREDICTION API
# --------------------------------
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        if not data:
            return jsonify({"error": "No data provided"}), 400

        # -----------------------
        # INPUT DATA
        # -----------------------
        c_raw = float(data.get("cycle", 0))
        t_raw = float(data.get("temperature", 25))
        v_raw = float(data.get("voltage", 3.7))
        mode = data.get("mode", "moderate")

        # -----------------------
        # SENSOR NOISE FILTERING
        # -----------------------
        history_buffer.append({
            "c": c_raw,
            "t": t_raw,
            "v": v_raw
        })

        c = sum(d["c"] for d in history_buffer) / len(history_buffer)
        t = sum(d["t"] for d in history_buffer) / len(history_buffer)
        v = sum(d["v"] for d in history_buffer) / len(history_buffer)

        # -----------------------
        # CURRENT PREDICTION
        # -----------------------
        soh, soc = get_model_output(c, t, v)

        if soh is None:
            return jsonify({"error": "Prediction failed"}), 500

        # -----------------------
        # TREND DETECTION
        # -----------------------
        soh_history.append(soh)

        trend = "Stable"

        if len(soh_history) >= 3:

            drops = []

            for i in range(1, len(soh_history)):
                drop = soh_history[i-1] - soh_history[i]
                drops.append(drop)

            avg_drop = sum(drops) / len(drops)

            if avg_drop > 0.5:
                trend = "Rapid Degradation"

        # -----------------------
        # RUL PREDICTION
        # -----------------------
        rul_cycles = estimate_rul(soh, soh_history)

        # -----------------------
        # STRESS TESTING
        # -----------------------
        stressed_soh, _ = get_model_output(c, t + 10, v)

        thermal_impact = abs(soh - stressed_soh)
        voltage_impact = abs(v - 3.7)

        # -----------------------
        # ANOMALY SCORE
        # -----------------------
        anomaly_score = (thermal_impact * 2.0) + (voltage_impact * 5.0)

        anomaly_level = "Critical" if anomaly_score > 4 else "Stable"

        # -----------------------
        # FUTURE PROJECTION
        # -----------------------
        cycle_add = {
            "slow": 100,
            "moderate": 300,
            "fast": 600
        }.get(mode, 300)

        future_soh, _ = get_model_output(c + cycle_add, t, v)

        # -----------------------
        # ADVICE GENERATION
        # -----------------------
        advice_list = []

        if trend == "Rapid Degradation":
            advice_list.append("Battery health dropping rapidly")

        if anomaly_level == "Critical":
            advice_list.append("Immediate load reduction recommended")

        if thermal_impact * 10 > 5:
            advice_list.append("High thermal stress detected")

        if t > 45:
            advice_list.append("Temperature too high")

        if v > 4.1:
            advice_list.append("Over-voltage stress")

        if v < 3.2:
            advice_list.append("Deep discharge risk")

        if future_soh < 75:
            advice_list.append("Battery replacement planning recommended")

        if not advice_list:
            advice_list.append("Battery operating normally")

        advice = " | ".join(advice_list)

        # -----------------------
        # FINAL RESPONSE
        # -----------------------
        return jsonify({

            "soh": round(soh, 2),
            "soc": round(soc, 2),

            "status":
                "Excellent" if soh > 85
                else "Nominal" if soh > 70
                else "Warning",

            "trend": trend,

            "rul_cycles": rul_cycles,

            "future_cycle": int(c + cycle_add),

            "future_soh": round(future_soh, 2),

            "advice": advice,

            "anomaly": {
                "score": round(anomaly_score, 3),
                "level": anomaly_level
            },

            "explain": {
                "Thermal Stress": round(thermal_impact * 10, 2),
                "Voltage Strain": round(voltage_impact * 20, 2),
                "Cycle Aging": 1.5
            }

        })

    except Exception as e:

        print("Backend Error:", e)

        return jsonify({"error": str(e)}), 500


# --------------------------------
# RUN SERVER
# --------------------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)