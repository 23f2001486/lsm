from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
import os
from collections import deque # Added for industrial filtering

app = Flask(__name__)
CORS(app)

# --- NEW: INDUSTRIAL SENSOR BUFFER ---
# Stores the last 5 readings to filter noise, just like a real Tesla BMS
history_buffer = deque(maxlen=5)

# --- ASSET LOADING ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "abh_model.keras")
SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ ABHONet-X Neural Engine Online. Assets loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Critical Error: Assets not found! {e}")

def get_model_output(c, t, v):
    """Processes features and returns (SoH, SoC)"""
    try:
        # Physics-Informed Feature Engineering
        arr_term = np.exp(-1 / (float(t) + 273.15))
        v_t_inter = float(v) * float(t)
        
        f_names = ["cycle", "temperature", "voltage", "arrhenius_term", "v_t_interaction"]
        df = pd.DataFrame([[float(c), float(t), float(v), arr_term, v_t_inter]], columns=f_names)
        
        scaled = scaler.transform(df)
        preds = model.predict(scaled, verbose=0)
        
        soh = float(preds[0][0][0])
        soc = float(preds[1][0][0])
        return np.clip(soh, 0, 100), np.clip(soc, 0, 100)
    except Exception as e:
        print(f"Inference Error: {e}")
        return None, None

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Input Parsing
        c_raw = float(data.get('cycle', 0))
        t_raw = float(data.get('temperature', 25))
        v_raw = float(data.get('voltage', 3.7))
        mode = data.get('mode', 'moderate')

        # --- NEW: NOISE FILTERING LAYER ---
        history_buffer.append({'c': c_raw, 't': t_raw, 'v': v_raw})
        # Calculate moving average (Filtered Data)
        c = sum(d['c'] for d in history_buffer) / len(history_buffer)
        t = sum(d['t'] for d in history_buffer) / len(history_buffer)
        v = sum(d['v'] for d in history_buffer) / len(history_buffer)

        # 1. Dual Diagnosis (Current State)
        soh, soc = get_model_output(c, t, v)
        if soh is None: 
            return jsonify({"error": "Prediction Failure"}), 500

        # --- NEW: TREND DETECTION ---
        trend = "Stable"
        if len(history_buffer) > 1:
            # Check the health of the previous buffered reading
            prev = history_buffer[-2]
            p_soh, _ = get_model_output(prev['c'], prev['t'], prev['v'])
            if p_soh and (p_soh - soh) > 0.3:
                trend = "Rapid Degradation"

        # 2. X-AI Sensitivity Analysis (Stress Testing)
        s_soh, _ = get_model_output(c, t + 10, v)
        t_impact = abs(soh - s_soh)
        v_impact = abs(v - 3.7) 
        
        # 3. Anomaly Pattern Detection
        anomaly_score = (t_impact * 2.0) + (v_impact * 5.0)
        level = "Critical" if anomaly_score > 4.0 else "Stable"

        # 4. Non-Linear Projection (Future State)
        added = {"slow": 100, "moderate": 300, "fast": 600}.get(mode, 300)
        f_soh, _ = get_model_output(c + added, t, v)

        # 5. Prescriptive Solution Engine
        solutions = []
        
        # Add Trend Warning to solutions
        if trend == "Rapid Degradation":
            solutions.append(" TREND: Health dropping fast. High discharge load detected.")

        if level == "Critical":
            solutions.append(" EMERGENCY: Immediate load reduction required.")

        if t_impact * 10 > 5:
            solutions.append(" THERMAL: Active cooling required. Check fan integrity.")
        elif t > 45:
            solutions.append(" TEMP: Ambient heat too high. Improve ventilation.")

        if v > 4.1:
            solutions.append(" VOLTAGE: Over-voltage stress. Lower charging ceiling.")
        elif v < 3.2:
            solutions.append(" DISCHARGE: Deep discharge risk. Recharge immediately.")

        if f_soh < 75:
            solutions.append(" RETIREMENT: Capacity fade accelerating. Plan replacement.")

        if not solutions:
            solutions.append(" OPTIMAL: Continue current operating parameters.")

        prescriptive_advice = " | ".join(solutions)

        # 6. Comprehensive Response
        return jsonify({
            "soh": round(soh, 2),
            "soc": round(soc, 2),
            "status": "Excellent" if soh > 85 else "Nominal" if soh > 70 else "Warning",
            "trend": trend, # Added for UI
            "advice": prescriptive_advice,
            "future_cycle": int(c + added),
            "future_soh": round(f_soh, 2),
            "anomaly": {
                "score": round(anomaly_score, 4),
                "level": level
            },
            "explain": {
                "Thermal Stress": round(t_impact * 10, 1),
                "Voltage Strain": round(v_impact * 20, 1),
                "Cycle Aging": 1.5
            }
        })

    except Exception as e:
        print(f"Backend Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)