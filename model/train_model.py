import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# 1. LOAD & PREPARE DATA
data = pd.read_csv("../battery_capacity.csv")
initial_cap = data["capacity"].iloc[0]

# --- FEATURE 1: SoH Calculation ---
data["SoH"] = (data["capacity"] / initial_cap) * 100

# --- FEATURE 2: SoC Estimation (Training Target) ---
# We normalize voltage to 0-100 to teach the model SoC mapping
data["SoC"] = ((data["voltage"] - 3.0) / (4.2 - 3.0)) * 100 
data["SoC"] = data["SoC"].clip(0, 100)

# --- FEATURE 3: Adaptive Physics-Informed Engineering ---
# Arrhenius term (Chemical Aging) and V-T Interaction (Electro-thermal stress)
data['arrhenius_term'] = np.exp(-1 / (data['temperature'] + 273.15))
data['v_t_interaction'] = data['voltage'] * data['temperature']

# Inputs for the Model
features = ["cycle", "temperature", "voltage", "arrhenius_term", "v_t_interaction"]
X = data[features]

# --- DUAL TARGETS: SoH and SoC ---
y = data[["SoH", "SoC"]] 

# 2. SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# --- 3. DUAL-HEAD NEURAL ARCHITECTURE ---
# This architecture is required for the "Dual Prediction Model" patent claim
input_layer = tf.keras.layers.Input(shape=(len(features),))

# Shared Adaptive Base
x = tf.keras.layers.Dense(256, activation="swish")(input_layer)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.1)(x) # Feature 4: Dropout used for Anomaly/Uncertainty detection

x = tf.keras.layers.Dense(128, activation="swish")(x)
x = tf.keras.layers.Dense(64, activation="relu")(x)

# Output Head 1: State of Health (SoH)
soh_output = tf.keras.layers.Dense(1, name="soh_output")(x)

# Output Head 2: State of Charge (SoC)
soc_output = tf.keras.layers.Dense(1, name="soc_output")(x)

model = tf.keras.Model(inputs=input_layer, outputs=[soh_output, soc_output])

# 4. TRAINING
# --- 4. TRAINING (FIXED FOR MULTI-OUTPUT) ---
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    # We specify which loss goes to which output head
    loss={"soh_output": "huber", "soc_output": "mse"},
    # FIXED: We provide a dictionary for metrics to match the 2 outputs
    metrics={"soh_output": ["mae"], "soc_output": ["mae"]}
)

print(" Training Multi-Task ABHONet Engine...")

# We convert y_train to a dictionary to ensure the targets match the output heads
model.fit(
    X_train, 
    {"soh_output": y_train["SoH"], "soc_output": y_train["SoC"]}, 
    epochs=350, 
    batch_size=16, 
    verbose=1
)

# 5. SAVE ASSETS
model.save("abh_model.keras")
print(" Model Trained successfully with Dual Heads!")