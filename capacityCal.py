import pandas as pd
import os
import numpy as np

DATA_PATH = r"C:\Users\divya\.cache\kagglehub\datasets\patrickfleith\nasa-battery-dataset\versions\2\cleaned_dataset\data"
files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".csv")])

processed_data = []

print(" Extracting NASA Battery Telemetry...")

for i, file in enumerate(files):
    file_path = os.path.join(DATA_PATH, file)
    df = pd.read_csv(file_path)
    
    # 1. Identify Columns (NASA datasets often use these names)
    current_col = "Current_measured" if "Current_measured" in df.columns else "Battery_current"
    temp_col = "Temperature_measured" if "Temperature_measured" in df.columns else "Temperature"
    volt_col = "Voltage_measured" if "Voltage_measured" in df.columns else "Voltage_battery"
    time_col = "Time"

    if current_col not in df.columns or time_col not in df.columns:
        continue

    # 2. Compute Capacity (Ah)
    df["delta_time"] = df[time_col].diff().fillna(0)
    capacity = abs((df[current_col] * df["delta_time"]).sum() / 3600)

    # 3. EXTRACT ENVIRONMENTAL FEATURES (This fixes your KeyError)
    # We take the mean temperature and voltage during this specific cycle
    avg_temp = df[temp_col].mean() if temp_col in df.columns else 25.0
    avg_volt = df[volt_col].mean() if volt_col in df.columns else 3.7

    processed_data.append([i + 1, capacity, avg_temp, avg_volt])

# 4. Create the enriched CSV
capacity_df = pd.DataFrame(processed_data, columns=["cycle", "capacity", "temperature", "voltage"])

# Remove any rows with 0 capacity (errors in NASA raw data)
capacity_df = capacity_df[capacity_df["capacity"] > 0.1]

capacity_df.to_csv("battery_capacity.csv", index=False)
print(f" Created battery_capacity.csv with {len(capacity_df)} cycles and 4 features!")