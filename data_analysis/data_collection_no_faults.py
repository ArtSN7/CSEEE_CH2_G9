import paho.mqtt.client as mqtt
import json
import pandas as pd
from datetime import datetime
import os
import sys

from config import BROKER_HOST, BROKER_PORT, CONNECTION_TIMEOUT, STREAM_TYPES


data_records = []
csv_filename = None
df_buffer = None 


def flatten_payload(payload):
    try:
        flat = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            # Temperature
            "temp_mean": payload["temperature_C"]["mean"],
            "temp_min": payload["temperature_C"]["min"],
            "temp_max": payload["temperature_C"]["max"],
            # pH
            "ph_mean": payload["pH"]["mean"],
            "ph_min": payload["pH"]["min"],
            "ph_max": payload["pH"]["max"],
            # RPM
            "rpm_mean": payload["rpm"]["mean"],
            "rpm_min": payload["rpm"]["min"],
            "rpm_max": payload["rpm"]["max"],
            # Actuators 
            "heater_pwm": payload["actuators_avg"]["heater_pwm"],
            "motor_pwm": payload["actuators_avg"]["motor_pwm"],
            "acid_pwm": payload["actuators_avg"]["acid_pwm"],
            "base_pwm": payload["actuators_avg"]["base_pwm"],
            # Dosing amounts
            "acid_dose_l": payload["dosing_l"]["acid"],
            "base_dose_l": payload["dosing_l"]["base"],
            # Setpoints (useful to keep!)
            "setpoint_temp": payload["setpoints"]["temperature_C"],
            "setpoint_ph": payload["setpoints"]["pH"],
            "setpoint_rpm": payload["setpoints"]["rpm"],
            # Fault ground truth (only present in test streams)
            "faults_active": (
                ", ".join(payload["faults"]["last_active"])
                if payload["faults"]["last_active"]
                else "none"
            ),
            "has_fault": int(len(payload["faults"]["last_active"]) > 0),
        }
        return flat
    except KeyError as e:
        print(f"Warning: Missing expected key {e}, skipping sample")
        return None


# saves very 100 samples or when forced
def save_buffer_to_disk(force=False):
    global df_buffer, csv_filename

    if df_buffer is not None and (len(df_buffer) >= 100 or force):

        header = not os.path.exists(csv_filename)
        df_buffer.to_csv(csv_filename, mode="a", header=header, index=False)
        df_buffer = None  


def on_connect(client, userdata, flags, rc):
    global csv_filename, df_buffer
    if rc == 0:
        print(f"Connected successfully to {BROKER_HOST}:{BROKER_PORT}")
        topic = userdata["topic"]
        client.subscribe(topic)
        print(f"Subscribed to: {topic}")

        # Create timestamped CSV file
        stream_name = topic.split("/")[-3]  # e.g. nofaults, single_fault, etc.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"data_{stream_name}_{timestamp}.csv"
        df_buffer = pd.DataFrame()

        print(f"→ All data will be saved to: {csv_filename}")
        if stream_name == "nofaults":
            print("→ Running in BASELINE mode → only fault-free samples will be saved")
        else:
            print("→ Running in TEST mode → all samples + fault labels will be saved")

        print("-" * 80)
    else:
        print(f"Connection failed with code: {rc}")


def on_message(client, userdata, msg):
    global data_records, df_buffer

    try:
        payload = json.loads(msg.payload.decode())
        flat = flatten_payload(payload)
        if flat is None:
            return

        # Decide whether to keep this sample
        stream_type = msg.topic.split("/")[-3]
        keep = True

        if stream_type == "nofaults":
            # Only keep truly clean samples for training
            if flat["has_fault"]:
                keep = False
            else:
                flat["faults_active"] = "none"
                flat["has_fault"] = 0

        if keep:
            # Add to buffer
            df_buffer = pd.concat([df_buffer, pd.DataFrame([flat])], ignore_index=True)
            save_buffer_to_disk()  # auto-save every 100

            # Live counter
            count = len(df_buffer) + (len(data_records) if data_records else 0)
            faults = flat["faults_active"]
            status = "ANOMALY" if flat["has_fault"] else "Normal"
            print(f"[{count:4d}] {flat['timestamp']} | {status:6} | Faults: {faults}")

    except Exception as e:
        print(f"Error processing message: {e}")


def on_disconnect(client, userdata, rc):
    global df_buffer
    save_buffer_to_disk(force=True)  # final save
    if rc != 0:
        print("Unexpected disconnection")
    else:
        print("Disconnected cleanly. All data saved.")


def connect_and_listen(stream_type="nofaults", duration=None):
    if stream_type not in STREAM_TYPES:
        return

    topic = STREAM_TYPES[stream_type]

    client = mqtt.Client(userdata={"topic": topic})

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    try:
        print(f"Connecting to {BROKER_HOST}:{BROKER_PORT} → {stream_type}")
        client.connect(BROKER_HOST, BROKER_PORT, CONNECTION_TIMEOUT)

        if duration:
            client.loop_start()
            import time

            time.sleep(duration)
            client.loop_stop()
        else:
            client.loop_forever()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        client.disconnect()


if __name__ == "__main__":

    stream = sys.argv[1] if len(sys.argv) > 1 else "nofaults"
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else None

    connect_and_listen(stream, duration)
