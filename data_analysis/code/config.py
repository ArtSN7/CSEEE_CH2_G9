# MQTT Broker Settings
BROKER_HOST = "engf0001.cs.ucl.ac.uk"
BROKER_PORT = 1883
CONNECTION_TIMEOUT = 60  # secs

# Available data streams
STREAM_TYPES = {
    "nofaults": "bioreactor_sim/nofaults/telemetry/summary",
    "single_fault": "bioreactor_sim/single_fault/telemetry/summary",
    "three_faults": "bioreactor_sim/three_faults/telemetry/summary",
    "variable_setpoints": "bioreactor_sim/variable_setpoints/telemetry/summary",
}





"""
DATA WITHIN JSON: 


[Message #43] - 2025-11-24 15:08:32.071
Topic: bioreactor_sim/nofaults/telemetry/summary
--------------------------------------------------------------------------------
Data received:
{
  "window": {
    "start": 1763996910,
    "end": 1763996912,
    "seconds": 2,
    "samples": 11
  },
  "temperature_C": {
    "mean": 29.992750353782615,
    "min": 29.92378153941172,
    "max": 30.061760883079785
  },
  "pH": {
    "mean": 4.9991703135719385,
    "min": 4.9682118647635924,
    "max": 5.0499421696176325
  },
  "rpm": {
    "mean": 996.3746810025721,
    "min": 971.3449205540567,
    "max": 1020.95551756811
  },
  "actuators_avg": {
    "heater_pwm": 0.4656695856828525,
    "motor_pwm": 0.8485373848337573,
    "acid_pwm": 0.0,
    "base_pwm": 0.0
  },
  "dosing_l": {
    "acid": 2.5294548788274927e-05,
    "base": 4.405097957758599e-05
  },
  "heater_energy_Wh": 0.018847518839838245,
  "photoevents": 38,
  "setpoints": {
    "temperature_C": 30.0,
    "pH": 5.0,
    "rpm": 1000.0
  },
  "faults": {
    "last_active": [],
    "counts": {}
  }
}

"""


"""
# 1. Collect clean baseline (10–30 mins = ~1000–3000 samples is plenty)
python mqtt_connection_with_storage.py nofaults

# 2. Collect test data with faults (for scoring your model)
python mqtt_connection_with_storage.py single_fault
python mqtt_connection_with_storage.py three_faults
python mqtt_connection_with_storage.py variable_setpoints
"""
