# data_collector.py
# Collects data from MQTT streams and saves to CSV
# Can collect from multiple streams sequentially or in parallel

# python data_collector.py --stream single_fault --infinite

import paho.mqtt.client as mqtt
import json
import pandas as pd
from datetime import datetime
import os
import sys
import time
import argparse
from threading import Thread, Lock

# ================================
# CONFIGURATION
# ================================
BROKER_HOST = "engf0001.cs.ucl.ac.uk"
BROKER_PORT = 1883
CONNECTION_TIMEOUT = 60

STREAM_TYPES = {
    "nofaults": "bioreactor_sim/nofaults/telemetry/summary",
    "single_fault": "bioreactor_sim/single_fault/telemetry/summary",
    "three_faults": "bioreactor_sim/three_faults/telemetry/summary",
    "variable_setpoints": "bioreactor_sim/variable_setpoints/telemetry/summary",
}

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# ================================
# DATA COLLECTOR CLASS
# ================================


class DataCollector:
    """Collects data from a single MQTT stream."""

    def __init__(
        self, stream_name, duration_seconds=None, target_samples=None, infinite=False
    ):
        self.stream_name = stream_name
        self.duration_seconds = duration_seconds
        self.target_samples = target_samples
        self.infinite = infinite
        self.topic = STREAM_TYPES[stream_name]

        # Data storage
        self.data_buffer = []
        self.csv_filename = None
        self.sample_count = 0
        self.start_time = None

        # MQTT client
        self.client = None
        self.connected = False
        self.lock = Lock()

    def flatten_payload(self, payload):
        """Extract and flatten MQTT payload."""
        try:
            flat = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "temp_mean": payload["temperature_C"]["mean"],
                "temp_min": payload["temperature_C"]["min"],
                "temp_max": payload["temperature_C"]["max"],
                "ph_mean": payload["pH"]["mean"],
                "ph_min": payload["pH"]["min"],
                "ph_max": payload["pH"]["max"],
                "rpm_mean": payload["rpm"]["mean"],
                "rpm_min": payload["rpm"]["min"],
                "rpm_max": payload["rpm"]["max"],
                "heater_pwm": payload["actuators_avg"]["heater_pwm"],
                "motor_pwm": payload["actuators_avg"]["motor_pwm"],
                "acid_pwm": payload["actuators_avg"]["acid_pwm"],
                "base_pwm": payload["actuators_avg"]["base_pwm"],
                "acid_dose_l": payload["dosing_l"]["acid"],
                "base_dose_l": payload["dosing_l"]["base"],
                "setpoint_temp": payload["setpoints"]["temperature_C"],
                "setpoint_ph": payload["setpoints"]["pH"],
                "setpoint_rpm": payload["setpoints"]["rpm"],
                "faults_active": (
                    ", ".join(
                        [
                            f.get("type", str(f)) if isinstance(f, dict) else str(f)
                            for f in payload["faults"]["last_active"]
                        ]
                    )
                    if payload["faults"]["last_active"]
                    else "none"
                ),
                "has_fault": int(len(payload["faults"]["last_active"]) > 0),
            }
            return flat
        except KeyError as e:
            print(f"[{self.stream_name}] Warning: Missing key {e}")
            return None

    def on_connect(self, client, userdata, flags, rc):
        """MQTT connection callback."""
        if rc == 0:
            self.connected = True
            client.subscribe(self.topic)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.csv_filename = os.path.join(
                DATA_DIR, f"data_{self.stream_name}_{timestamp}.csv"
            )
            self.start_time = time.time()

            print(f"\n{'='*70}")
            print(f"[{self.stream_name}] Connected!")
            print(f"  Topic: {self.topic}")
            print(f"  Saving to: {self.csv_filename}")
            if self.infinite:
                print(f"  Mode: INFINITE (press Ctrl+C to stop)")
            elif self.target_samples:
                print(f"  Target: {self.target_samples} samples")
            elif self.duration_seconds:
                print(
                    f"  Duration: {self.duration_seconds}s ({self.duration_seconds/60:.1f} min)"
                )
            print(f"{'='*70}")
        else:
            print(f"[{self.stream_name}] Connection failed: {rc}")

    def on_message(self, client, userdata, msg):
        """Process incoming MQTT message."""
        try:
            payload = json.loads(msg.payload.decode())
            flat = self.flatten_payload(payload)

            if flat is None:
                return

            with self.lock:
                self.data_buffer.append(flat)
                self.sample_count += 1

                # Periodic save
                if len(self.data_buffer) >= 100:
                    self.save_buffer()

                # Progress update
                elapsed = time.time() - self.start_time
                if self.sample_count % 50 == 0:
                    fault_count = sum(
                        1 for d in self.data_buffer if d["has_fault"] == 1
                    )
                    elapsed_min = elapsed / 60
                    print(
                        f"[{self.stream_name}] Samples: {self.sample_count:4d} | "
                        f"Elapsed: {elapsed_min:6.1f}m | Faults seen: {fault_count}"
                    )

                # Check if we should stop (only if not infinite mode)
                if not self.infinite:
                    if self.target_samples and self.sample_count >= self.target_samples:
                        print(
                            f"\n[{self.stream_name}] ✓ Reached target of {self.target_samples} samples!"
                        )
                        client.disconnect()
                    elif self.duration_seconds and elapsed >= self.duration_seconds:
                        print(
                            f"\n[{self.stream_name}] ✓ Reached duration of {self.duration_seconds}s!"
                        )
                        client.disconnect()

        except Exception as e:
            print(f"[{self.stream_name}] Error: {e}")
            import traceback

            traceback.print_exc()

    def save_buffer(self):
        """Save buffered data to CSV."""
        if len(self.data_buffer) == 0:
            return

        df = pd.DataFrame(self.data_buffer)
        header = not os.path.exists(self.csv_filename)
        df.to_csv(self.csv_filename, mode="a", header=header, index=False)
        self.data_buffer = []

    def on_disconnect(self, client, userdata, rc):
        """MQTT disconnect callback."""
        with self.lock:
            self.save_buffer()

        print(f"\n{'='*70}")
        print(f"[{self.stream_name}] COLLECTION COMPLETE")
        print(f"{'='*70}")
        print(f"Total samples collected: {self.sample_count}")
        if self.start_time:
            elapsed = time.time() - self.start_time
            print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Saved to: {self.csv_filename}")

        # Load and show summary
        if os.path.exists(self.csv_filename):
            df = pd.read_csv(self.csv_filename)
            fault_count = len(df[df["has_fault"] == 1])
            normal_count = len(df[df["has_fault"] == 0])
            print(f"  Normal samples: {normal_count}")
            print(f"  Fault samples:  {fault_count}")
            if fault_count > 0:
                print(
                    f"  Fault types: {df[df['has_fault']==1]['faults_active'].unique()}"
                )
        print(f"{'='*70}\n")

    def collect(self):
        """Start collecting data."""
        self.client = mqtt.Client(userdata={"stream": self.stream_name})
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        self.client.on_disconnect = self.on_disconnect

        try:
            print(f"[{self.stream_name}] Connecting to {BROKER_HOST}:{BROKER_PORT}...")
            self.client.connect(BROKER_HOST, BROKER_PORT, CONNECTION_TIMEOUT)
            self.client.loop_forever()
        except KeyboardInterrupt:
            print(f"\n[{self.stream_name}] Stopped by user")
            self.client.disconnect()
        except Exception as e:
            print(f"[{self.stream_name}] Error: {e}")
            import traceback

            traceback.print_exc()


# ================================
# COLLECTION MODES
# ================================


def collect_single_stream(stream_name, duration=None, samples=None, infinite=False):
    """Collect data from a single stream."""
    if stream_name not in STREAM_TYPES:
        print(f"ERROR: Invalid stream '{stream_name}'")
        print(f"Available streams: {list(STREAM_TYPES.keys())}")
        return

    collector = DataCollector(stream_name, duration, samples, infinite)
    collector.collect()


def collect_all_streams_sequential(
    duration_per_stream=None, samples_per_stream=None, infinite=False
):
    """Collect from all streams one after another."""
    print("\n" + "=" * 70)
    print("SEQUENTIAL COLLECTION FROM ALL STREAMS")
    print("=" * 70)

    for stream_name in STREAM_TYPES.keys():
        print(f"\n>>> Starting collection from: {stream_name}")
        collector = DataCollector(
            stream_name, duration_per_stream, samples_per_stream, infinite
        )
        collector.collect()
        print(f">>> Finished: {stream_name}\n")
        time.sleep(2)  # Brief pause between streams

    print("\n" + "=" * 70)
    print("✓ ALL STREAMS COLLECTED!")
    print("=" * 70)


def collect_all_streams_parallel(duration=None, samples=None, infinite=False):
    """Collect from all streams simultaneously using threads."""
    print("\n" + "=" * 70)
    print("PARALLEL COLLECTION FROM ALL STREAMS")
    print("=" * 70)
    if infinite:
        print("Mode: INFINITE (press Ctrl+C to stop all)")
    elif duration:
        print(f"Duration: {duration}s ({duration/60:.1f} min) per stream")
    if samples:
        print(f"Target: {samples} samples per stream")
    print("=" * 70)

    collectors = []
    threads = []

    # Create collectors
    for stream_name in STREAM_TYPES.keys():
        collector = DataCollector(stream_name, duration, samples, infinite)
        collectors.append(collector)

    # Start collection threads
    for collector in collectors:
        thread = Thread(target=collector.collect)
        thread.daemon = True
        threads.append(thread)
        thread.start()
        time.sleep(0.5)  # Stagger starts slightly

    # Wait for all to complete
    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("\n\nStopping all collections...")
        for collector in collectors:
            if collector.client:
                collector.client.disconnect()

    print("\n" + "=" * 70)
    print("✓ ALL PARALLEL COLLECTIONS COMPLETE!")
    print("=" * 70)


# ================================
# MAIN
# ================================


def main():
    parser = argparse.ArgumentParser(
        description="Collect bioreactor data from MQTT streams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect from single stream indefinitely
  python data_collector.py --stream nofaults --infinite
  
  # Collect from single stream for 5 minutes
  python data_collector.py --stream nofaults --duration 300
  
  # Collect 1000 samples from single_fault stream
  python data_collector.py --stream single_fault --samples 1000
  
  # Collect from all streams indefinitely (in parallel)
  python data_collector.py --all-parallel --infinite
  
  # Collect from all streams sequentially (5 min each)
  python data_collector.py --all-sequential --duration 300
  
  # Collect from all streams in parallel (10 min each)
  python data_collector.py --all-parallel --duration 600
  
  # Quick collection (2000 samples from each, parallel)
  python data_collector.py --all-parallel --samples 2000
        """,
    )

    parser.add_argument(
        "--stream",
        type=str,
        choices=list(STREAM_TYPES.keys()),
        help="Single stream to collect from",
    )
    parser.add_argument(
        "--all-sequential",
        action="store_true",
        help="Collect from all streams one after another",
    )
    parser.add_argument(
        "--all-parallel",
        action="store_true",
        help="Collect from all streams simultaneously",
    )
    parser.add_argument(
        "--infinite",
        action="store_true",
        help="Run indefinitely until stopped with Ctrl+C",
    )
    parser.add_argument(
        "--duration",
        type=int,
        help="Duration in seconds (ignored if --infinite is used)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        help="Target number of samples (ignored if --infinite is used)",
    )

    args = parser.parse_args()

    # Show header
    print("\n" + "=" * 70)
    print("BIOREACTOR DATA COLLECTOR")
    print("=" * 70)
    print(f"Broker: {BROKER_HOST}:{BROKER_PORT}")
    print(f"Data directory: {DATA_DIR}")
    print("=" * 70)

    # Validate arguments
    mode_count = sum([bool(args.stream), args.all_sequential, args.all_parallel])
    if mode_count == 0:
        print("\nERROR: Must specify collection mode!")
        print("Use --stream, --all-sequential, or --all-parallel")
        print("\nFor help: python data_collector.py --help")
        return
    elif mode_count > 1:
        print("\nERROR: Can only use one collection mode at a time!")
        return

    # Warn if duration/samples specified with infinite
    if args.infinite and (args.duration or args.samples):
        print("\nWARNING: --duration and --samples are ignored when using --infinite")
        args.duration = None
        args.samples = None

    # Set default duration if not infinite and no duration/samples specified
    if not args.infinite and not args.duration and not args.samples:
        args.duration = 300
        print(f"\nUsing default duration: {args.duration}s (5 minutes)")

    # Execute collection
    try:
        if args.stream:
            collect_single_stream(
                args.stream, args.duration, args.samples, args.infinite
            )
        elif args.all_sequential:
            collect_all_streams_sequential(args.duration, args.samples, args.infinite)
        elif args.all_parallel:
            collect_all_streams_parallel(args.duration, args.samples, args.infinite)
    except KeyboardInterrupt:
        print("\n\nCollection interrupted by user")

    print("\n✓ Done!\n")


if __name__ == "__main__":
    main()
