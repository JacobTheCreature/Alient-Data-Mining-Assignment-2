import pandas as pd
from pathlib import Path

DATA_DIR = Path("Datasets")
FIGURES_DIR = Path("figures")


def load_data():
    batches = pd.read_csv(DATA_DIR / "batches.csv", parse_dates=["batch_start", "batch_end"])
    aborts = pd.read_csv(DATA_DIR / "batch_aborts.csv", parse_dates=["timestamp"])
    logs = pd.read_csv(DATA_DIR / "batch_equipment_log.csv", parse_dates=["step_start", "step_end"])
    downtime = pd.read_csv(DATA_DIR / "downtime_events.csv", parse_dates=["start_timestamp"])
    incidents = pd.read_csv(DATA_DIR / "incidents.csv", parse_dates=["timestamp"])
    equipment = pd.read_csv(DATA_DIR / "equipment.csv")
    plants = pd.read_csv(DATA_DIR / "plants.csv")
    return batches, aborts, logs, downtime, incidents, equipment, plants
