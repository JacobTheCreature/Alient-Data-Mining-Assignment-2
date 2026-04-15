import warnings
warnings.filterwarnings("ignore")
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
from pathlib import Path
from data_loader import load_data, FIGURES_DIR
from descriptive_analytics import run_descriptive_analytics
from pattern_mining import run_pattern_mining
from cross_plant_correlation import run_cross_plant_correlation
from predictive_maintenance import run_predictive_maintenance


def main():
    FIGURES_DIR.mkdir(exist_ok=True)

    batches, aborts, logs, downtime, incidents, equipment, plants = load_data()

    monthly_oee = run_descriptive_analytics(batches, aborts, logs, downtime)

    run_pattern_mining(incidents, downtime, batches, aborts, logs, equipment)

    run_cross_plant_correlation(monthly_oee)

    run_predictive_maintenance(batches, downtime, incidents, logs, monthly_oee)

if __name__ == "__main__":
    main()
