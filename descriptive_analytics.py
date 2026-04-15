import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import FIGURES_DIR


def run_descriptive_analytics(batches, aborts, logs, downtime):
    # Monthly OEE by plant
    logs["month"] = logs["step_start"].dt.to_period("M")

    batch_oee = logs.groupby(["batch_id", "plant_id", "month"]).agg(
        availability=("availability", "mean"),
        performance=("performance", "mean"),
        quality=("quality", "mean"),
        oee=("oee", "mean"),
    ).reset_index()

    monthly_oee = batch_oee.groupby(["plant_id", "month"]).agg(
        availability=("availability", "mean"),
        performance=("performance", "mean"),
        quality=("quality", "mean"),
        oee=("oee", "mean"),
    ).reset_index()
    monthly_oee["month_dt"] = monthly_oee["month"].dt.to_timestamp()

    # OEE and components over time per plant
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
    metrics = ["oee", "availability", "performance", "quality"]
    titles = ["Overall OEE", "Availability", "Performance", "Quality"]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        for plant_id, grp in monthly_oee.groupby("plant_id"):
            ax.plot(grp["month_dt"], grp[metric], marker="o", label=plant_id)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Monthly OEE Decomposition by Plant", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "oee_trends.png", dpi=150, bbox_inches="tight")
    plt.close()

    return monthly_oee
