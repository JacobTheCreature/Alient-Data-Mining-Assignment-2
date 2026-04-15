import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from data_loader import FIGURES_DIR


def run_predictive_maintenance(batches, downtime, incidents, logs, monthly_oee):
    # Feature engineering per batch: predict if downtime occurs

    batch_df = batches[["batch_id", "plant_id", "batch_start"]].copy()
    batch_df["week"] = batch_df["batch_start"].dt.isocalendar().week.astype(int)

    # Target: did this batch have a downtime event?
    dt_batches = set(downtime["batch_id"].unique())
    batch_df["has_downtime"] = batch_df["batch_id"].isin(dt_batches).astype(int)

    # Incident features
    inc_feats = incidents.groupby("batch_id").agg(inc_count=("incident_id", "count")).reset_index()
    severity_map = {"Low": 1, "Medium": 2, "High": 3}
    incidents["sev_num"] = incidents["severity"].map(severity_map)
    inc_sev = incidents.groupby("batch_id")["sev_num"].max().reset_index(name="max_sev")

    # Incident type one-hot
    inc_type_counts = incidents.groupby(["batch_id", "reason"]).size().unstack(fill_value=0)
    inc_type_counts.columns = [f"inc_{c.replace(' ', '_')}" for c in inc_type_counts.columns]
    inc_type_counts = inc_type_counts.reset_index()

    # OEE features from equipment log
    oee_feats = logs.groupby("batch_id").agg(
        mean_availability=("availability", "mean"),
        mean_performance=("performance", "mean"),
        mean_quality=("quality", "mean"),
        mean_oee=("oee", "mean"),
    ).reset_index()

    # Merge
    feat_df = batch_df.merge(inc_feats, on="batch_id", how="left")
    feat_df = feat_df.merge(inc_sev, on="batch_id", how="left")
    feat_df = feat_df.merge(inc_type_counts, on="batch_id", how="left")
    feat_df = feat_df.merge(oee_feats, on="batch_id", how="left")
    feat_df = feat_df.fillna(0)

    le = LabelEncoder()
    feat_df["plant_enc"] = le.fit_transform(feat_df["plant_id"])

    drop_cols = ["batch_id", "plant_id", "batch_start", "has_downtime"]
    feature_cols = [c for c in feat_df.columns if c not in drop_cols]
    
    # Exclude specified features from training
    features_to_remove = ['mean_availability', 'mean_performance', 'mean_oee', 'mean_quality', 'week', 'max_sev', 'plant_enc', 'inc_count']
    feature_cols = [c for c in feature_cols if c not in features_to_remove]

    X = feat_df[feature_cols]
    y = feat_df["has_downtime"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    auc_rf = roc_auc_score(y_test, y_proba_rf)

    # Dedicated Random Forest Results Figure
    rf_importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    cm = confusion_matrix(y_test, y_pred_rf)
    fpr, tpr, _ = roc_curve(y_test, y_proba_rf)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Random Forest Analysis: Downtime Prediction Results", fontsize=16, fontweight="bold")

    # Top 15 Feature Importances
    top_imp = rf_importances.head(15).sort_values()
    axes[0, 0].barh(top_imp.index, top_imp.values, color="steelblue", edgecolor="navy", linewidth=0.5)
    axes[0, 0].set_title("Top 15 Feature Importances", fontsize=13, fontweight="bold")
    axes[0, 0].set_xlabel("Importance")
    for i, (val, name) in enumerate(zip(top_imp.values, top_imp.index)):
        axes[0, 0].text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)

    # Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0, 1],
                xticklabels=["No Downtime", "Downtime"],
                yticklabels=["No Downtime", "Downtime"],
                linewidths=1, linecolor="white")
    axes[0, 1].set_title("Confusion Matrix", fontsize=13, fontweight="bold")
    axes[0, 1].set_ylabel("Actual")
    axes[0, 1].set_xlabel("Predicted")
    total = cm.sum()
    acc = (cm[0, 0] + cm[1, 1]) / total
    axes[0, 1].text(0.5, -0.12, f"Accuracy: {acc:.1%}  |  Samples: {total}", ha="center", transform=axes[0, 1].transAxes, fontsize=11)

    # ROC Curve
    axes[1, 0].plot(fpr, tpr, color="steelblue", linewidth=2.5, label=f"Random Forest (AUC = {auc_rf:.3f})")
    axes[1, 0].plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Random Chance")
    axes[1, 0].fill_between(fpr, tpr, alpha=0.15, color="steelblue")
    axes[1, 0].set_title("ROC Curve", fontsize=13, fontweight="bold")
    axes[1, 0].set_xlabel("False Positive Rate")
    axes[1, 0].set_ylabel("True Positive Rate")
    axes[1, 0].legend(loc="lower right", fontsize=11)
    axes[1, 0].set_xlim([-0.02, 1.02])
    axes[1, 0].set_ylim([-0.02, 1.02])
    axes[1, 0].grid(True, alpha=0.3)

    # Predicted Probability Distribution
    axes[1, 1].hist(y_proba_rf[y_test == 0], bins=30, alpha=0.6, color="steelblue", label="No Downtime", edgecolor="white", linewidth=0.5)
    axes[1, 1].hist(y_proba_rf[y_test == 1], bins=30, alpha=0.6, color="crimson", label="Downtime", edgecolor="white", linewidth=0.5)
    axes[1, 1].axvline(0.5, color="black", linestyle="--", linewidth=1, label="Decision Threshold (0.5)")
    axes[1, 1].set_title("Predicted Probability Distribution", fontsize=13, fontweight="bold")
    axes[1, 1].set_xlabel("Predicted Probability of Downtime")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(FIGURES_DIR / "random_forest_results.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Gradient Boosted Trees
    gbt = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, min_samples_leaf=10, random_state=42)
    gbt.fit(X_train, y_train)
    y_pred_gbt = gbt.predict(X_test)
    y_proba_gbt = gbt.predict_proba(X_test)[:, 1]

    auc_gbt = roc_auc_score(y_test, y_proba_gbt)

    # Feature importance
    best_model = rf if auc_rf >= auc_gbt else gbt
    best_name = "Random Forest" if auc_rf >= auc_gbt else "Gradient Boosted Trees"
    importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Plot feature importances
    _, ax = plt.subplots(figsize=(10, 8))
    importances.head(15).plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"{best_name} Feature Importances (Downtime Prediction)")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "downtime_feature_importances.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Cross-Plant Alerting Logic

    pivot = monthly_oee.pivot(index="month", columns="plant_id", values="oee")
    plants = pivot.columns.tolist()

    # Rolling correlation (window=3 months)
    window = 3
    rolling_corrs = []
    months_list = pivot.index.tolist()

    for end_idx in range(window, len(months_list) + 1):
        window_data = pivot.iloc[end_idx - window:end_idx]
        corr_mat = window_data.corr()
        pair_corrs = []
        for i in range(len(plants)):
            for j in range(i + 1, len(plants)):
                pair_corrs.append(corr_mat.iloc[i, j])
        avg_corr = np.mean(pair_corrs)
        rolling_corrs.append({
            "month": months_list[end_idx - 1],
            "avg_pairwise_corr": avg_corr,
            "alert": avg_corr > 0.90
        })

    alert_df = pd.DataFrame(rolling_corrs)
    alert_df["month_dt"] = alert_df["month"].dt.to_timestamp()

    return best_model, importances, alert_df
