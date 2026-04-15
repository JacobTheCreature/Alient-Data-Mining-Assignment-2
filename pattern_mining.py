import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from data_loader import FIGURES_DIR


def run_pattern_mining(incidents, downtime, batches, aborts, logs, equipment):
    # Apriori Association Rule Mining on incidents -> downtime

    # Create a mapping: for each batch, which incident reasons occurred
    inc_by_batch = incidents.groupby("batch_id")["reason"].apply(set).to_dict()
    dt_by_batch = downtime.groupby("batch_id")["reason"].apply(set).to_dict()

    # Get ordered batch list per plant
    batches_ordered = batches.sort_values("batch_start").groupby("plant_id")["batch_id"].apply(list).to_dict()

    # Build transactions: incident reason -> downtime reason within next 2 batches
    transactions = []
    for plant_id, batch_list in batches_ordered.items():
        for i, batch_id in enumerate(batch_list):
            if batch_id not in inc_by_batch:
                continue
            # Look ahead 2 batches for downtime
            future_dt_reasons = set()
            for j in range(i, min(i + 3, len(batch_list))):
                future_batch = batch_list[j]
                if future_batch in dt_by_batch:
                    future_dt_reasons.update(dt_by_batch[future_batch])

            if future_dt_reasons:
                for inc_reason in inc_by_batch[batch_id]:
                    for dt_reason in future_dt_reasons:
                        transactions.append({f"INC:{inc_reason}": True, f"DT:{dt_reason}": True})

    # Create one-hot encoded transaction dataframe
    all_items = set()
    for t in transactions:
        all_items.update(t.keys())

    txn_matrix = pd.DataFrame([{item: item in t for item in all_items} for t in transactions])

    # Run Apriori
    freq_items = apriori(txn_matrix, min_support=0.02, use_colnames=True)
    if len(freq_items) > 0:
        rules = association_rules(freq_items, metric="confidence", min_threshold=0.15)
        rules = rules.sort_values("confidence", ascending=False)
    else:
        rules = pd.DataFrame()

    # Decision Tree to predict batch aborts

    # Build feature matrix per batch
    batches_feat = batches[["batch_id", "plant_id", "batch_start"]].copy()
    batches_feat["is_aborted"] = batches_feat["batch_id"].isin(aborts["batch_id"]).astype(int)

    # Add incident counts and severity per batch
    inc_counts = incidents.groupby("batch_id").agg(incident_count=("incident_id", "count")).reset_index()

    severity_map = {"Low": 1, "Medium": 2, "High": 3}
    incidents["severity_num"] = incidents["severity"].map(severity_map)
    inc_severity = incidents.groupby("batch_id")["severity_num"].max().reset_index(name="max_severity")

    # One-hot encode incident reasons
    inc_reason_counts = incidents.groupby(["batch_id", "reason"]).size().unstack(fill_value=0)
    inc_reason_counts.columns = [f"inc_{col.replace(' ', '_').replace('-', '_').lower()}" for col in inc_reason_counts.columns]
    inc_reason_counts = inc_reason_counts.reset_index()

    # Downtime features per batch
    dt_feats = downtime.groupby("batch_id").agg(dt_count=("downtime_id", "count"), dt_total_duration=("duration_minutes", "sum")).reset_index()

    # One-hot encode downtime reasons
    dt_reason_counts = downtime.groupby(["batch_id", "reason"]).size().unstack(fill_value=0)
    dt_reason_counts.columns = [f"dt_{col.replace(' ', '_').lower()}" for col in dt_reason_counts.columns]
    dt_reason_counts = dt_reason_counts.reset_index()

    # Merge all features (exclude OEE features -- they leak the abort outcome)
    feat_df = batches_feat.merge(inc_counts, on="batch_id", how="left")
    feat_df = feat_df.merge(inc_severity, on="batch_id", how="left")
    feat_df = feat_df.merge(inc_reason_counts, on="batch_id", how="left")
    feat_df = feat_df.merge(dt_feats, on="batch_id", how="left")
    feat_df = feat_df.merge(dt_reason_counts, on="batch_id", how="left")
    feat_df = feat_df.fillna(0)

    # Encode plant_id
    le_plant = LabelEncoder()
    feat_df["plant_encoded"] = le_plant.fit_transform(feat_df["plant_id"])

    # Build feature list dynamically (exclude ID, target, and aggregate columns)
    exclude_cols = ["batch_id", "plant_id", "batch_start", "is_aborted", "dt_total_duration", "incident_count", "max_severity", "dt_count", "plant_encoded", "dt_scheduled_maintenance_overrun"]
    feature_cols = [c for c in feat_df.columns if c not in exclude_cols]

    X = feat_df[feature_cols]
    y = feat_df["is_aborted"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    dt_clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=42)
    dt_clf.fit(X_train, y_train)
    y_pred = dt_clf.predict(X_test)

    # Feature importance
    importances = pd.Series(dt_clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    # Decision tree text
    export_text(dt_clf, feature_names=feature_cols, max_depth=3)

    # Plot feature importances
    _, ax = plt.subplots(figsize=(10, 6))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title("Decision Tree Feature Importances for Abort Prediction")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "abort_feature_importances.png", dpi=150, bbox_inches="tight")
    plt.close()

    return rules, dt_clf, importances
