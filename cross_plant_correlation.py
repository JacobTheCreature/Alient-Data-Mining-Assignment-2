import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from data_loader import FIGURES_DIR


def run_cross_plant_correlation(monthly_oee):
    # Pairwise Pearson correlation

    pivot = monthly_oee.pivot(index="month", columns="plant_id", values="oee")
    plants = pivot.columns.tolist()

    corr_matrix = pivot.corr()

    def dtw_distance(s, t):
        n, m = len(s), len(t)
        dtw_mat = np.full((n + 1, m + 1), np.inf)
        dtw_mat[0, 0] = 0.0
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(s[i-1] - t[j-1])
                dtw_mat[i, j] = cost + min(dtw_mat[i-1, j], dtw_mat[i, j-1], dtw_mat[i-1, j-1])
        return dtw_mat[n, m]

    dtw_results = {}
    for i in range(len(plants)):
        for j in range(i + 1, len(plants)):
            s1 = pivot[plants[i]].dropna().values
            s2 = pivot[plants[j]].dropna().values
            dist = dtw_distance(s1, s2)
            dtw_results[(plants[i], plants[j])] = dist

    oee_matrix = pivot.dropna().values  # rows=months, cols=plants
    pca = PCA()
    pca.fit(oee_matrix)

    pc1_var = pca.explained_variance_ratio_[0]

    # Plot PCA + overlay
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for plant in plants:
        vals = pivot[plant].dropna()
        axes[0].plot(vals.index.to_timestamp(), vals.values, marker="o", label=plant)
    axes[0].set_title("Monthly OEE by Plant (Overlaid)")
    axes[0].set_ylabel("OEE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis="x", rotation=45)

    axes[1].bar(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_, color="steelblue", alpha=0.8)
    axes[1].set_xlabel("Principal Component")
    axes[1].set_ylabel("Explained Variance Ratio")
    axes[1].set_title(f"PCA -- PC1 explains {pc1_var*100:.1f}% of variance")
    axes[1].set_xticks(range(1, len(pca.explained_variance_ratio_) + 1))
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Cross-Plant Correlation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "cross_plant_pca.png", dpi=150, bbox_inches="tight")
    plt.close()

    return corr_matrix, dtw_results, pca
