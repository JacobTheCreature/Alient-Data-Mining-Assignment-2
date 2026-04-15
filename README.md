# Pharmaceutical Manufacturing Alien Sabotage Investigation

A data mining pipeline that analyzes operational data from three pharmaceutical manufacturing plants to detect, characterize, and predict an extraterrestrial sabotage campaign targeting equipment performance.

## Overview

Three geographically dispersed plants (Minneapolis, Austin, Newark) are experiencing synchronized equipment degradation that cannot be explained by independent failures. This project applies four data mining techniques to the operational data to prove the attack is externally coordinated and build an early warning system:

1. **Descriptive Analytics** — OEE decomposition and downtime/abort heatmaps across plants
2. **Pattern Mining** — Apriori association rules and decision trees to link incidents to downtime and aborts
3. **Cross-Plant Correlation** — Pearson correlation, Dynamic Time Warping, and PCA to prove lockstep degradation
4. **Predictive Maintenance** — Random Forest and Gradient Boosted Trees (75/25 train/test split) to predict downtime

## Setup

### Prerequisites

- Python 3.8+

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn mlxtend
```

### Run

```bash
python main.py
```
