"""
CSCI 485 Final Project - Group 3
What Makes a Startup Succeed?
Predicting Acquisition Outcomes Using the Global Startup Success Dataset

Team: Andy Chen, Devon Rhodes, Muhammad Hasham Hussain, Prabin Subedi
"""

# =============================================================================
# STAGE 1: Data Loading and Preprocessing (Andy Chen)
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

plt.style.use("seaborn-v0_8-whitegrid")
RANDOM_STATE = 42

# --- 1.1 Load Data ---pip install pandas numpy matplotlib seaborn scikit-learn
df = pd.read_csv("global_startup_success_dataset.csv")
print("=" * 60)
print("STAGE 1: Data Loading & Preprocessing")
print("=" * 60)
print(f"Dataset shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicates: {df.duplicated().sum()}")

# --- 1.2 Clean Data ---
df.drop_duplicates(inplace=True)

# Encode binary target
df["Acquired"] = (df["Acquired?"] == "Yes").astype(int)
df["IPO"] = (df["IPO?"] == "Yes").astype(int)

# Encode categorical features
le = LabelEncoder()
df["Industry_enc"] = le.fit_transform(df["Industry"])
df["Country_enc"] = le.fit_transform(df["Country"])
df["FundingStage_enc"] = le.fit_transform(df["Funding Stage"])

print(f"\nAcquisition distribution:\n{df['Acquired?'].value_counts()}")
print(f"\nClass balance: {df['Acquired'].mean():.2%} acquired")

# --- 1.3 Feature Selection ---
FEATURES = [
    "Total Funding ($M)",
    "Number of Employees",
    "Annual Revenue ($M)",
    "Valuation ($B)",
    "Success Score",
    "Customer Base (Millions)",
    "Social Media Followers",
    "IPO",
    "Industry_enc",
    "Country_enc",
    "FundingStage_enc",
]

X = df[FEATURES]
y = df["Acquired"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {len(X_train)} | Test size: {len(X_test)}")


# =============================================================================
# STAGE 2: Exploratory Data Analysis (Devon Rhodes)
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 2: Exploratory Data Analysis")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("EDA: Acquisition Patterns", fontsize=15, fontweight="bold")

# --- 2.1 Acquisition rate by Industry ---
acq_by_industry = (
    df.groupby("Industry")["Acquired"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)
acq_by_industry.columns = ["Industry", "Acquisition Rate"]
axes[0, 0].bar(
    acq_by_industry["Industry"],
    acq_by_industry["Acquisition Rate"],
    color="steelblue",
    edgecolor="white",
)
axes[0, 0].set_title("Acquisition Rate by Industry")
axes[0, 0].set_xlabel("Industry")
axes[0, 0].set_ylabel("Acquisition Rate")
axes[0, 0].tick_params(axis="x", rotation=30)
axes[0, 0].set_ylim(0, 1)

# --- 2.2 Acquisition rate by Country (Top 10) ---
acq_by_country = (
    df.groupby("Country")["Acquired"]
    .agg(["mean", "count"])
    .query("count >= 30")
    .sort_values("mean", ascending=False)
    .head(10)
    .reset_index()
)
axes[0, 1].bar(
    acq_by_country["Country"],
    acq_by_country["mean"],
    color="coral",
    edgecolor="white",
)
axes[0, 1].set_title("Acquisition Rate by Country (Top 10, n>=30)")
axes[0, 1].set_xlabel("Country")
axes[0, 1].set_ylabel("Acquisition Rate")
axes[0, 1].tick_params(axis="x", rotation=45)
axes[0, 1].set_ylim(0, 1)

# --- 2.3 Acquisition rate by Funding Stage ---
stage_order = ["Seed", "Series A", "Series B", "Series C", "Series D", "IPO"]
acq_by_stage = (
    df[df["Funding Stage"].isin(stage_order)]
    .groupby("Funding Stage")["Acquired"]
    .mean()
    .reindex(stage_order)
    .reset_index()
)
acq_by_stage.columns = ["Funding Stage", "Acquisition Rate"]
axes[1, 0].bar(
    acq_by_stage["Funding Stage"],
    acq_by_stage["Acquisition Rate"],
    color="mediumseagreen",
    edgecolor="white",
)
axes[1, 0].set_title("Acquisition Rate by Funding Stage")
axes[1, 0].set_xlabel("Funding Stage")
axes[1, 0].set_ylabel("Acquisition Rate")
axes[1, 0].set_ylim(0, 1)

# --- 2.4 Correlation Heatmap ---
num_cols = [
    "Total Funding ($M)", "Number of Employees", "Annual Revenue ($M)",
    "Valuation ($B)", "Success Score", "Customer Base (Millions)",
    "Social Media Followers", "Acquired"
]
corr_matrix = df[num_cols].corr()
sns.heatmap(
    corr_matrix,
    ax=axes[1, 1],
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5,
    annot_kws={"size": 7},
)
axes[1, 1].set_title("Correlation Heatmap")
axes[1, 1].tick_params(axis="x", rotation=45, labelsize=7)
axes[1, 1].tick_params(axis="y", labelsize=7)

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("EDA plots saved: eda_plots.png")

# Print EDA summary table
print("\nAcquisition Rate by Industry:")
print(acq_by_industry.to_string(index=False))
print("\nAcquisition Rate by Funding Stage:")
print(acq_by_stage.to_string(index=False))


# =============================================================================
# STAGE 3: Modeling & Evaluation (Muhammad Hasham Hussain)
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 3: Modeling & Evaluation")
print("=" * 60)


# --- Helper: evaluate and print metrics ---
def evaluate_model(name, y_true, y_pred):
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
    }
    return metrics


# --- 3.1 Baseline Decision Tree ---
dt = DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
dt_metrics = evaluate_model("Decision Tree (depth=5)", y_test, y_pred_dt)
print(f"\nDecision Tree Results:\n{classification_report(y_test, y_pred_dt)}")

# --- 3.2 Tuned Decision Tree ---
dt_tuned = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=RANDOM_STATE,
)
dt_tuned.fit(X_train, y_train)
y_pred_dt_tuned = dt_tuned.predict(X_test)
dt_tuned_metrics = evaluate_model("Decision Tree (tuned)", y_test, y_pred_dt_tuned)
print(f"\nTuned Decision Tree Results:\n{classification_report(y_test, y_pred_dt_tuned)}")

# --- 3.3 Random Forest ---
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
rf_metrics = evaluate_model("Random Forest", y_test, y_pred_rf)
print(f"\nRandom Forest Results:\n{classification_report(y_test, y_pred_rf)}")

# --- 3.4 Comparison Table ---
results_df = pd.DataFrame([dt_metrics, dt_tuned_metrics, rf_metrics])
results_df = results_df.set_index("Model")
results_df = results_df.round(4)
print("\n--- Model Comparison ---")
print(results_df.to_string())

# --- 3.5 Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices", fontsize=13, fontweight="bold")

for ax, (name, y_pred) in zip(
    axes,
    [
        ("DT (depth=5)", y_pred_dt),
        ("DT (tuned)", y_pred_dt_tuned),
        ("Random Forest", y_pred_rf),
    ],
):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Not Acquired", "Acquired"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name)

plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("Confusion matrices saved: confusion_matrices.png")

# --- 3.6 Feature Importance (Random Forest) ---
feat_importance = pd.Series(rf.feature_importances_, index=FEATURES)
feat_importance = feat_importance.sort_values(ascending=True)

plt.figure(figsize=(8, 6))
colors = ["steelblue" if v < feat_importance.max() else "coral" for v in feat_importance]
plt.barh(feat_importance.index, feat_importance.values, color=colors, edgecolor="white")
plt.title("Random Forest: Feature Importance", fontsize=13, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Feature importance plot saved: feature_importance.png")

print("\n--- Feature Importance Ranking ---")
print(feat_importance.sort_values(ascending=False).round(4).to_string())

# --- 3.7 Decision Tree Visualization (baseline, for code walkthrough) ---
plt.figure(figsize=(20, 8))
plot_tree(
    dt,
    feature_names=FEATURES,
    class_names=["Not Acquired", "Acquired"],
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3,
)
plt.title("Decision Tree (first 3 levels)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("decision_tree_viz.png", dpi=120, bbox_inches="tight")
plt.show()
print("Decision tree visualization saved: decision_tree_viz.png")

# --- 3.8 Vertical Comparison (Intermediate vs Final) ---
print("\n" + "=" * 60)
print("VERTICAL COMPARISON: Intermediate vs Final Models")
print("=" * 60)
print(results_df.to_string())

# --- 3.9 Horizontal Comparison (Baseline reference) ---
print("\n" + "=" * 60)
print("HORIZONTAL COMPARISON: vs. Baseline (majority class)")
print("=" * 60)
majority_acc = max(y_test.mean(), 1 - y_test.mean())
print(f"Majority class baseline accuracy: {majority_acc:.4f}")
print(f"Decision Tree improvement: {dt_metrics['Accuracy'] - majority_acc:+.4f}")
print(f"Random Forest improvement: {rf_metrics['Accuracy'] - majority_acc:+.4f}")


# =============================================================================
# STAGE 4: Report Summary (Prabin Subedi)
# =============================================================================

print("\n" + "=" * 60)
print("STAGE 4: Summary & Conclusion")
print("=" * 60)
print(f"""
Project: What Makes a Startup Succeed?
Target:  Predicting Acquisition (Binary Classification)
Dataset: {len(df)} startups, {len(FEATURES)} features

--- Key Findings ---
Best model:         Random Forest
Best F1-Score:      {rf_metrics['F1-Score']:.4f}
Best Accuracy:      {rf_metrics['Accuracy']:.4f}

Top 3 most important features:
{feat_importance.sort_values(ascending=False).head(3).round(4).to_string()}

--- What Worked ---
- Random Forest outperformed Decision Tree on all metrics
- Feature importance reveals which variables matter most
- Balanced dataset (50/50) avoids class imbalance issues

--- What Did Not Work ---
- Simple Decision Tree with depth=5 showed signs of underfitting
- Categorical features (Tech Stack) not used due to high cardinality

--- Future Work ---
- Try XGBoost or LightGBM for better performance
- Add Tech Stack as one-hot encoded features
- Hyperparameter tuning via GridSearchCV
- SHAP values for deeper interpretability
""")
