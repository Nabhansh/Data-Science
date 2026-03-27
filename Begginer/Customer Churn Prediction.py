"""
Customer Churn Prediction
==========================
Predicts whether a telecom customer will churn using:
- Feature engineering
- Logistic Regression & Random Forest
- Evaluation: accuracy, confusion matrix, ROC-AUC, classification report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
)

np.random.seed(42)

# ── Synthetic telecom dataset ──────────────────────────────────────────────────
def generate_churn_data(n=1000):
    tenure          = np.random.randint(1, 73, n)
    monthly_charges = np.random.uniform(20, 120, n)
    total_charges   = tenure * monthly_charges + np.random.normal(0, 50, n)
    num_services    = np.random.randint(1, 9, n)
    contract        = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    internet        = np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment         = np.random.choice(["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n)

    # Churn probability
    p = (
        0.40 * (contract == "Month-to-month")
        + 0.10 * (internet == "Fiber optic")
        + 0.004 * monthly_charges
        - 0.005 * tenure
        - 0.02 * num_services
    ).clip(0.05, 0.90)
    churn = np.random.binomial(1, p)

    return pd.DataFrame({
        "Tenure": tenure,
        "MonthlyCharges": np.round(monthly_charges, 2),
        "TotalCharges": np.round(total_charges.clip(0), 2),
        "NumServices": num_services,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "Churn": churn,
    })


print("=" * 60)
print("  CUSTOMER CHURN PREDICTION")
print("=" * 60)

df = generate_churn_data()
print(f"\nDataset shape : {df.shape}")
print(f"Churn rate    : {df['Churn'].mean():.1%}\n")

# ── Preprocessing ─────────────────────────────────────────────────────────────
le = LabelEncoder()
for col in ["Contract", "InternetService", "PaymentMethod"]:
    df[col + "_enc"] = le.fit_transform(df[col])

feature_cols = ["Tenure", "MonthlyCharges", "TotalCharges", "NumServices",
                "Contract_enc", "InternetService_enc", "PaymentMethod_enc"]

X = df[feature_cols]
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── Models ────────────────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    X_tr = X_train_sc if name == "Logistic Regression" else X_train
    X_te = X_test_sc  if name == "Logistic Regression" else X_test
    model.fit(X_tr, y_train)
    preds = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]
    results[name] = {"model": model, "preds": preds, "proba": proba,
                     "acc": accuracy_score(y_test, preds),
                     "auc": roc_auc_score(y_test, proba)}
    print(f"\n── {name} ──")
    print(f"  Accuracy : {results[name]['acc']:.4f}")
    print(f"  ROC-AUC  : {results[name]['auc']:.4f}")
    print(classification_report(y_test, preds, target_names=["No Churn", "Churn"]))

# ── Feature importance (RF) ───────────────────────────────────────────────────
rf = results["Random Forest"]["model"]
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nFeature Importances (Random Forest):")
print(importances.round(4))

# ── Visualisations ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Customer Churn Prediction", fontsize=15, fontweight="bold")

# Confusion matrix
cm = confusion_matrix(y_test, results["Random Forest"]["preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", ax=axes[0],
            xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
axes[0].set_title("Confusion Matrix (RF)")
axes[0].set_ylabel("Actual"); axes[0].set_xlabel("Predicted")

# ROC curves
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res["proba"])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC={res['auc']:.3f})")
axes[1].plot([0, 1], [0, 1], "k--")
axes[1].set_title("ROC Curve"); axes[1].set_xlabel("FPR"); axes[1].set_ylabel("TPR")
axes[1].legend()

# Feature importance
importances.plot(kind="barh", ax=axes[2], color="#e74c3c")
axes[2].set_title("Feature Importance (RF)")
axes[2].invert_yaxis()

plt.tight_layout()
plt.savefig("churn_prediction_results.png", dpi=150, bbox_inches="tight")
print("\nPlots saved → churn_prediction_results.png")
print("\nChurn prediction complete!")