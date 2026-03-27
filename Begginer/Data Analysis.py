"""
Exploratory Data Analysis (EDA) Project
=========================================
Performs comprehensive EDA on the Titanic dataset including:
- Data loading & overview
- Missing value analysis
- Statistical summaries
- Distribution plots
- Correlation heatmap
- Survival analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# ── Reproducibility ──────────────────────────────────────────────────────────
np.random.seed(42)

# ── Synthetic Titanic-like dataset ────────────────────────────────────────────
def generate_titanic_data(n=891):
    ages   = np.random.normal(30, 14, n).clip(1, 80)
    fares  = np.random.exponential(32, n).clip(5, 512)
    pclass = np.random.choice([1, 2, 3], n, p=[0.24, 0.21, 0.55])
    sex    = np.random.choice(["male", "female"], n, p=[0.65, 0.35])
    sibsp  = np.random.choice(range(9), n, p=[0.68,0.23,0.03,0.02,0.01,0.01,0.01,0.005,0.005])
    parch  = np.random.choice(range(7), n, p=[0.76,0.13,0.05,0.03,0.01,0.01,0.01])

    # Survival probability influenced by pclass, sex, age
    p_surv = (
        0.74 * (sex == "female")
        + 0.19 * (sex == "male")
        + 0.10 * (pclass == 1)
        - 0.05 * (pclass == 3)
        - 0.003 * ages
    ).clip(0.05, 0.95)
    survived = np.random.binomial(1, p_surv)

    age_missing = np.random.choice([True, False], n, p=[0.20, 0.80])
    ages[age_missing] = np.nan

    return pd.DataFrame({
        "PassengerId": range(1, n + 1),
        "Survived":    survived,
        "Pclass":      pclass,
        "Sex":         sex,
        "Age":         np.round(ages, 1),
        "SibSp":       sibsp,
        "Parch":       parch,
        "Fare":        np.round(fares, 2),
    })


# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  EXPLORATORY DATA ANALYSIS — TITANIC DATASET")
print("=" * 60)

df = generate_titanic_data()
print(f"\n[1] Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())

# ── 2. Data types & info ──────────────────────────────────────────────────────
print("\n[2] Data Types & Non-null Counts:")
buf = StringIO()
df.info(buf=buf)
print(buf.getvalue())

# ── 3. Statistical summary ────────────────────────────────────────────────────
print("[3] Statistical Summary:")
print(df.describe().round(2))

# ── 4. Missing value analysis ─────────────────────────────────────────────────
print("\n[4] Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
print(pd.DataFrame({"Count": missing, "Percentage (%)": missing_pct}))

# ── 5. Visualisations ─────────────────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted")
fig, axes = plt.subplots(3, 2, figsize=(14, 16))
fig.suptitle("Titanic EDA Dashboard", fontsize=18, fontweight="bold", y=1.01)

# 5a. Survival count
sns.countplot(x="Survived", data=df, ax=axes[0, 0], palette=["#e74c3c", "#2ecc71"])
axes[0, 0].set_title("Survival Count")
axes[0, 0].set_xticklabels(["Did Not Survive", "Survived"])

# 5b. Age distribution
df["Age"].dropna().hist(bins=30, ax=axes[0, 1], color="#3498db", edgecolor="white")
axes[0, 1].set_title("Age Distribution")
axes[0, 1].set_xlabel("Age")

# 5c. Fare distribution (log scale)
df["Fare"].hist(bins=40, ax=axes[1, 0], color="#9b59b6", edgecolor="white")
axes[1, 0].set_title("Fare Distribution")
axes[1, 0].set_xlabel("Fare ($)")

# 5d. Survival by Pclass
survival_pclass = df.groupby("Pclass")["Survived"].mean().reset_index()
sns.barplot(x="Pclass", y="Survived", data=survival_pclass, ax=axes[1, 1], palette="Blues_d")
axes[1, 1].set_title("Survival Rate by Passenger Class")
axes[1, 1].set_ylabel("Survival Rate")

# 5e. Survival by Sex
survival_sex = df.groupby("Sex")["Survived"].mean().reset_index()
sns.barplot(x="Sex", y="Survived", data=survival_sex, ax=axes[2, 0], palette=["#e67e22", "#1abc9c"])
axes[2, 0].set_title("Survival Rate by Sex")
axes[2, 0].set_ylabel("Survival Rate")

# 5f. Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[2, 1], linewidths=0.5)
axes[2, 1].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("eda_dashboard.png", dpi=150, bbox_inches="tight")
print("\n[5] Plots saved → eda_dashboard.png")

# ── 6. Key insights ───────────────────────────────────────────────────────────
print("\n[6] Key Insights:")
print(f"  • Overall survival rate : {df['Survived'].mean():.1%}")
for cls in [1, 2, 3]:
    rate = df[df['Pclass'] == cls]['Survived'].mean()
    print(f"  • Class {cls} survival rate : {rate:.1%}")
for sex in ["female", "male"]:
    rate = df[df['Sex'] == sex]['Survived'].mean()
    print(f"  • {sex.capitalize()} survival rate : {rate:.1%}")
print(f"  • Missing Age values    : {df['Age'].isna().sum()} ({df['Age'].isna().mean():.1%})")
print("\nEDA complete!")