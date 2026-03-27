"""
Fake News Detection
====================
Classifies news articles as real or fake using:
- Text preprocessing (lowercase, punctuation, stopwords)
- TF-IDF feature extraction
- Logistic Regression, Naive Bayes, and SVM
- Evaluation with accuracy, F1-score, confusion matrix
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

np.random.seed(42)

# ── Synthetic news corpus ─────────────────────────────────────────────────────
REAL_TEMPLATES = [
    "Scientists at {} University published findings in {} journal showing evidence of {}.",
    "The government announced new {} policy affecting millions of citizens nationwide.",
    "Stock markets rose {} percent today following positive {} economic data.",
    "Health officials confirmed {} cases of {} reported across multiple states.",
    "A new study involving {} participants found that {} may reduce risk of {}.",
    "The {} central bank raised interest rates by {} basis points citing inflation concerns.",
    "Researchers discovered a potential treatment for {} using {} compounds.",
    "International leaders met in {} to discuss the ongoing {} crisis.",
]
FAKE_TEMPLATES = [
    "SHOCKING: {} secretly controls {} and nobody is talking about it!!!",
    "BREAKING: {} CONFIRMS that {} causes {} — mainstream media SILENT!",
    "You won't believe what {} is hiding from you about {} — share before deleted!",
    "{} EXPOSES the truth about {}: the {} conspiracy they don't want you to know!",
    "URGENT WARNING: {} is planning to {} by {} — spread the word NOW!",
    "Scientists BANNED from revealing {} link to {}! Here's what they found!",
    "This {} trick reverses {} in just {} days — doctors hate this!",
    "EXCLUSIVE: {} whistleblower leaks {} documents proving {} coverup!",
]
ENTITIES = ["Harvard", "MIT", "Stanford", "Oxford", "NASA", "WHO", "CDC", "FBI",
            "vaccines", "5G towers", "water supply", "climate change", "economy",
            "government", "big pharma", "election results"]
NUMBERS  = ["12", "45", "3", "100", "0.5", "72", "1000", "seven"]

def make_article(template, fake=False):
    ents = np.random.choice(ENTITIES, 3, replace=False)
    nums = np.random.choice(NUMBERS,  1)[0]
    try:
        return template.format(*ents[:3]), int(fake)
    except Exception:
        return template.format(ents[0], ents[1], nums), int(fake)

def generate_news(n_per_class=500):
    rows = []
    for _ in range(n_per_class):
        t = np.random.choice(REAL_TEMPLATES)
        text, label = make_article(t, fake=False)
        # Add variety
        text += " " + " ".join(np.random.choice(
            ["The report was peer-reviewed.", "Data was collected over five years.",
             "Experts called the findings significant.", "Further research is planned."],
            k=np.random.randint(1, 3)))
        rows.append({"text": text, "label": label})
    for _ in range(n_per_class):
        t = np.random.choice(FAKE_TEMPLATES)
        text, label = make_article(t, fake=True)
        text += " " + " ".join(np.random.choice(
            ["Like and share!", "Wake up sheeple!", "Do your own research!",
             "They deleted this twice already!", "Forward to everyone you know!"],
            k=np.random.randint(1, 3)))
        rows.append({"text": text, "label": label})
    df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
    return df

# ── Preprocessing ─────────────────────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

print("=" * 60)
print("  FAKE NEWS DETECTION")
print("=" * 60)

df = generate_news(500)
df["clean_text"] = df["text"].apply(clean_text)
print(f"\nDataset size : {len(df)}")
print(f"Fake news    : {df['label'].sum()} ({df['label'].mean():.1%})")

X = df["clean_text"]
y = df["label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── Pipelines ─────────────────────────────────────────────────────────────────
pipelines = {
    "Logistic Regression": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, random_state=42)),
    ]),
    "Naive Bayes": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf",   MultinomialNB()),
    ]),
    "Linear SVM": Pipeline([
        ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
        ("clf",   LinearSVC(random_state=42, max_iter=2000)),
    ]),
}

results = {}
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    cv    = cross_val_score(pipe, X_train, y_train, cv=5, scoring="f1").mean()
    results[name] = {
        "acc": accuracy_score(y_test, preds),
        "f1":  f1_score(y_test, preds),
        "cv_f1": cv,
        "preds": preds,
        "pipe": pipe,
    }
    print(f"\n── {name} ──")
    print(f"  Accuracy     : {results[name]['acc']:.4f}")
    print(f"  F1 Score     : {results[name]['f1']:.4f}")
    print(f"  CV F1 (5-fold): {cv:.4f}")
    print(classification_report(y_test, preds, target_names=["Real", "Fake"]))

# ── Top TF-IDF features ───────────────────────────────────────────────────────
lr_pipe = pipelines["Logistic Regression"]
feat_names = lr_pipe.named_steps["tfidf"].get_feature_names_out()
coefs      = lr_pipe.named_steps["clf"].coef_[0]
top_fake = pd.Series(coefs, index=feat_names).nlargest(10)
top_real = pd.Series(coefs, index=feat_names).nsmallest(10)

print("\nTop 10 words → FAKE news:")
print(top_fake.round(3).to_string())
print("\nTop 10 words → REAL news:")
print(top_real.round(3).to_string())

# ── Visualisations ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Fake News Detection", fontsize=15, fontweight="bold")

best_name = max(results, key=lambda k: results[k]["f1"])
cm = confusion_matrix(y_test, results[best_name]["preds"])
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", ax=axes[0],
            xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
axes[0].set_title(f"Confusion Matrix ({best_name})")

model_names = list(results.keys())
f1_scores   = [results[m]["f1"] for m in model_names]
axes[1].bar(model_names, f1_scores, color=["#2ecc71", "#3498db", "#e74c3c"])
axes[1].set_ylim(0, 1.1)
axes[1].set_title("F1 Score Comparison")
axes[1].set_ylabel("F1 Score")
for i, v in enumerate(f1_scores):
    axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("fake_news_results.png", dpi=150, bbox_inches="tight")
print("\nPlots saved → fake_news_results.png")

# ── Live demo ─────────────────────────────────────────────────────────────────
demo_texts = [
    "Researchers at Johns Hopkins University published a peer-reviewed study on vaccine efficacy.",
    "SHOCKING: Government hiding PROOF that 5G causes mind control — share before deleted!!!",
]
best_pipe = pipelines[best_name]
print(f"\n── Live Demo ({best_name}) ──")
for text in demo_texts:
    label = best_pipe.predict([clean_text(text)])[0]
    print(f"\n  Input  : {text[:70]}...")
    print(f"  Result : {'🚨 FAKE' if label == 1 else '✅ REAL'}")

print("\nFake news detection complete!")
