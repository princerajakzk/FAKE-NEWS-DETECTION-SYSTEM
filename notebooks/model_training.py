"""
MODEL TRAINING SCRIPT — Premium Edition
------------------------------------------------------------------
What this does:
1. Loads the dataset (True.csv + Fake.csv)
2. Cleans text
3. Converts text to TF-IDF features (now with unigrams + bigrams)
4. Trains and compares 5 models: Logistic Regression, Naive Bayes,
   Random Forest, calibrated Linear SVM, and a Voting Ensemble of the
   three strongest individual models
5. Saves the best model + vectorizer (the app uses these)
6. Saves comparison graphs for the report

Run:
    python3 notebooks/model_training.py
"""

import pandas as pd
import numpy as np
import re
import joblib
import os
import matplotlib
matplotlib.use("Agg")  # no GUI needed on a server
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# -----------------------------
# STEP 1: LOAD DATASET
# -----------------------------
print("Step 1: Loading dataset...")
true_df = pd.read_csv(os.path.join(DATASET_DIR, "True.csv"))
fake_df = pd.read_csv(os.path.join(DATASET_DIR, "Fake.csv"))

true_df["label"] = 1   # 1 = Real
fake_df["label"] = 0   # 0 = Fake

df = pd.concat([true_df, fake_df], axis=0).reset_index(drop=True)
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

print(f"Total records: {len(df)}  |  Real: {(df['label']==1).sum()}  |  Fake: {(df['label']==0).sum()}")

# -----------------------------
# STEP 2: TEXT PREPROCESSING
# -----------------------------
print("Step 2: Cleaning text...")

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_content"] = df["content"].apply(clean_text)

# -----------------------------
# STEP 3: TF-IDF (unigrams + bigrams — captures phrases like "breaking news")
# -----------------------------
print("Step 3: Converting text to TF-IDF features (1-2 grams)...")
tfidf = TfidfVectorizer(max_features=6000, stop_words="english", ngram_range=(1, 2))
X = tfidf.fit_transform(df["clean_content"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# STEP 4: TRAIN + COMPARE MULTIPLE MODELS
# -----------------------------
print("Step 4: Training multiple models...")

log_reg = LogisticRegression(max_iter=1000)
nb = MultinomialNB()
rf = RandomForestClassifier(n_estimators=150, random_state=42)
# Calibrate the SVM so it also exposes predict_proba (needed for a real confidence %)
svm = CalibratedClassifierCV(LinearSVC(), cv=3)

models = {
    "Logistic Regression": log_reg,
    "Naive Bayes": nb,
    "Random Forest": rf,
    "SVM (Linear, calibrated)": svm,
}

results = []
trained_models = {}

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
    trained_models[name] = mdl
    print(f"  {name:28s} -> Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

# Voting ensemble of the three strongest models (soft voting on probabilities)
print("  Training Voting Ensemble (Premium)...")
ensemble = VotingClassifier(
    estimators=[("lr", log_reg), ("nb", nb), ("svm", svm)],
    voting="soft"
)
ensemble.fit(X_train, y_train)
ens_preds = ensemble.predict(X_test)
acc = accuracy_score(y_test, ens_preds)
prec = precision_score(y_test, ens_preds)
rec = recall_score(y_test, ens_preds)
f1 = f1_score(y_test, ens_preds)
results.append({"Model": "Voting Ensemble (Premium)", "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1})
trained_models["Voting Ensemble (Premium)"] = ensemble
print(f"  {'Voting Ensemble (Premium)':28s} -> Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
print("\n===== MODEL COMPARISON TABLE =====")
print(results_df.to_string(index=False))
results_df.to_csv(os.path.join(MODELS_DIR, "model_comparison.csv"), index=False)

# -----------------------------
# STEP 5: SELECT + SAVE BEST MODEL
# -----------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBest Model: {best_model_name}")

joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
    f.write(best_model_name)
print("Model and vectorizer saved to 'models/'.")

# -----------------------------
# STEP 6: GRAPHS (for the report)
# -----------------------------
print("Step 6: Generating graphs...")

plt.figure(figsize=(9, 5))
results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar")
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "model_comparison_chart.png"))
plt.close()

best_preds = best_model.predict(X_test)
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.title(f"Confusion Matrix - {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"))
plt.close()

plt.figure(figsize=(5, 4))
sns.countplot(x=df["label"].map({0: "Fake", 1: "Real"}))
plt.title("Dataset Distribution")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "dataset_distribution.png"))
plt.close()

print("\nGraphs saved to 'models/':")
print("  - model_comparison_chart.png")
print("  - confusion_matrix.png")
print("  - dataset_distribution.png")
print("\nTRAINING COMPLETE!")
