"""
retrain_with_live_data.py — Continuous Learning & Live Retraining Pipeline
-------------------------------------------------------------------------
This script combines:
1. Base training dataset (dataset/True.csv & dataset/Fake.csv)
2. Live real-time internet-verified predictions & feedback from models/truthline.db

Retrains the ensemble & ML classifiers to continuously learn from live real-world news!
"""

import os
import re
import sqlite3
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_PATH = os.path.join(MODELS_DIR, "truthline.db")

print("=" * 60)
print("  TruthLine — Continuous Retraining Pipeline")
print("=" * 60)

# 1. Load Base Dataset
print("\n[1/5] Loading base dataset...")
true_path = os.path.join(DATASET_DIR, "True.csv")
fake_path = os.path.join(DATASET_DIR, "Fake.csv")

dfs = []
if os.path.exists(true_path) and os.path.exists(fake_path):
    t_df = pd.read_csv(true_path)
    f_df = pd.read_csv(fake_path)
    t_df["label"] = 1
    f_df["label"] = 0
    base_df = pd.concat([t_df, f_df], ignore_index=True)
    base_df["content"] = base_df.get("title", "").fillna("") + " " + base_df.get("text", "").fillna("")
    dfs.append(base_df[["content", "label"]])
    print(f"  Base dataset loaded: {len(base_df)} samples")

# 2. Extract Live Verified History from SQLite Database
print("\n[2/5] Ingesting live verified news from real-time database...")
if os.path.exists(DB_PATH):
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT full_text as content, label
            FROM history
            WHERE label IN ('Real', 'Fake') AND length(full_text) > 15
        """
        db_df = pd.read_sql_query(query, conn)
        conn.close()
        if not db_df.empty:
            db_df["label"] = db_df["label"].map({"Real": 1, "Fake": 0})
            dfs.append(db_df)
            print(f"  Ingested {len(db_df)} live real-world news records from database!")
        else:
            print("  No previous history records found in DB yet.")
    except Exception as e:
        print(f"  Warning reading database: {e}")

if not dfs:
    print("Error: No data available to train.")
    exit(1)

full_df = pd.concat(dfs, ignore_index=True)

# 3. Clean Text
print("\n[3/5] Preprocessing and cleaning text (stripping dataset wire artifacts)...")
def clean_text(text):
    text = str(text or "")
    text = re.sub(r'^[A-Z\s,]+(?:\([A-Za-z\s]+\))?\s*[-—]\s*', '', text)
    text = re.sub(r'\b(?:reuters|reutersipsos)\b', '', text, flags=re.I)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

full_df["clean_content"] = full_df["content"].apply(clean_text)
full_df = full_df[full_df["clean_content"].str.len() > 10].drop_duplicates(subset=["clean_content"])
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"  Total unique training samples: {len(full_df)} (Real: {(full_df['label']==1).sum()}, Fake: {(full_df['label']==0).sum()})")

# 4. TF-IDF & Train Models
print("\n[4/5] Training models on combined live + base data...")
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
custom_stops = list(ENGLISH_STOP_WORDS.union({"reuters", "reutersipsos", "washington", "said", "told", "reported", "via"}))
tfidf = TfidfVectorizer(max_features=8000, stop_words=custom_stops, ngram_range=(1, 2))
X = tfidf.fit_transform(full_df["clean_content"])
y = full_df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

lr = LogisticRegression(max_iter=1000)
nb = MultinomialNB()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = CalibratedClassifierCV(LinearSVC(), cv=3)

models = {
    "Logistic Regression": lr,
    "Naive Bayes": nb,
    "Random Forest": rf,
    "Calibrated SVM": svm
}

results = []
trained_models = {}

for name, mdl in models.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results.append({"Model": name, "Accuracy": acc})
    trained_models[name] = mdl
    print(f"  Model: {name:22s} -> Accuracy: {acc*100:.2f}%")

# Train Soft Voting Ensemble
print("  Training Soft Voting Ensemble...")
ensemble = VotingClassifier(
    estimators=[("lr", lr), ("nb", nb), ("svm", svm)],
    voting="soft"
)
ensemble.fit(X_train, y_train)
ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
results.append({"Model": "Voting Ensemble", "Accuracy": ens_acc})
trained_models["Voting Ensemble"] = ensemble
print(f"  Model: {'Voting Ensemble':22s} -> Accuracy: {ens_acc*100:.2f}%")

# Select best model
results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
best_name = results_df.iloc[0]["Model"]
best_mdl = trained_models[best_name]

# 5. Save Artifacts
print(f"\n[5/5] Saving retrained model & vectorizer ({best_name})...")
joblib.dump(best_mdl, os.path.join(MODELS_DIR, "best_model.pkl"))
joblib.dump(tfidf, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
with open(os.path.join(MODELS_DIR, "best_model_name.txt"), "w") as f:
    f.write(best_name)

print("=" * 60)
print(f"  SUCCESS: Model successfully updated with live real-time data!")
print(f"  Best Model: {best_name} (Accuracy: {results_df.iloc[0]['Accuracy']*100:.2f}%)")
print("=" * 60)
