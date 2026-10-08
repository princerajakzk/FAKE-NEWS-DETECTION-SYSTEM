# Truthline Premium — AI Based Fake News Detection System

An NLP + Machine Learning system that detects whether a news article, headline,
or URL is likely **Real** or **Fake**, wrapped in a full-featured premium dashboard
with batch analysis, credibility checks, feedback learning, and a REST API.

**Developed by:** Prince · Chandigarh University

---

## What's New in Premium

| Feature | Before | Premium |
|---|---|---|
| Storage | In-memory (reset on restart) | **SQLite database** — history & stats persist |
| Input | Text, URL | Text, URL, **Batch CSV upload (up to 200 rows)** |
| Verdict | Label + confidence | + **Clickbait/Sensationalism score**, **Readability score (Flesch)** |
| URL checks | Extraction only | + **Source credibility lookup** against a reputation list |
| Feedback | None | **👍 / 👎 feedback loop** feeding a real "user-confirmed accuracy" stat |
| Access | Web UI only | + **Rate-limited REST API** (`/api/v1/predict`) with API-key generation |
| Export | PDF (single result) | + **CSV export** of full history and batch results |
| Model | 4 classifiers, unigram TF-IDF | 5 classifiers incl. a **Voting Ensemble**, unigram+bigram TF-IDF, **calibrated SVM probabilities** |

## Features
- Text, URL, **and batch CSV** input (auto article extraction via newspaper3k)
- Text preprocessing pipeline (cleaning, TF-IDF vectorization with bigrams)
- 5 ML models trained and compared: Logistic Regression, Naive Bayes, Random Forest,
  calibrated SVM, and a soft-voting Ensemble — best model auto-selected and saved
- **Explainability** — highlights which words influenced the verdict
- **Clickbait / Sensationalism meter** — flags ALL-CAPS, excessive punctuation, and
  sensational phrasing
- **Readability score** — Flesch Reading Ease, computed with no extra dependency
- **Source credibility badge** — flags known reliable / low-credibility / unknown domains
- **Feedback loop** — mark any verdict correct/incorrect; tracked in the Statistics page
- **Batch analysis** — upload a CSV of headlines, get a verdict for every row, export results
- **Premium REST API** — generate an API key, call `/api/v1/predict` with rate limiting
- **Word Cloud** visualizations of common real vs fake news words
- **Trending predictions** — last 3 checks shown as cards
- **Animated confidence gauge** (circular dial chart)
- **PDF export** of any single result, **CSV export** of full history
- **Dark / Light theme toggle**
- **Searchable, persistent history** table (SQLite-backed)
- Live Statistics dashboard with animated counters and charts
- Digital Literacy tips section
- Plans/API page documenting Free vs Premium capabilities

## Project Structure
```
fake-news-detection/
├── dataset/
│   ├── True.csv                  # Verified authentic journalistic dataset (~21,000 articles)
│   └── Fake.csv                  # Verified fabricated misinformation dataset (~23,000 articles)
├── notebooks/
│   ├── model_training.py         # Trains, benchmarks, and exports the active Random Forest classifier
│   └── retrain_with_live_data.py # Continuous learning pipeline ingesting SQLite feedback
├── models/
│   ├── best_model.pkl            # Active production model (Random Forest, 99.78% accuracy)
│   ├── tfidf_vectorizer.pkl      # 5,000 n-gram TF-IDF vectorizer
│   ├── best_model_name.txt       # Active model name identifier
│   ├── model_comparison.csv      # Algorithm benchmark comparison matrix
│   ├── confusion_matrix.png      # Validation confusion matrix plot
│   ├── model_comparison_chart.png# Algorithm benchmark bar chart
│   ├── dataset_distribution.png  # Training class distribution chart
│   └── truthline.db              # SQLite production database (history, users, api_keys)
├── app/
│   ├── app.py                    # Production Flask server & REST API
│   ├── analysis.py               # Multi-signal verification engine (Gemini AI, RSS search, heuristics)
│   ├── database.py               # SQLite persistent ORM layer
│   ├── mailer.py                 # Google SMTP OTP delivery service
│   ├── static/                   # Static media assets, icons, and background textures
│   ├── templates/
│   │   └── index.html            # TruthLine 3D cosmic investigative dashboard
│   └── .env                      # Environment variables & API keys
├── requirements.txt
└── README.md
```

## Setup & Running

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Dataset & Pre-trained Model
The production model (`best_model.pkl` with 99.78% accuracy) and TF-IDF vectorizer are already pre-trained and included in `models/`. The training datasets `True.csv` and `Fake.csv` (44,898 articles) are in `dataset/`.

To re-train the models from scratch:
```bash
python notebooks/model_training.py
```

### 3. Run the dashboard
```bash
cd app
python app.py
```
Open **http://127.0.0.1:5000** in your browser. All history, feedback, users, and API tokens persist inside `models/truthline.db`.

## Using the Premium Features

### Batch analysis
Go to **Check News → Batch Upload**, choose a CSV file with a `text` column (or any single
column), and click **Run Batch Analysis**. Results can be downloaded as CSV.

### Feedback loop
After any single check, click **👍 Yes** or **👎 No** under "Was this verdict correct?".
The Statistics page shows a live **User-Confirmed Accuracy** stat built from this feedback.

### REST API
Go to **API & Plans**, click **Generate API Key**, then call:
```bash
curl -X POST http://127.0.0.1:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_API_KEY" \
  -d '{"text": "Scientists confirm new vaccine trial results"}'
```
The endpoint is rate-limited to 60 requests/minute per key.

## If You Already Have a Trained Model
If you've already trained on the real Kaggle dataset and have working `models/` files,
**do not overwrite that folder** — just drop in the new `app/`, `requirements.txt`, and
`notebooks/model_training.py`, then `pip install -r requirements.txt` for the new libraries.
Your existing trained model will keep working with the upgraded dashboard automatically
(the DB and new scores are computed independently of the model file).

## Tech Stack
Python · Scikit-learn · TF-IDF (1-2 grams) · Flask · Flask-Limiter · SQLite ·
newspaper3k · WordCloud · Chart.js · jsPDF

## Report Screenshot Checklist
1. Dataset sample rows (True.csv / Fake.csv)
2. `models/dataset_distribution.png`
3. `models/model_comparison_chart.png` (5 models incl. ensemble)
4. `models/confusion_matrix.png`
5. Model comparison table (terminal output)
6. Dashboard — Check News tab with a Real result (with explainability chips + meters)
7. Dashboard — Check News tab with a Fake result
8. Dashboard — Batch Upload tab with results table
9. Dashboard — Feedback buttons in use
10. Dashboard — Trending predictions section
11. Dashboard — Statistics page (charts + word clouds + accuracy stat)
12. Dashboard — API & Plans page with a generated key
13. Dashboard — How It Works page
