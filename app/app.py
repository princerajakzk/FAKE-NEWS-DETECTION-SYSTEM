"""
app.py — TruthLine Fake News Detection — Flask Backend
=======================================================
Routes:
  POST /predict             — Analyze a news article (text or URL)
  POST /batch               — Batch CSV analysis
  GET  /stats               — Global history + stats + trending
  GET  /history/search      — Search history
  GET  /model-comparison    — Bar chart data
  GET  /wordcloud/<label>   — Word-cloud image (base64)
  GET  /live-news           — Live internet news feed with AI labeling
  POST /login               — User authentication
  POST /register            — User registration
  POST /logout              — Clear session
  GET  /me                  — Current session user
  POST /feedback            — User correctness feedback
  GET  /user/stats          — Per-user stats
  GET  /user/history        — Per-user history
  POST /api-key/generate    — Generate API key
  POST /api/v1/predict      — External REST API (requires X-API-Key header)
"""

import os
import sys
import io
import re
import csv
import json
import base64
import hashlib
import secrets
import urllib.parse
import xml.etree.ElementTree as ET
from functools import wraps

import joblib
import numpy as np
import requests as http_requests
from flask import Flask, request, jsonify, render_template, session

# ── load .env manually (no python-dotenv dependency needed) ──────────────────
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# ── local imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from analysis import (
    sensationalism_score, sensationalism_label,
    readability_score, readability_label,
    source_credibility, verify_on_internet,
    check_fact_database, combine_verdict,
)
from database import (
    init_db, create_user, get_user_by_identifier, get_user_by_id,
    add_history, set_feedback, get_history, get_user_history,
    get_user_stats, get_trending, get_stats,
    create_api_key, validate_api_key,
)

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# ── Model paths ───────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.pkl")
VECT_PATH  = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
COMP_PATH  = os.path.join(MODEL_DIR, "model_comparison.csv")
NAME_PATH  = os.path.join(MODEL_DIR, "best_model_name.txt")

# ── Load ML model once at startup ────────────────────────────────────────────
try:
    _clf   = joblib.load(MODEL_PATH)
    _vect  = joblib.load(VECT_PATH)
    _mname = open(NAME_PATH).read().strip() if os.path.exists(NAME_PATH) else "ML Model"
    print(f"[TruthLine] Model loaded: {_mname}")
except Exception as e:
    _clf = _vect = None
    _mname = "Model unavailable"
    print(f"[TruthLine] Could not load model: {e}")

# ── Initialise DB ─────────────────────────────────────────────────────────────
init_db()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _hash_pw(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _current_user():
    uid = session.get("user_id")
    if uid:
        return get_user_by_id(uid)
    return None


def _ml_predict(text: str):
    """Run text through TF-IDF + classifier. Returns (label, confidence, top_words)."""
    if _clf is None or _vect is None:
        return "Uncertain", 50.0, []
    try:
        vec   = _vect.transform([text])
        proba = _clf.predict_proba(vec)[0]
        idx   = int(np.argmax(proba))
        label = "Real" if idx == 1 else "Fake"
        conf  = round(float(proba[idx]) * 100, 1)

        # Top-words from TF-IDF feature names
        top_words = []
        try:
            feature_names = _vect.get_feature_names_out()
            dense = np.asarray(vec.todense()).flatten()
            top_idx = np.argsort(dense)[::-1][:8]
            top_words = [feature_names[i] for i in top_idx if dense[i] > 0]
        except Exception:
            pass

        return label, conf, top_words
    except Exception as e:
        print(f"[predict] ML error: {e}")
        return "Uncertain", 50.0, []


def _extract_text_from_url(url: str):
    """Try to pull article text from a URL using newspaper3k or a simple fallback."""
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        text = (art.title or "") + " " + (art.text or "")
        return text.strip(), url
    except Exception:
        pass
    # Fallback: raw requests + strip HTML
    try:
        r = http_requests.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        raw = re.sub(r"<[^>]+>", " ", r.text)
        raw = re.sub(r"\s+", " ", raw).strip()
        return raw[:5000], url
    except Exception as e:
        return "", url


def _full_analyze(text: str, source_url: str = None) -> dict:
    """Run all signals and return a complete result dict."""
    # 1. Heuristics
    s_score = sensationalism_score(text)
    s_label = sensationalism_label(s_score)
    r_score = readability_score(text)
    r_label = readability_label(r_score)

    # 2. Source credibility (if URL given)
    cred = source_credibility(source_url) if source_url else None

    # 3. ML model
    ai_label, ai_conf, top_words = _ml_predict(text)

    # 4. Multi-clause live internet search
    net_verify = verify_on_internet(text)

    # 5. Gemini AI Fact-Checking Module
    from analysis import check_gemini_ai
    gemini_res = check_gemini_ai(text)

    # 6. Google Fact-Check database
    fact_check = check_fact_database(text)

    # 7. Final combined verdict
    verdict = combine_verdict(ai_label, ai_conf, net_verify, fact_check, cred, gemini_res)

    return {
        "label":                verdict["final_label"],
        "confidence":           verdict["final_confidence"],
        "ai_label":             ai_label,
        "ai_confidence":        ai_conf,
        "model_used":           _mname,
        "top_words":            top_words,
        "sensationalism":       s_score,
        "sensationalism_label": s_label,
        "readability":          r_score,
        "readability_label":    r_label,
        "credibility":          cred,
        "source":               source_url,
        "net_verify":           net_verify,
        "gemini":               gemini_res,
        "fact_check":           fact_check,
        "verdict_score":        verdict["score"],
        "verdict_reasons":      verdict["reasons"],
        "fact_check_match":     verdict["fact_check_match"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PAGE
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html", model_name=_mname)


# ═══════════════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/register", methods=["POST"])
def register():
    data     = request.get_json(force=True) or {}
    username = data.get("username", "").strip()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "").strip()
    if not username or not email or not password:
        return jsonify({"error": "All fields required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400
    uid, err = create_user(username, email, _hash_pw(password))
    if err:
        return jsonify({"error": err}), 409
    session["user_id"] = uid
    user = get_user_by_id(uid)
    return jsonify({"user": user}), 201


@app.route("/login", methods=["POST"])
def login():
    data       = request.get_json(force=True) or {}
    identifier = data.get("identifier", "").strip()
    password   = data.get("password", "").strip()
    if not identifier or not password:
        return jsonify({"error": "Please fill in all fields."}), 400
    user = get_user_by_identifier(identifier)
    if not user or user["password_hash"] != _hash_pw(password):
        return jsonify({"error": "Invalid credentials."}), 401
    session["user_id"] = user["id"]
    return jsonify({"user": {k: user[k] for k in ("id", "username", "email", "created_at")}})


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/me")
def me():
    user = _current_user()
    if user:
        return jsonify({"user": {k: user[k] for k in ("id", "username", "email", "created_at")}})
    return jsonify({"user": None})


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICT (single article)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/predict", methods=["POST"])
def predict():
    data       = request.get_json(force=True) or {}
    text       = data.get("text", "").strip()
    req_type   = data.get("type", "text")    # "text" | "url"
    source_url = None

    if req_type == "url":
        source_url = text
        text, _    = _extract_text_from_url(source_url)
        if not text:
            return jsonify({"error": "Could not fetch article text from that URL."}), 400

    if not text:
        return jsonify({"error": "Please provide some text to analyze."}), 400
    if len(text) < 15:
        return jsonify({"error": "Text is too short to analyze (min 15 characters)."}), 400

    result = _full_analyze(text, source_url)

    # Persist to DB
    user     = _current_user()
    entry_id = add_history({
        "user_id":        user["id"] if user else None,
        "text_preview":   text[:120],
        "full_text":      text[:3000],
        "label":          result["label"],
        "confidence":     result["confidence"],
        "model_used":     _mname,
        "sensationalism": result["sensationalism"],
        "readability":    result["readability"],
        "credibility":    result["credibility"]["level"] if result["credibility"] else None,
        "source":         source_url,
    })
    result["entry_id"] = entry_id
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH CSV ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/batch", methods=["POST"])
def batch():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    f = request.files["file"]
    try:
        content = f.read().decode("utf-8", errors="replace")
        reader  = csv.DictReader(io.StringIO(content))
        if "text" not in (reader.fieldnames or []):
            return jsonify({"error": "CSV must have a 'text' column."}), 400
        rows = [row for row in reader][:50]   # cap at 50 rows
    except Exception as e:
        return jsonify({"error": f"CSV parse error: {e}"}), 400

    results = []
    for row in rows:
        t = row.get("text", "").strip()
        if not t:
            continue
        s_score = sensationalism_score(t)
        ai_label, ai_conf, _ = _ml_predict(t)
        results.append({
            "text":           t[:80] + ("..." if len(t) > 80 else ""),
            "label":          ai_label,
            "confidence":     ai_conf,
            "sensationalism": s_score,
        })

    return jsonify({"count": len(results), "results": results})


# ═══════════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/feedback", methods=["POST"])
def feedback():
    data     = request.get_json(force=True) or {}
    entry_id = data.get("entry_id")
    correct  = data.get("correct")
    if entry_id is None:
        return jsonify({"error": "entry_id required"}), 400
    set_feedback(entry_id, "correct" if correct else "incorrect")
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════════════
# STATS + HISTORY
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/stats")
def stats():
    s = get_stats()
    s["history"]  = get_history(limit=50)
    s["trending"] = get_trending(limit=5)
    return jsonify(s)


@app.route("/history/search")
def history_search():
    q = request.args.get("q", "").strip()
    return jsonify(get_history(limit=50, search=q if q else None))


@app.route("/user/stats")
def user_stats():
    user = _current_user()
    if not user:
        return jsonify({"error": "Not logged in"}), 401
    return jsonify(get_user_stats(user["id"]))


@app.route("/user/history")
def user_history():
    user = _current_user()
    if not user:
        return jsonify([])
    return jsonify(get_user_history(user["id"], limit=20))


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPARISON CHART
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/model-comparison")
def model_comparison():
    labels, accuracy = [], []
    try:
        with open(COMP_PATH, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.append(row.get("model", row.get("Model", "?")))
                acc_val = row.get("accuracy", row.get("Accuracy", "0"))
                try:
                    accuracy.append(float(acc_val))
                except ValueError:
                    accuracy.append(0.0)
    except Exception:
        labels   = [_mname]
        accuracy = [0.95]
    return jsonify({"labels": labels, "accuracy": accuracy})


# ═══════════════════════════════════════════════════════════════════════════════
# WORD CLOUD
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/wordcloud/<label>")
def wordcloud(label: str):
    label = label.lower()
    if label not in ("real", "fake"):
        return jsonify({"image": None}), 400
    try:
        from wordcloud import WordCloud
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows  = get_history(limit=200)
        texts = [r["full_text"] or r["text_preview"]
                 for r in rows if (r.get("label") or "").lower() == label]
        if not texts:
            return jsonify({"image": None})

        combined = " ".join(texts)
        color    = "#3ECF8E" if label == "real" else "#FF5C5C"

        wc = WordCloud(
            width=600, height=300, background_color="#12152B",
            color_func=lambda *a, **k: color,
            max_words=80, collocations=False
        ).generate(combined)

        fig, ax = plt.subplots(figsize=(6, 3), facecolor="#12152B")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", facecolor="#12152B")
        plt.close(fig)
        buf.seek(0)
        img_b64 = "data:image/png;base64," + base64.b64encode(buf.read()).decode()
        return jsonify({"image": img_b64})
    except Exception as e:
        return jsonify({"image": None, "error": str(e)})


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE NEWS (real-time internet feed via Google News RSS)
# ═══════════════════════════════════════════════════════════════════════════════

CATEGORY_QUERIES = {
    "general":       "top news",
    "technology":    "technology AI innovation",
    "science":       "science research discovery",
    "health":        "health medicine",
    "sports":        "sports cricket football",
    "business":      "business economy markets",
    "entertainment": "entertainment bollywood movies",
    "politics":      "politics government election",
    "india":         "India news today",
    "world":         "world news international",
}

DEMO_ARTICLES = [
    {"title": "Scientists discover breakthrough in renewable energy storage",
     "description": "Researchers have developed a battery technology that stores solar energy for months.",
     "source": "MIT News", "url": "", "urlToImage": None, "publishedAt": None},
    {"title": "Global markets rally on positive economic data",
     "description": "Stock markets rose sharply after better-than-expected GDP figures.",
     "source": "Reuters", "url": "", "urlToImage": None, "publishedAt": None},
    {"title": "SHOCKING: Celebrities secretly control the weather, insiders reveal",
     "description": "Anonymous sources claim Hollywood A-listers have been directing storm systems for decades.",
     "source": "SensationalDaily.net", "url": "", "urlToImage": None, "publishedAt": None},
]


@app.route("/live-news")
def live_news():
    category = request.args.get("category", "general").strip().lower()
    query    = CATEGORY_QUERIES.get(category, "top news")
    enc_q    = urllib.parse.quote(query)
    rss_url  = f"https://news.google.com/rss/search?q={enc_q}&hl=en-IN&gl=IN&ceid=IN:en"

    articles = []
    is_demo  = False
    provider = None

    try:
        resp = http_requests.get(
            rss_url, timeout=8,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        )
        if resp.status_code == 200:
            root  = ET.fromstring(resp.content)
            items = root.findall(".//item")
            for item in items[:12]:
                title_el   = item.find("title")
                link_el    = item.find("link")
                pub_el     = item.find("pubDate")
                source_el  = item.find("source")
                desc_el    = item.find("description")

                raw_title   = (title_el.text or "")  if title_el  is not None else ""
                source_name = (source_el.text or "News") if source_el is not None else "News"
                pub_date    = (pub_el.text or None)   if pub_el    is not None else None
                description = (desc_el.text or "")   if desc_el   is not None else ""

                title = raw_title
                if " - " in raw_title:
                    parts       = raw_title.rsplit(" - ", 1)
                    title       = parts[0].strip()
                    source_name = parts[1].strip()

                # Google News RSS: link text is between <link/> tag and next element
                link = ""
                if link_el is not None:
                    link = (link_el.tail or "").strip()
                    if not link and link_el.text:
                        link = link_el.text.strip()

                articles.append({
                    "title":       title,
                    "description": re.sub(r"<[^>]+>", "", description).strip(),
                    "source":      source_name,
                    "url":         link,
                    "urlToImage":  None,
                    "publishedAt": pub_date,
                })
            provider = "Google News RSS"
    except Exception as e:
        print(f"[live-news] RSS error: {e}")

    if not articles:
        articles = DEMO_ARTICLES
        is_demo  = True

    # Run ML + heuristics on each article
    enriched = []
    for art in articles:
        text     = (art["title"] or "") + " " + (art.get("description") or "")
        ai_label, ai_conf, _ = _ml_predict(text)
        s_score  = sensationalism_score(text)
        enriched.append({
            **art,
            "label":                ai_label,
            "confidence":           ai_conf,
            "sensationalism":       s_score,
            "sensationalism_label": sensationalism_label(s_score),
        })

    return jsonify({"articles": enriched, "is_demo": is_demo, "provider": provider})


# ═══════════════════════════════════════════════════════════════════════════════
# API KEY MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api-key/generate", methods=["POST"])
def api_key_generate():
    data  = request.get_json(force=True) or {}
    label = data.get("label", "default")
    key   = create_api_key(label)
    return jsonify({"api_key": key})


# ═══════════════════════════════════════════════════════════════════════════════
# EXTERNAL REST API  (requires X-API-Key header)
# ═══════════════════════════════════════════════════════════════════════════════

def _require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key", "")
        row = validate_api_key(key)
        if not row:
            return jsonify({"error": "Invalid or missing API key."}), 401
        return f(*args, **kwargs)
    return decorated


@app.route("/api/v1/predict", methods=["POST"])
@_require_api_key
def api_predict():
    data = request.get_json(force=True) or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "text field is required"}), 400

    result = _full_analyze(text)
    return jsonify({
        "label":                result["label"],
        "confidence":           result["confidence"],
        "ai_label":             result["ai_label"],
        "ai_confidence":        result["ai_confidence"],
        "model_used":           result["model_used"],
        "sensationalism":       result["sensationalism"],
        "sensationalism_label": result["sensationalism_label"],
        "readability":          result["readability"],
        "readability_label":    result["readability_label"],
        "verdict_score":        result["verdict_score"],
        "verdict_reasons":      result["verdict_reasons"],
    })


# ═══════════════════════════════════════════════════════════════════════════════
# STATIC BACKGROUND PLACEHOLDER
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/static/bg.png")
def bg_png():
    """Return 1x1 transparent PNG if background image doesn't exist on disk."""
    from flask import send_from_directory, Response
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    bg_path    = os.path.join(static_dir, "bg.png")
    if os.path.exists(bg_path):
        return send_from_directory(static_dir, "bg.png")
    PNG_1x1 = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    )
    return Response(PNG_1x1, mimetype="image/png")


# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI API KEY SAVE ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/set-gemini-key", methods=["POST"])
def set_gemini_key():
    data = request.get_json(force=True) or {}
    key = data.get("api_key", "").strip()
    if not key:
        return jsonify({"error": "Key cannot be empty"}), 400

    os.environ["GEMINI_API_KEY"] = key

    # Write to .env
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    lines = []
    found = False
    if os.path.exists(env_file):
        for l in open(env_file):
            if l.startswith("GEMINI_API_KEY="):
                lines.append(f"GEMINI_API_KEY={key}\n")
                found = True
            else:
                lines.append(l)
    if not found:
        lines.append(f"\nGEMINI_API_KEY={key}\n")

    with open(env_file, "w") as f:
        f.writelines(lines)

    return jsonify({"ok": True, "message": "Gemini API Key saved successfully!"})


# ═══════════════════════════════════════════════════════════════════════════════
# RUN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  TruthLine — Fake News Detection System")
    print(f"  Model: {_mname}")
    print("  URL:   http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)
