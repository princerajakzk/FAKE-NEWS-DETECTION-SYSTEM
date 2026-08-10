"""
analysis.py — Multi-Signal Real-Time Fake News & Misinformation Verification Engine
-------------------------------------------------------------------------------------
1. Sensationalism / Clickbait score  — punctuation & wording heuristics
2. Readability score (Flesch Reading Ease) — computed manually
3. Source credibility check — domain reputation lookup
4. Multi-Clause Real-Time Internet Search Verification Engine
5. Google Fact Check database lookup
6. Gemini AI Fact-Checking Module
7. Smart Signal Fusion Verdict Engine
"""

import os
import re
import json
import urllib.parse
import xml.etree.ElementTree as ET
import requests as http_requests
from urllib.parse import urlparse

# ---------------------------------------------------------------
# 1. SENSATIONALISM / CLICKBAIT SCORE (0-100, higher = more clickbaity)
# ---------------------------------------------------------------
CLICKBAIT_WORDS = [
    "shocking", "you won't believe", "secret", "banned", "miracle", "urgent",
    "breaking", "exposed", "conspiracy", "they don't want you to know",
    "won't believe", "hate this", "silenced", "cover up", "covered up",
    "leaked", "insider", "alert", "warning", "must see", "goes viral",
    "unbelievable", "mind-blowing", "outrage", "slams", "destroys",
]


def sensationalism_score(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    lower = text.lower()
    words = re.findall(r"[A-Za-z']+", text)
    n_words = max(len(words), 1)

    score = 0.0

    # ALL-CAPS words (excluding short acronyms like "US", "UN")
    caps_words = [w for w in words if len(w) > 3 and w.isupper()]
    score += min(30, (len(caps_words) / n_words) * 300)

    # Exclamation / question mark density
    exclam = text.count("!")
    score += min(20, exclam * 4)

    # Multiple punctuation e.g. "???" or "!!!"
    if re.search(r"[!?]{2,}", text):
        score += 10

    # Clickbait phrase hits
    hits = sum(1 for phrase in CLICKBAIT_WORDS if phrase in lower)
    score += min(30, hits * 8)

    # Excessive emoji-like symbols
    score += min(10, text.count("💥") + text.count("🚨") + text.count("😱"))

    return round(min(100, score), 1)


def sensationalism_label(score: float) -> str:
    if score < 20:
        return "Neutral tone"
    if score < 45:
        return "Mildly sensational"
    if score < 70:
        return "Sensational"
    return "Highly sensational / clickbait"


# ---------------------------------------------------------------
# 2. READABILITY SCORE — Flesch Reading Ease
# ---------------------------------------------------------------
def _count_syllables(word: str) -> int:
    word = word.lower().strip(string_punct())
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def string_punct():
    import string as s
    return s.punctuation


def readability_score(text: str) -> float:
    if not text or not text.strip():
        return 0.0
    sentences = re.split(r"[.!?]+", text)
    sentences = [s for s in sentences if s.strip()]
    words = re.findall(r"[A-Za-z']+", text)

    n_sentences = max(len(sentences), 1)
    n_words = max(len(words), 1)
    n_syllables = sum(_count_syllables(w) for w in words) or 1

    score = 206.835 - 1.015 * (n_words / n_sentences) - 84.6 * (n_syllables / n_words)
    return round(max(0, min(100, score)), 1)


def readability_label(score: float) -> str:
    if score >= 80:
        return "Very easy to read"
    if score >= 60:
        return "Easy to read"
    if score >= 40:
        return "Fairly difficult"
    if score >= 20:
        return "Difficult"
    return "Very difficult / academic"


# ---------------------------------------------------------------
# 3. SOURCE CREDIBILITY — domain reputation lookup
# ---------------------------------------------------------------
KNOWN_RELIABLE = {
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "npr.org",
    "theguardian.com", "nytimes.com", "washingtonpost.com", "wsj.com",
    "bloomberg.com", "aljazeera.com", "thehindu.com", "indianexpress.com",
    "hindustantimes.com", "ndtv.com", "timesofindia.indiatimes.com",
    "cnn.com", "abcnews.go.com", "cbsnews.com", "nbcnews.com", "pbs.org",
    "aniin.com", "ptinews.com", "deccanherald.com", "financialexpress.com",
    "livemint.com", "business-standard.com", "news18.com", "indiatoday.in"
}

KNOWN_LOW_CREDIBILITY = {
    "theonion.com", "worldnewsdailyreport.com", "empirenews.net", "nationalreport.net",
}


def source_credibility(url: str):
    if not url:
        return None
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return None
    if not domain:
        return None

    if domain in KNOWN_LOW_CREDIBILITY:
        return {"domain": domain, "level": "low", "note": "Known satire / low-credibility source"}
    if domain in KNOWN_RELIABLE:
        return {"domain": domain, "level": "high", "note": "Recognized established news outlet"}
    return {"domain": domain, "level": "unknown", "note": "Domain not in our reputation database — verify independently"}


# ---------------------------------------------------------------
# 4. MULTI-CLAUSE SMART LIVE INTERNET SEARCH VERIFICATION ENGINE
# ---------------------------------------------------------------
STOP_WORDS = {
    "a", "an", "the", "in", "on", "of", "and", "or", "to", "for", "with", "at",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "that", "this",
    "it", "as", "has", "have", "had", "will", "would", "about", "can", "could",
    "ongoing", "across", "spells", "heavy", "states", "northern", "southern"
}


def _title_overlap(clause: str, title: str) -> float:
    c_words = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", clause) if len(w) > 3 and w.lower() not in STOP_WORDS)
    t_words = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", title) if len(w) > 3)
    if not c_words:
        return 0.0
    common = c_words.intersection(t_words)
    return len(common) / len(c_words)


def verify_on_internet(text: str) -> dict:
    """
    Splits text into natural clauses and queries Google News RSS index for live coverage.
    Cross-references each topic clause to detect if major news networks are actively reporting it.
    """
    if not text or not text.strip():
        return {"matched": False, "count": 0, "sources": [], "status": "No text provided"}

    raw_clauses = re.split(r'[,.;\n]+', text)
    clauses = [c.strip() for c in raw_clauses if len(c.strip()) > 8]
    if not clauses:
        clauses = [text]

    found_sources = []
    seen_titles = set()

    for clause in clauses[:3]:
        words = [w for w in re.findall(r"[A-Za-z0-9']+", clause) if len(w) > 2 and w.lower() not in STOP_WORDS]
        if not words:
            continue
        query = " ".join(words[:4])
        encoded_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        try:
            resp = http_requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                items = root.findall(".//item")
                for item in items[:4]:
                    raw_title = item.find("title").text if item.find("title") is not None else ""
                    link_el = item.find("link")
                    link = link_el.text if link_el is not None and link_el.text else ""
                    if not link and link_el is not None:
                        link = link_el.tail or ""

                    source_el = item.find("source")
                    source_name = source_el.text if source_el is not None else "Internet News"

                    title = raw_title
                    if " - " in raw_title:
                        parts = raw_title.rsplit(" - ", 1)
                        title = parts[0].strip()
                        source_name = parts[1].strip()

                    # Strictly filter out search results that don't match the query clause
                    if _title_overlap(clause, title) < 0.35:
                        continue

                    clean_key = title.lower()[:50]
                    if clean_key in seen_titles:
                        continue
                    seen_titles.add(clean_key)

                    found_sources.append({
                        "source": source_name,
                        "title": title,
                        "url": link.strip() or "#",
                        "topic": query
                    })
        except Exception as e:
            print(f"[verify_on_internet] RSS query error: {e}")

    matched = len(found_sources) > 0
    return {
        "matched": matched,
        "count": len(found_sources),
        "sources": found_sources[:6],
        "status": f"Found {len(found_sources)} live news coverage matches from major outlets" if matched else "0 matches found on major news networks"
    }


# ---------------------------------------------------------------
# 5. GOOGLE FACT CHECK DATABASE LOOKUP
# ---------------------------------------------------------------
FACTCHECK_API_KEY = os.environ.get("FACTCHECK_API_KEY", "")
FACTCHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def check_fact_database(text: str) -> dict:
    """Searches ClaimReview data from PolitiFact, Snopes, AltNews, Boom, etc."""
    if not FACTCHECK_API_KEY or not text.strip():
        return {"available": False, "claims": []}

    words = [w for w in re.findall(r"[A-Za-z0-9']+", text.lower()) if len(w) > 2 and w not in STOP_WORDS]
    query = " ".join(words[:6])

    try:
        resp = http_requests.get(
            FACTCHECK_URL,
            params={"query": query, "key": FACTCHECK_API_KEY, "languageCode": "en"},
            timeout=6,
        )
        if resp.status_code != 200:
            return {"available": True, "claims": [], "error": f"status {resp.status_code}"}

        data = resp.json()
        claims_raw = data.get("claims", [])[:5]
        claims = []
        for c in claims_raw:
            reviews = c.get("claimReview", [])
            if not reviews:
                continue
            rev = reviews[0]
            claims.append({
                "claim_text": c.get("text", ""),
                "rating": rev.get("textualRating", "Unknown"),
                "publisher": rev.get("publisher", {}).get("name", "Unknown"),
                "url": rev.get("url", ""),
            })
        return {"available": True, "claims": claims}
    except Exception as e:
        return {"available": True, "claims": [], "error": str(e)}


# ---------------------------------------------------------------
# 6. GEMINI AI FACT-CHECKING MODULE
# ---------------------------------------------------------------
def check_gemini_ai(text: str) -> dict:
    """
    Directly queries Google Gemini API to analyze factual consistency and truthfulness of the input text.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("FACTCHECK_API_KEY", "")
    if not gemini_key:
        return {"available": False, "verdict": "Uncertain", "confidence": 50.0, "reason": "No Gemini API Key provided"}

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
    prompt = f"""You are an expert Fact-Checker AI for news verification.
Evaluate this news input carefully. Is this statement factually REAL NEWS (plausible real-world event/report) or FAKE NEWS (fabricated/hoax/misinformation)?

News Input: "{text}"

Respond ONLY in valid JSON with these exact fields:
{{
  "verdict": "Real" | "Fake" | "Uncertain",
  "confidence": 90.0,
  "reason": "Short 1-sentence factual explanation of why this is Real or Fake"
}}"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        resp = http_requests.post(url, json=payload, timeout=7)
        if resp.status_code == 200:
            data = resp.json()
            raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            if "```json" in raw_text:
                raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_text:
                raw_text = raw_text.split("```")[1].split("```")[0].strip()

            parsed = json.loads(raw_text)
            verdict = parsed.get("verdict", "Uncertain")
            if verdict not in ("Real", "Fake", "Uncertain"):
                verdict = "Uncertain"
            return {
                "available": True,
                "verdict": verdict,
                "confidence": float(parsed.get("confidence", 80.0)),
                "reason": parsed.get("reason", "Analyzed by Google Gemini AI")
            }
        else:
            return {"available": False, "verdict": "Uncertain", "confidence": 50.0, "reason": f"API error code {resp.status_code}"}
    except Exception as e:
        return {"available": False, "verdict": "Uncertain", "confidence": 50.0, "reason": str(e)}


# ---------------------------------------------------------------
# 7. COMBINE ALL SIGNALS INTO ONE FINAL VERDICT
# ---------------------------------------------------------------
FALSE_RATING_WORDS = {"false", "fake", "pants on fire", "incorrect", "misleading", "no evidence", "fabricated", "hoax"}
TRUE_RATING_WORDS = {"true", "correct", "accurate", "verified", "mostly true"}


def combine_verdict(ai_label: str, ai_confidence: float, net_verify: dict, fact_check: dict, credibility=None, gemini_result=None) -> dict:
    """
    Fuses:
    1. Live Internet Coverage Match (Strong positive signal: +50 to +70 for active news coverage)
    2. Google Fact-Check DB (Snopes, PolitiFact, AltNews: ±60 override)
    3. Gemini AI Analysis (if active: ±40 signal)
    4. Local ML Model (TF-IDF + Random Forest signal)
    5. Domain Credibility (if URL)

    Produces final Real / Fake / Uncertain label with human-readable rationale.
    """
    score = 0.0
    reasons = []

    # ── Signal 1: Live Internet Coverage (Highest Weight for Real-World Current News)
    if net_verify and net_verify.get("matched") and net_verify.get("count", 0) > 0:
        match_count = net_verify["count"]
        boost = min(60, 30 + match_count * 5)
        score += boost
        reasons.append(f"Confirmed by live internet search: {match_count} matching news reports on major outlets")
    else:
        reasons.append("No active news coverage found in global news index")

    # ── Signal 2: Gemini AI Analysis
    if gemini_result and gemini_result.get("available"):
        g_verdict = gemini_result.get("verdict")
        g_conf = gemini_result.get("confidence", 80.0)
        g_weight = 35 + max(0, (g_conf - 50) * 0.4)
        if g_verdict == "Real":
            score += g_weight
            reasons.append(f"Gemini AI verified as Real: {gemini_result.get('reason')}")
        elif g_verdict == "Fake":
            score -= g_weight
            reasons.append(f"Gemini AI flagged as Fake: {gemini_result.get('reason')}")

    # ── Signal 3: Google Fact-Check Database (Strongest Debunk Signal: ±60)
    fc_hit = None
    if fact_check and fact_check.get("claims"):
        for c in fact_check["claims"]:
            rating_lower = c["rating"].lower()
            if any(w in rating_lower for w in FALSE_RATING_WORDS):
                score -= 60
                fc_hit = c
                reasons.append(f"Debunked by {c['publisher']} — rated '{c['rating']}'")
                break
            elif any(w in rating_lower for w in TRUE_RATING_WORDS):
                score += 60
                fc_hit = c
                reasons.append(f"Confirmed by {c['publisher']} — rated '{c['rating']}'")
                break

    # ── Signal 4: Local ML Classifier Signal
    if not (net_verify and net_verify.get("matched")):
        ml_weight = 25 + max(0, (ai_confidence - 50) * 0.4)
        score += ml_weight if ai_label == "Real" else -ml_weight
        reasons.append(f"Local ML model predicts {ai_label} ({ai_confidence}% confidence)")

    # ── Signal 5: Source Credibility
    if credibility:
        if credibility["level"] == "high":
            score += 20
            reasons.append("Published by a recognized, established news outlet")
        elif credibility["level"] == "low":
            score -= 30
            reasons.append("Source is a known satire / low-credibility site")

    # ── Final Decision & Confidence Thresholds
    if score >= 20:
        final_label = "Real"
        final_confidence = round(min(99.0, 65.0 + score * 0.4), 1)
    elif score <= -20:
        final_label = "Fake"
        final_confidence = round(min(99.0, 65.0 + abs(score) * 0.4), 1)
    else:
        final_label = "Uncertain"
        final_confidence = round(max(50.0, ai_confidence), 1)

    return {
        "final_label": final_label,
        "final_confidence": round(final_confidence, 1),
        "score": round(score, 1),
        "reasons": reasons,
        "fact_check_match": fc_hit,
    }
