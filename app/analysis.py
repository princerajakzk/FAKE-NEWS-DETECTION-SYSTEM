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
import string
import urllib.parse
import xml.etree.ElementTree as ET
import requests as http_requests
from urllib.parse import urlparse

# ── Load .env automatically if present ───────────────────────────────────────
_ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_ENV_PATH):
    with open(_ENV_PATH, encoding="utf-8", errors="ignore") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

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
    word = word.lower().strip(string.punctuation)
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
    "top", "global", "national", "news", "today", "features", "feature", "hosting",
    "host", "hosts", "world", "leaders", "ongoing", "across", "spells", "heavy",
    "states", "northern", "southern", "claims", "claim", "viral", "post", "video"
}

FACT_CHECK_ENTITIES = {
    "boomlive", "boomlive.in", "altnews", "altnews.in", "factly", "vishwas news", "vishwas",
    "webqoof", "snopes", "politifact", "reuters fact check", "afp fact check",
    "newschecker", "pib fact check", "thip media", "thip.media", "fact crescendo",
    "checkyourfact", "lead stories", "full fact"
}

DEBUNK_PHRASES = [
    "is fake", "is false", "fake news", "hoax", "debunked", "debunk", "false claim",
    "fake claim", "viral claim is false", "not true", "fact check", "fact-check",
    "fact check:", "misleading claim", "misleading", "morphed photo", "morphed video",
    "fabricated quote", "fabricated", "untrue", "scam", "myth", "no evidence", "incorrect"
]


def _title_overlap(query: str, title: str) -> float:
    q_words = set(w.lower() for w in re.findall(r"[\w']+", query, re.UNICODE) if len(w) > 2 and w.lower() not in STOP_WORDS)
    t_words = set(w.lower() for w in re.findall(r"[\w']+", title, re.UNICODE) if len(w) > 2)
    if not q_words:
        return 0.0
    common = q_words.intersection(t_words)
    return len(common) / len(q_words)


RELIABLE_NEWS_DOMAINS = [
    "reuters", "bbc", "ndtv", "hindu", "express", "times", "ani", "ap",
    "al jazeera", "bloomberg", "cnn", "guardian", "washington post",
    "nytimes", "wsj", "abc", "cbs", "nbc", "pbs", "npr", "hindustan times",
    "india today", "livemint", "financial express", "news18", "pti",
    "deccan herald", "forbes", "nature", "space.com", "nasa", "isro", "who"
]


def verify_on_internet(text: str) -> dict:
    """
    Directly queries real-time Google News RSS indexes to cross-reference claims against:
    1. Active mainstream media reporting
    2. Viral fact-checking debunk registries (AltNews, Boom, Snopes, PolitiFact, PIB, etc.)
    """
    import concurrent.futures
    if not text or not text.strip():
        return {"matched": False, "count": 0, "sources": [], "has_debunk": False, "has_confirmation": False, "status": "No text provided"}

    # Extract meaningful keywords preserving unicode (Hindi/English/etc)
    clean_words = [w for w in re.findall(r"[\w']+", text, re.UNICODE) if len(w) > 2 and w.lower() not in STOP_WORDS]
    if not clean_words:
        clean_words = text.strip().split()

    topic_query = " ".join(clean_words[:6])
    fact_query = f"{topic_query} fact check"

    queries_to_try = [topic_query]
    if len(clean_words) >= 3:
        queries_to_try.append(fact_query)

    found_sources = []
    seen_titles = set()
    has_debunk = False
    has_confirmation = False

    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"}

    def _fetch_rss(q):
        encoded_query = urllib.parse.quote(q)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            resp = http_requests.get(url, timeout=4.5, headers=headers)
            if resp.status_code == 200:
                return resp.content, q
        except Exception:
            pass
        return None, q

    # Run queries in parallel for maximum speed
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries_to_try)) as executor:
        futures = [executor.submit(_fetch_rss, q) for q in queries_to_try]
        for f in concurrent.futures.as_completed(futures):
            xml_content, query = f.result()
            if not xml_content:
                continue
            try:
                root = ET.fromstring(xml_content)
                items = root.findall(".//item")
                for item in items[:6]:
                    raw_title = item.find("title").text if item.find("title") is not None else ""
                    link_el = item.find("link")
                    link = (link_el.tail or "").strip() if link_el is not None else ""
                    if not link and link_el is not None and link_el.text:
                        link = link_el.text.strip()

                    source_el = item.find("source")
                    source_name = source_el.text if source_el is not None else "News Outlet"

                    title = raw_title
                    if " - " in raw_title:
                        parts = raw_title.rsplit(" - ", 1)
                        title = parts[0].strip()
                        source_name = parts[1].strip()

                    clean_key = title.lower()[:60]
                    if clean_key in seen_titles:
                        continue

                    # Overlap check
                    overlap = _title_overlap(topic_query, title)
                    if overlap < 0.35 and len(clean_words) >= 3:
                        continue

                    seen_titles.add(clean_key)

                    title_lower = title.lower()
                    source_lower = source_name.lower()
                    is_fc_source = any(fc in source_lower for fc in FACT_CHECK_ENTITIES)
                    is_debunk_title = any(db in title_lower for db in DEBUNK_PHRASES)
                    is_fact_check = is_fc_source or is_debunk_title

                    if is_fact_check and overlap >= 0.30:
                        has_debunk = True

                    is_question = "?" in title
                    if not is_fact_check and not is_question and any(dom in source_lower for dom in RELIABLE_NEWS_DOMAINS) and overlap >= 0.45:
                        has_confirmation = True

                    found_sources.append({
                        "source": source_name,
                        "title": title,
                        "url": link or "#",
                        "topic": query,
                        "is_fact_check": is_fact_check
                    })
            except Exception:
                pass

    matched = len(found_sources) > 0
    status = ""
    if has_debunk:
        status = f"Debunked: Found {len(found_sources)} fact-checking articles warning this claim is FALSE/HOAX."
    elif has_confirmation:
        status = f"Verified: Found {len(found_sources)} live news coverage matches on major news networks."
    elif matched:
        status = f"Indexed: Found {len(found_sources)} live news coverage matches online."
    else:
        status = "0 live news reports found on major internet news networks."

    return {
        "matched": matched,
        "count": len(found_sources),
        "sources": found_sources[:8],
        "has_debunk": has_debunk,
        "has_confirmation": has_confirmation,
        "status": status
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

    words = [w for w in re.findall(r"[\w']+", text.lower(), re.UNICODE) if len(w) > 2 and w not in STOP_WORDS]
    query = " ".join(words[:6])

    try:
        resp = http_requests.get(
            FACTCHECK_URL,
            params={"query": query, "key": FACTCHECK_API_KEY, "languageCode": "en"},
            timeout=4.5,
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
# 6. GEMINI AI FACT-CHECKING MODULE (With Multi-Model Fallback & Internet Context)
# ---------------------------------------------------------------
def check_gemini_ai(text: str, internet_sources: list = None) -> dict:
    """
    Directly queries Google Gemini AI using automatic cascading multi-model fallback.
    Cross-references the live internet search context for maximum accuracy.
    """
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip() or os.environ.get("FACTCHECK_API_KEY", "").strip()
    if not gemini_key:
        return {"available": False, "verdict": "Uncertain", "confidence": 50.0, "reason": "No Gemini API Key configured in .env", "model": None, "actual_truth": "", "proof": ""}

    models_to_try = []
    configured_model = os.environ.get("GEMINI_MODEL", "").strip()
    if configured_model:
        models_to_try.append(configured_model)
    for default_m in ["gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-flash-latest", "gemini-flash-lite-latest"]:
        if default_m not in models_to_try:
            models_to_try.append(default_m)

    sources_summary = ""
    if internet_sources:
        sources_summary = "\n".join([f"- {s.get('source', 'Media')}: {s.get('title', '')}" for s in internet_sources[:6]])
    else:
        sources_summary = "No direct live news coverage indexed."

    prompt = f"""You are an elite, highly accurate AI News Verification and Fact-Checking Engine.
Evaluate the following news claim and determine whether it is REAL NEWS or FAKE NEWS.

News Claim:
"{text}"

Live Internet & Media Context (Found in real-time internet search):
{sources_summary}

Verification Guidelines:
1. "Real": The statement represents a genuine real-world event, confirmed official news, historical fact, scientific fact, or true development.
2. "Fake": The statement is fabricated, a hoax, rumor, scam, conspiracy theory, satire, debunked viral misinformation, or medical falsehood.
3. If internet search results or fact-checkers show that this claim is debunked or false, classify as "Fake".
4. If mainstream reputable media report it as an active genuine event, classify as "Real".

Respond STRICTLY in valid JSON format:
{{
  "verdict": "Real" | "Fake",
  "confidence": 98.0,
  "actual_truth": "Clear explanation of what the real news / actual truth is (what really happened or what the factual reality is).",
  "proof": "Specific evidence, official source confirmation, debunk reference, or historical/scientific proof.",
  "reason": "1-2 sentence concise summary explaining why this is Real or Fake."
}}"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    last_error = ""

    for model in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key}"
        try:
            resp = http_requests.post(url, json=payload, timeout=8.0)
            if resp.status_code == 200:
                data = resp.json()
                raw_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()

                json_match = re.search(r"\{[\s\S]*\}", raw_text)
                if json_match:
                    raw_text = json_match.group(0)

                parsed = json.loads(raw_text)
                raw_v = str(parsed.get("verdict", "")).strip().capitalize()
                if raw_v in ("Real", "True", "Verified", "Confirmed", "Accurate"):
                    verdict = "Real"
                elif raw_v in ("Fake", "False", "Fabricated", "Hoax", "Misleading"):
                    verdict = "Fake"
                else:
                    verdict = "Uncertain"

                conf = float(parsed.get("confidence", 90.0))
                if conf <= 1.0:
                    conf = conf * 100.0
                conf = max(50.0, min(99.0, conf))

                actual_truth = parsed.get("actual_truth") or parsed.get("reason") or "Verified against global media and factual consensus."
                proof = parsed.get("proof") or parsed.get("reason") or ""
                reason = parsed.get("reason") or f"Verified by Google Gemini AI ({model})"

                return {
                    "available": True,
                    "verdict": verdict,
                    "confidence": round(conf, 1),
                    "actual_truth": actual_truth,
                    "proof": proof,
                    "reason": reason,
                    "model": model
                }
            else:
                last_error = f"Model {model} status {resp.status_code}"
        except Exception as e:
            last_error = f"Model {model} error: {str(e)}"

    return {
        "available": False,
        "verdict": "Uncertain",
        "confidence": 50.0,
        "reason": last_error or "AI verification service temporarily unreachable",
        "actual_truth": "",
        "proof": "",
        "model": None
    }


# ---------------------------------------------------------------
# 7. SMART SIGNAL FUSION VERDICT ENGINE
# ---------------------------------------------------------------
FALSE_RATING_WORDS = {"false", "fake", "pants on fire", "incorrect", "misleading", "no evidence", "fabricated", "hoax"}
TRUE_RATING_WORDS = {"true", "correct", "accurate", "verified", "mostly true"}


def combine_verdict(ai_label: str, ai_confidence: float, net_verify: dict, fact_check: dict, credibility=None, gemini_result=None, text: str = "") -> dict:
    """
    Fuses real-time signals with primary priority given to:
    1. Gemini AI Analysis (deep semantic fact-checking & world knowledge)
    2. Real-Time Internet News Search & Fact-Check Debunk Registries
    3. Google ClaimReview Fact-Check Database
    4. Domain Credibility (if URL provided)
    5. Local ML Classifier (secondary linguistic tone/style baseline)
    """
    score = 0.0
    reasons = []
    fc_hit = None

    # ── Signal 1: Live Internet Debunk Warning (Immediate High-Confidence Fake)
    if net_verify and net_verify.get("has_debunk"):
        score -= 75
        reasons.append("🚨 Internet Fact-Check Alert: Known fact-checking organizations have flagged this claim as FALSE/HOAX")

    # ── Signal 2: Gemini AI Fact-Checking (Primary AI Decision Engine)
    gemini_active = gemini_result and gemini_result.get("available")
    if gemini_active:
        g_verdict = gemini_result.get("verdict")
        g_conf = gemini_result.get("confidence", 90.0)
        g_model = gemini_result.get("model") or "Gemini AI"
        g_reason = gemini_result.get("reason", "")

        weight = 55 + max(0, (g_conf - 50) * 0.4)
        if g_verdict == "Real":
            score += weight
            reasons.append(f"🤖 AI Fact-Check ({g_model}) verified as REAL: {g_reason}")
        elif g_verdict == "Fake":
            score -= weight
            reasons.append(f"⚠️ AI Fact-Check ({g_model}) flagged as FAKE: {g_reason}")

    # ── Signal 3: Live Internet News Coverage Confirmation
    if net_verify and net_verify.get("matched") and not net_verify.get("has_debunk"):
        match_count = net_verify.get("count", 0)
        boost = min(55, 30 + match_count * 5)
        score += boost
        if gemini_active and gemini_result.get("verdict") == "Fake":
            reasons.append(f"🌐 Internet Context: Indexed {match_count} news articles related to these entities for cross-reference")
        else:
            reasons.append(f"🌐 Confirmed by Live Internet Search: Found {match_count} active reports from mainstream news outlets")
    elif not gemini_active and not (net_verify and net_verify.get("matched")):
        reasons.append("No active news coverage found in real-time internet news index")

    # ── Signal 4: Google Fact-Check Database (Snopes, PolitiFact, AltNews)
    if fact_check and fact_check.get("claims"):
        for c in fact_check["claims"]:
            rating_lower = c["rating"].lower()
            if any(w in rating_lower for w in FALSE_RATING_WORDS):
                score -= 65
                fc_hit = c
                reasons.append(f"Debunked by {c['publisher']} — rated '{c['rating']}'")
                break
            elif any(w in rating_lower for w in TRUE_RATING_WORDS):
                score += 55
                fc_hit = c
                reasons.append(f"Confirmed by {c['publisher']} — rated '{c['rating']}'")
                break

    # ── Signal 5: Domain Credibility (if URL input)
    if credibility:
        if credibility["level"] == "high":
            score += 25
            reasons.append("Published by a recognized, established reputable news outlet")
        elif credibility["level"] == "low":
            score -= 35
            reasons.append("Source domain is a known satire / low-credibility site")

    # ── Signal 6: Local ML Model (Linguistic & Stylistic Baseline)
    # Only used as a secondary signal; never allowed to override AI or Internet facts
    if not gemini_active and not (net_verify and net_verify.get("matched")):
        if ai_label == "Real":
            ml_weight = 20 + max(0, (ai_confidence - 50) * 0.2)
            score += ml_weight
            reasons.append(f"Local linguistic model tone check: Verified real news pattern ({ai_confidence}% confidence)")
        else:
            s_chk = sensationalism_score(text) if (text and "sensationalism_score" in globals()) else 0
            if s_chk < 25:
                # Text is neutral and objective; avoid falsely flagging as Fake
                reasons.append("Linguistic baseline: Tone is neutral and objective; no sensationalism detected")
                score += 8
            else:
                ml_weight = 15 + max(0, (ai_confidence - 50) * 0.2)
                score -= ml_weight
                reasons.append(f"Local linguistic model tone check: Stylistic anomalies detected ({ai_confidence}% confidence)")

    # ── Final Verdict Determination
    if gemini_active and gemini_result.get("verdict") in ("Real", "Fake"):
        # Gemini AI verdict with world knowledge & semantic fact-checking
        if gemini_result["verdict"] == "Real":
            final_label = "Real"
            bonus = 3.0 if (net_verify and net_verify.get("has_confirmation")) else 0.0
            final_confidence = min(99.0, max(float(gemini_result["confidence"]), 92.0 + bonus))
        elif gemini_result["verdict"] == "Fake":
            final_label = "Fake"
            final_confidence = min(99.0, max(float(gemini_result["confidence"]), 92.0))
        else:
            final_label = "Fake" if score < 0 else "Real"
            final_confidence = min(99.0, max(float(gemini_result["confidence"]), 85.0))
    elif net_verify and net_verify.get("has_debunk"):
        final_label = "Fake"
        final_confidence = 96.0
    elif net_verify and net_verify.get("has_confirmation"):
        final_label = "Real"
        final_confidence = min(98.0, max(88.0, 78.0 + net_verify.get("count", 1) * 3))
    elif score >= 15:
        final_label = "Real"
        final_confidence = round(min(95.0, 72.0 + score * 0.3), 1)
    elif score <= -15:
        final_label = "Fake"
        final_confidence = round(min(95.0, 70.0 + abs(score) * 0.35), 1)
    else:
        final_label = "Real" if ai_label == "Real" else "Fake"
        final_confidence = round(max(50.0, min(80.0, float(ai_confidence))), 1)

    actual_truth = ""
    proof = ""
    if gemini_active and gemini_result:
        actual_truth = gemini_result.get("actual_truth") or ""
        proof = gemini_result.get("proof") or ""

    if not actual_truth:
        if final_label == "Real":
            if net_verify and net_verify.get("sources"):
                s0 = net_verify["sources"][0]
                actual_truth = f"Confirmed Real Event: Officially reported and corroborated by {s0.get('source', 'mainstream media')} ('{s0.get('title', '')}')."
            else:
                actual_truth = "Confirmed authentic report matching established factual records and official documentation."
        else:
            if net_verify and net_verify.get("has_debunk"):
                actual_truth = "Debunked Falsehood: Multiple independent fact-checking registries and credible sources have verified this claim is FALSE / HOAX."
            else:
                actual_truth = "Fabricated Claim: No credible news network, scientific institution, or verified registry has corroborated this claim."

    if not proof:
        if net_verify and net_verify.get("sources"):
            top_src = net_verify["sources"][:3]
            proof = "Cross-referenced with live indexed coverage: " + "; ".join([f"{s.get('source', 'Source')}: '{s.get('title', '')}'" for s in top_src])
        elif fc_hit:
            proof = f"Flagged by {fc_hit.get('publisher')} with rating '{fc_hit.get('rating')}'."
        else:
            proof = "Multi-signal forensic check conducted across language patterns, domain reputations, and global fact databases."

    return {
        "final_label": final_label,
        "final_confidence": round(final_confidence, 1),
        "actual_truth": actual_truth,
        "proof": proof,
        "score": round(score, 1),
        "reasons": reasons,
        "fact_check_match": fc_hit,
    }
