"""
mailer.py — Google SMTP Email Verification & OTP Service
=========================================================
Handles secure 6-digit OTP generation, SMTP TLS connection to Google (smtp.gmail.com:587),
and dark-themed cyber security emails for TruthLine user authentication.
"""

import os
import time
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# In-memory OTP storage: { email.lower(): { "code": "123456", "expires_at": float, "attempts": int } }
_OTP_CACHE = {}


def get_smtp_config():
    """Retrieve Google SMTP configuration from environment."""
    user = os.environ.get("GMAIL_SMTP_USER", "").strip()
    app_pw = os.environ.get("GMAIL_SMTP_APP_PASSWORD", "").strip()
    # Normalize app password (remove spaces)
    app_pw_clean = app_pw.replace(" ", "")
    host = os.environ.get("GMAIL_SMTP_HOST", "smtp.gmail.com").strip()
    port = int(os.environ.get("GMAIL_SMTP_PORT", "587").strip() or 587)
    
    is_configured = bool(user and app_pw_clean)
    return {
        "user": user,
        "app_password": app_pw_clean,
        "host": host,
        "port": port,
        "configured": is_configured,
        "masked_user": (user[:3] + "***@" + user.split("@")[-1]) if "@" in user else ("***" if user else "")
    }


def save_smtp_config(gmail_user, gmail_app_password):
    """Save Google SMTP credentials to app/.env and os.environ."""
    gmail_user = gmail_user.strip()
    gmail_app_password = gmail_app_password.strip().replace(" ", "")
    
    os.environ["GMAIL_SMTP_USER"] = gmail_user
    os.environ["GMAIL_SMTP_APP_PASSWORD"] = gmail_app_password

    env_path = os.path.join(os.path.dirname(__file__), ".env")
    lines = []
    found_user = False
    found_pw = False

    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("GMAIL_SMTP_USER="):
                    lines.append(f"GMAIL_SMTP_USER={gmail_user}\n")
                    found_user = True
                elif line.startswith("GMAIL_SMTP_APP_PASSWORD="):
                    lines.append(f"GMAIL_SMTP_APP_PASSWORD={gmail_app_password}\n")
                    found_pw = True
                else:
                    lines.append(line)

    if not found_user:
        lines.append(f"GMAIL_SMTP_USER={gmail_user}\n")
    if not found_pw:
        lines.append(f"GMAIL_SMTP_APP_PASSWORD={gmail_app_password}\n")

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    return True


def generate_otp(email, expires_seconds=600):
    """Generate a cryptographic 6-digit OTP code and store it with expiration."""
    code = f"{secrets.randbelow(900000) + 100000}"
    email_key = email.strip().lower()
    _OTP_CACHE[email_key] = {
        "code": code,
        "expires_at": time.time() + expires_seconds,
        "attempts": 0
    }
    return code


def verify_otp_code(email, input_code):
    """Validate user entered OTP code."""
    email_key = email.strip().lower()
    record = _OTP_CACHE.get(email_key)
    if not record:
        return False, "No active verification code found for this email. Please request a new one."
    
    if time.time() > record["expires_at"]:
        _OTP_CACHE.pop(email_key, None)
        return False, "Verification code has expired (valid for 10 minutes). Please request a new one."
    
    record["attempts"] += 1
    if record["attempts"] > 5:
        _OTP_CACHE.pop(email_key, None)
        return False, "Too many invalid attempts. Please request a new code."
    
    if record["code"] != input_code.strip():
        return False, "Incorrect verification code. Please check your email and try again."
    
    # Valid code, clear cache entry
    _OTP_CACHE.pop(email_key, None)
    return True, "Verification successful."


def send_verification_email(to_email, otp_code, username=None):
    """
    Connect to Google SMTP (smtp.gmail.com:587) via TLS and send cyber verification email.
    Returns (success: bool, message: str).
    """
    config = get_smtp_config()
    if not config["configured"]:
        return False, "Google SMTP credentials are not configured on the server yet."

    sender_email = config["user"]
    app_pw = config["app_password"]
    host = config["host"]
    port = config["port"]

    recipient_name = username or to_email.split("@")[0].capitalize()

    # Create MIMEMultipart message
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Your TruthLine Security Verification Code: {otp_code}"
    msg["From"] = f"TruthLine Security <{sender_email}>"
    msg["To"] = to_email

    # Plain-text version
    text_content = f"""TruthLine AI Security Verification
----------------------------------------
Hello {recipient_name},

Your one-time authentication passcode is: {otp_code}

This code is valid for 10 minutes.
If you did not request this code, please disregard this transmission.

TruthLine — Truth Powers a Brighter Tomorrow.
"""

    # Futuristic Cyber HTML version
    html_content = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>TruthLine Security Verification</title>
  <style>
    body {{
      font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
      background-color: #050813;
      color: #e2e8f0;
      margin: 0;
      padding: 24px;
    }}
    .container {{
      max-width: 540px;
      margin: 0 auto;
      background: linear-gradient(135deg, #0b1120 0%, #060b18 100%);
      border: 1px solid rgba(56, 189, 248, 0.25);
      border-radius: 16px;
      padding: 32px;
      box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6), 0 0 30px rgba(6, 182, 212, 0.15);
    }}
    .header {{
      display: flex;
      align-items: center;
      margin-bottom: 24px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      padding-bottom: 16px;
    }}
    .logo-text {{
      font-size: 22px;
      font-weight: 800;
      letter-spacing: 0.5px;
      color: #ffffff;
    }}
    .logo-accent {{
      color: #38bdf8;
    }}
    .tagline {{
      font-size: 11px;
      color: #94a3b8;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-top: 4px;
    }}
    .otp-box {{
      background: rgba(14, 165, 233, 0.08);
      border: 1px dashed #38bdf8;
      border-radius: 12px;
      padding: 24px;
      text-align: center;
      margin: 28px 0;
    }}
    .otp-code {{
      font-family: 'Courier New', Courier, monospace;
      font-size: 38px;
      font-weight: 900;
      letter-spacing: 10px;
      color: #38bdf8;
      text-shadow: 0 0 15px rgba(56, 189, 248, 0.5);
      display: inline-block;
      padding: 6px 16px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      background: rgba(16, 185, 129, 0.15);
      border: 1px solid rgba(16, 185, 129, 0.4);
      color: #34d399;
      font-size: 11px;
      font-weight: 700;
      border-radius: 9999px;
      margin-bottom: 12px;
    }}
    .footer {{
      margin-top: 32px;
      border-top: 1px solid rgba(255, 255, 255, 0.08);
      padding-top: 16px;
      font-size: 11px;
      color: #64748b;
      line-height: 1.6;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div class="logo-text">Truth<span class="logo-accent">Line</span></div>
        <div class="tagline">Global AI Misinformation & Authenticity Intelligence</div>
      </div>
    </div>

    <div>
      <span class="badge">SECURE TRANSMISSION</span>
      <h2 style="margin: 6px 0 12px; font-size: 20px; color: #ffffff;">Authentication Passcode</h2>
      <p style="font-size: 14px; color: #cbd5e1; line-height: 1.6;">
        Greetings <strong>{recipient_name}</strong>,<br>
        A request has been initiated to authenticate your access to the <strong>TruthLine Intelligence Portal</strong>. Use the secure authorization code below to complete sign-in:
      </p>

      <div class="otp-box">
        <div style="font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px;">Single-Use Security Code</div>
        <div class="otp-code">{otp_code}</div>
        <div style="font-size: 12px; color: #94a3b8; margin-top: 10px;">Valid for <strong>10 minutes</strong>. Never share this code.</div>
      </div>

      <p style="font-size: 12px; color: #94a3b8; line-height: 1.5;">
        If you did not initiate this authentication request, no action is required. Your account remains protected by quantum-safe encryption.
      </p>
    </div>

    <div class="footer">
      <div><strong>TruthLine AI Defense Network</strong> &bull; Powered by Google Gemini &amp; Live Wire Intelligence</div>
      <div>Security Gateway &bull; Session IP Logged &bull; End-to-End Encrypted</div>
    </div>
  </div>
</body>
</html>"""

    msg.attach(MIMEText(text_content, "plain"))
    msg.attach(MIMEText(html_content, "html"))

    try:
        # Connect to Google SMTP server
        server = smtplib.SMTP(host, port, timeout=15)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(sender_email, app_pw)
        server.sendmail(sender_email, [to_email], msg.as_string())
        server.quit()
        return True, f"Security verification code dispatched to {to_email} via Google SMTP."
    except smtplib.SMTPAuthenticationError as e:
        return False, f"Google SMTP Authentication Failed: Invalid Google App Password or Username ({e}). Make sure you are using a 16-character Google App Password (not standard password)."
    except Exception as e:
        return False, f"Google SMTP Delivery Error: {str(e)}"
