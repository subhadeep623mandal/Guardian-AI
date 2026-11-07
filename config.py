"""
Deepfake Guardian AI - Configuration Module
Centralized configuration, security secrets, and model parameters.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if available
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

# ==============================================================================
# Security & Authorization Factor Settings
# ==============================================================================
MASTER_PASSKEY = os.getenv("MASTER_PASSKEY", "GUARDIAN-SEC-2026-SHIELD")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "guardian-ai-sec-key-3.8-enterprise")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "GuardianAdmin@2026!")

USERS_DB_FILE = BASE_DIR / "auth_users.json"
SESSION_EXPIRY_MINUTES = 60
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_MINUTES = 10

# ==============================================================================
# AI & Detection Core Settings
# ==============================================================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

# Preprocessing & Vision Specs
INPUT_IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# DBSCAN Anomaly Analysis Parameters
DBSCAN_EPS = float(os.getenv("DBSCAN_EPS", "0.45"))
DBSCAN_MIN_SAMPLES = int(os.getenv("DBSCAN_MIN_SAMPLES", "5"))

# Assets & Storage
ASSETS_DIR = BASE_DIR / "assets"
SAMPLES_DIR = ASSETS_DIR / "samples"
ARCHITECTURE_IMAGE_PATH = ASSETS_DIR / "architecture.jpg"
