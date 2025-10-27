from pathlib import Path

# Répertoires principaux
BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "templates" / "assets"
IMAGES_DIR = ASSETS_DIR / "images"

# Fichiers du modèle
MODEL_PATH = ASSETS_DIR / "AnalyseSentiment.h5"
TOKENIZER_PATH = ASSETS_DIR / "tokenizer.pkl"

# Configuration multilingue
LANGUAGES = {
    "fr": "🇫🇷 Français",
    "en": "🇬🇧 English",
    "es": "🇪🇸 Español",
    "de": "🇩🇪 Deutsch",
    "it": "🇮🇹 Italiano",
    "pt": "🇵🇹 Português",
    "ja": "🇯🇵 日本語",
    "zh-CN": "🇨🇳 中文",
    "ar": "🇸🇦 العربية",
    "ru": "🇷🇺 Русский",
}

PAGE_CONFIG = {
    "page_title": "Analyse de Sentiments",
    "page_icon": "😊",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

APP_LINK = "https://gabriel.mariebrisson.fr"

CUSTOM_CSS = """
<style>
    /* Fond général */
    .reportview-container {
        background-color: #F4F6F9;
        font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
    }

    /* Titres */
    h1, h2, h3 {
        color: #2C3E50;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(45deg, #3498db, #2ecc71);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Amélioration des en-têtes */
    .css-1h7aky3 {
        background-color: rgba(44, 62, 80, 0.05);
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
    }

    /* Boutons */
    .stButton>button {
        color: white;
        background-color: #3498db;
        border: none;
        border-radius: 25px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }
</style>
"""

