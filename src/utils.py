import logging
from typing import Any

import streamlit as st
from deep_translator import GoogleTranslator
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import (
    DEFAULT_LANGUAGE,
    DEFAULT_MAX_SEQUENCE_LENGTH,
    DEFAULT_PADDING,
    DEFAULT_TRUNCATING,
    TRANSLATION_SOURCE_LANGUAGE,
)

logger = logging.getLogger(__name__)


def init_translation_state(default_language: str = DEFAULT_LANGUAGE) -> str:
    """Initialise la langue et le cache de traduction dans la session."""
    if "language" not in st.session_state:
        st.session_state.language = default_language
    if "translations_cache" not in st.session_state:
        st.session_state.translations_cache = {}
    return str(st.session_state.language)


def set_language(language_code: str) -> None:
    """Met à jour la langue courante dans la session."""
    st.session_state.language = language_code


def translate(text: str, target_language: str) -> str:
    """Traduit un texte depuis la langue source configurée en utilisant un cache simple."""
    if target_language == DEFAULT_LANGUAGE:
        return text

    cache_key = f"{target_language}_{text}"
    cache: dict[str, str] = st.session_state.translations_cache

    if cache_key in cache:
        return cache[cache_key]

    try:
        translated: str = GoogleTranslator(
            source=TRANSLATION_SOURCE_LANGUAGE,
            target=target_language,
        ).translate(text)
        cache[cache_key] = translated
        return translated
    except Exception as exc:  # noqa: BLE001
        logger.error("Erreur lors de la traduction '%s' vers '%s': %s", text, target_language, exc)
        return text


def seq_pad_and_trunc(
    sentence: str,
    tokenizer: Any,
    padding: str = DEFAULT_PADDING,
    truncating: str = DEFAULT_TRUNCATING,
    maxlen: int = DEFAULT_MAX_SEQUENCE_LENGTH,
) -> Any:
    """Convertit un texte en séquence pad/tronquée compatible avec le modèle."""
    sequences = tokenizer.texts_to_sequences([sentence])
    return pad_sequences(
        sequences,
        maxlen=maxlen,
        padding=padding,
        truncating=truncating,
    )

