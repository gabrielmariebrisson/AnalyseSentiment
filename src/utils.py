import streamlit as st
from deep_translator import GoogleTranslator
from tensorflow.keras.preprocessing.sequence import pad_sequences


def init_translation_state(default_language: str = "fr") -> str:
    """Initialise la langue et le cache de traduction dans la session."""
    if "language" not in st.session_state:
        st.session_state.language = default_language
    if "translations_cache" not in st.session_state:
        st.session_state.translations_cache = {}
    return st.session_state.language


def set_language(language_code: str) -> None:
    """Met à jour la langue courante dans la session."""
    st.session_state.language = language_code


def translate(text: str, target_language: str) -> str:
    """Traduit un texte depuis le français en utilisant un cache simple."""
    if target_language == "fr":
        return text

    cache_key = f"{target_language}_{text}"
    cache = st.session_state.translations_cache

    if cache_key in cache:
        return cache[cache_key]

    try:
        translated = GoogleTranslator(source="fr", target=target_language).translate(text)
        cache[cache_key] = translated
        return translated
    except Exception:
        return text


def seq_pad_and_trunc(
    sentence: str,
    tokenizer,
    padding: str = "post",
    truncating: str = "post",
    maxlen: int = 100,
):
    """Convertit un texte en séquence pad/tronquée compatible avec le modèle."""
    sequences = tokenizer.texts_to_sequences([sentence])
    return pad_sequences(
        sequences,
        maxlen=maxlen,
        padding=padding,
        truncating=truncating,
    )

