import pickle
from pathlib import Path
from typing import Any

import streamlit as st
import tensorflow as tf

from src import config
from src.utils import seq_pad_and_trunc


@st.cache_resource
def _load_model(model_path: str) -> tf.keras.Model:
    """Charge le modèle TensorFlow depuis le disque."""
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def _load_tokenizer(tokenizer_path: str) -> Any:
    """Charge le tokenizer sérialisé."""
    with open(tokenizer_path, "rb") as handle:
        return pickle.load(handle)


class SentimentModel:
    """Service chargé de la prédiction de sentiment."""

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        tokenizer_path: Path = config.TOKENIZER_PATH,
        max_sequence_length: int = 100,
    ) -> None:
        self.model_path = str(model_path)
        self.tokenizer_path = str(tokenizer_path)
        self.max_sequence_length = max_sequence_length
        self.model = _load_model(self.model_path)
        self.tokenizer = _load_tokenizer(self.tokenizer_path)

    def predict(self, sentence: str):
        """Retourne la prédiction brute du modèle pour un texte donné."""
        processed_input = seq_pad_and_trunc(
            sentence,
            self.tokenizer,
            maxlen=self.max_sequence_length,
        )
        return self.model.predict(processed_input)

