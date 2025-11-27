import pickle
from pathlib import Path
from typing import Any

import streamlit as st
import tensorflow as tf

from src import config
from src.utils import seq_pad_and_trunc


@st.cache_resource
def _load_model(model_path: str) -> tf.keras.Model:
    """Charge le modèle TensorFlow depuis le disque avec mise en cache.

    Le modèle est mis en cache via le décorateur @st.cache_resource pour éviter
    de le recharger à chaque interaction utilisateur dans Streamlit.

    Args:
        model_path: Chemin vers le fichier .h5 contenant le modèle sauvegardé.

    Returns:
        Modèle TensorFlow/Keras chargé et prêt à l'emploi.

    Raises:
        OSError: Si le fichier modèle n'existe pas ou n'est pas accessible.
        ValueError: Si le fichier modèle est corrompu ou incompatible.
    """
    return tf.keras.models.load_model(model_path)


@st.cache_resource
def _load_tokenizer(tokenizer_path: str) -> Any:
    """Charge le tokenizer sérialisé depuis un fichier pickle.

    Le tokenizer est mis en cache via le décorateur @st.cache_resource pour éviter
    de le recharger à chaque interaction utilisateur dans Streamlit.

    Args:
        tokenizer_path: Chemin vers le fichier .pkl contenant le tokenizer sérialisé.

    Returns:
        Tokenizer chargé (généralement un Tokenizer de Keras).

    Raises:
        FileNotFoundError: Si le fichier tokenizer n'existe pas.
        pickle.UnpicklingError: Si le fichier pickle est corrompu.
    """
    with open(tokenizer_path, "rb") as handle:
        return pickle.load(handle)


class SentimentModel:
    """Service chargé de la prédiction de sentiment sur des textes.

    Cette classe encapsule le chargement et l'utilisation d'un modèle de classification
    de sentiment basé sur des embeddings GloVe et une architecture CNN. Elle gère
    automatiquement le prétraitement des textes (tokenization, padding, troncature)
    avant de les passer au modèle.

    Attributes:
        model_path: Chemin vers le fichier du modèle TensorFlow.
        tokenizer_path: Chemin vers le fichier du tokenizer.
        max_sequence_length: Longueur maximale des séquences après padding/troncature.
        model: Modèle TensorFlow/Keras chargé.
        tokenizer: Tokenizer chargé pour la conversion texte -> séquences.
    """

    def __init__(
        self,
        model_path: Path = config.MODEL_PATH,
        tokenizer_path: Path = config.TOKENIZER_PATH,
        max_sequence_length: int = config.DEFAULT_MAX_SEQUENCE_LENGTH,
    ) -> None:
        """Initialise le service de prédiction de sentiment.

        Charge le modèle et le tokenizer depuis les chemins spécifiés. Les chemins
        par défaut sont définis dans src.config.

        Args:
            model_path: Chemin vers le fichier .h5 du modèle. Par défaut, utilise
                config.MODEL_PATH.
            tokenizer_path: Chemin vers le fichier .pkl du tokenizer. Par défaut,
                utilise config.TOKENIZER_PATH.
            max_sequence_length: Longueur maximale des séquences pour le padding/troncature.
                Par défaut, utilise config.DEFAULT_MAX_SEQUENCE_LENGTH (100).

        Raises:
            OSError: Si les fichiers modèle ou tokenizer ne peuvent pas être chargés.
            ValueError: Si les fichiers sont corrompus ou incompatibles.
        """
        self.model_path: str = str(model_path)
        self.tokenizer_path: str = str(tokenizer_path)
        self.max_sequence_length: int = max_sequence_length
        self.model: tf.keras.Model = _load_model(self.model_path)
        self.tokenizer: Any = _load_tokenizer(self.tokenizer_path)

    def predict(self, sentence: str) -> Any:
        """Prédit le sentiment d'un texte donné.

        Le texte est automatiquement prétraité (tokenization, padding, troncature)
        avant d'être passé au modèle. La prédiction retourne un score entre 0 (négatif)
        et 1 (positif).

        Args:
            sentence: Texte à analyser (en anglais de préférence pour de meilleures
                performances).

        Returns:
            Tableau numpy de shape (1, 1) contenant la probabilité que le sentiment
            soit positif. Valeur proche de 0 = négatif, proche de 1 = positif.

        Raises:
            ValueError: Si le texte ne peut pas être tokenisé (texte vide après
                prétraitement, caractères invalides, etc.).
            RuntimeError: Si le modèle échoue lors de la prédiction.
        """
        processed_input = seq_pad_and_trunc(
            sentence=sentence,
            tokenizer=self.tokenizer,
            maxlen=self.max_sequence_length,
        )
        return self.model.predict(processed_input)

