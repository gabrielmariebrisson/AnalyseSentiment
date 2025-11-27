from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src import config
from src.model_service import SentimentModel, _load_model, _load_tokenizer


def test_sentiment_model_predict_uses_padding_and_model(
    monkeypatch: pytest.MonkeyPatch,
    dummy_model: Any,
    dummy_tokenizer: Any,
) -> None:
    """
    Vérifie que SentimentModel.predict :
    - utilise le tokenizer pour transformer le texte,
    - applique le padding à la bonne longueur,
    - délègue la prédiction au modèle sous-jacent.
    """

    def fake_load_model(_: str) -> Any:
        return dummy_model

    def fake_load_tokenizer(_: str) -> Any:
        return dummy_tokenizer

    # On mocke les fonctions de chargement pour ne pas dépendre des fichiers réels
    monkeypatch.setattr("src.model_service._load_model", fake_load_model)
    monkeypatch.setattr("src.model_service._load_tokenizer", fake_load_tokenizer)

    service = SentimentModel()

    text = "this is a test"
    result = service.predict(text)

    # Type de retour : tableau numpy 2D
    result_array = np.asarray(result)
    assert result_array.shape == (1, 1)

    # Le modèle factice enregistre la dernière entrée passée à predict
    assert dummy_model.last_input is not None
    input_array = np.asarray(dummy_model.last_input)

    # Shape : (1, DEFAULT_MAX_SEQUENCE_LENGTH)
    assert input_array.ndim == 2
    assert input_array.shape[1] == config.DEFAULT_MAX_SEQUENCE_LENGTH


def test_sentiment_model_predict_propagates_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    dummy_model: Any,
    dummy_tokenizer: Any,
) -> None:
    """
    Vérifie que les erreurs remontent correctement (pas de swallow silencieux),
    ce qui est important pour diagnostiquer les problèmes en production.
    """

    def fake_load_model(_: str) -> Any:
        return dummy_model

    def fake_load_tokenizer(_: str) -> Any:
        return dummy_tokenizer

    monkeypatch.setattr("src.model_service._load_model", fake_load_model)
    monkeypatch.setattr("src.model_service._load_tokenizer", fake_load_tokenizer)

    service = SentimentModel()

    # On force une erreur dans le tokenizer
    def failing_texts_to_sequences(_: Any) -> None:
        raise ValueError("tokenizer failure")

    dummy_tokenizer.texts_to_sequences = failing_texts_to_sequences  # type: ignore[assignment]

    with pytest.raises(ValueError):
        service.predict("will fail")


