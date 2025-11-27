from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, List

import numpy as np
import pytest

# Ajouter le répertoire parent au PYTHONPATH pour permettre les imports `from src import ...`
# Cela permet de lancer pytest depuis n'importe quel répertoire
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from src import config


class DummyTokenizer:
    """Tokenizer minimal pour les tests."""

    def texts_to_sequences(self, texts: Iterable[str]) -> List[List[int]]:
        # Conversion très simple : longueur du mot comme id
        sequences: List[List[int]] = []
        for text in texts:
            tokens = text.split()
            sequences.append([len(token) for token in tokens])
        return sequences


@pytest.fixture()
def dummy_tokenizer() -> DummyTokenizer:
    """Retourne un tokenizer factice pour les tests."""
    return DummyTokenizer()


@pytest.fixture()
def dummy_model() -> Any:
    """Retourne un modèle Keras factice avec une méthode predict."""

    class _DummyModel:
        def __init__(self) -> None:
            self.last_input: Any | None = None

        def predict(self, x: Any) -> np.ndarray:
            self.last_input = x
            # Retourne une proba fixe pour simplifier les assertions
            return np.array([[0.7]], dtype=float)

    return _DummyModel()


@pytest.fixture(autouse=True)
def reset_default_sequence_length(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    S'assure que la longueur de séquence par défaut est cohérente dans les tests,
    même si elle est modifiée ailleurs.
    """

    monkeypatch.setattr(config, "DEFAULT_MAX_SEQUENCE_LENGTH", config.DEFAULT_MAX_SEQUENCE_LENGTH, raising=False)


