from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable, List
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ajouter le répertoire parent au PYTHONPATH pour permettre les imports `from src import ...`
# Cela permet de lancer pytest depuis n'importe quel répertoire
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))


def _setup_mocks() -> None:
    """Configure les mocks pour les dépendances externes.
    
    Cette fonction doit être appelée AVANT tout import des modules src
    pour s'assurer que les mocks sont en place.
    """
    # Si les modules sont déjà importés, on les supprime pour forcer le rechargement avec les mocks
    modules_to_remove = [
        "streamlit",
        "tensorflow",
        "tensorflow.keras",
        "tensorflow.keras.models",
        "tensorflow.keras.preprocessing",
        "tensorflow.keras.preprocessing.sequence",
        "deep_translator",
        "src.utils",
        "src.model_service",
    ]
    for module_name in modules_to_remove:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Mocker les dépendances AVANT l'import des modules src pour éviter ModuleNotFoundError
    # Cela permet de tester la logique sans dépendre de ces packages installés

    # 1. Mocker streamlit
    _mock_session_state = MagicMock()
    _mock_session_state.language = "fr"
    _mock_session_state.translations_cache = {}

    _mock_streamlit = MagicMock()
    _mock_streamlit.session_state = _mock_session_state
    # @st.cache_resource devient une fonction identité (pas de cache dans les tests)
    _mock_streamlit.cache_resource = lambda func: func
    sys.modules["streamlit"] = _mock_streamlit

    # 2. Mocker tensorflow et ses sous-modules
    _mock_tensorflow = MagicMock()
    _mock_keras = MagicMock()
    _mock_keras_preprocessing = MagicMock()
    _mock_keras_preprocessing_sequence = MagicMock()

    # pad_sequences doit être une vraie fonction (utilisée dans seq_pad_and_trunc)
    # On importe numpy pour créer une fonction mock qui retourne un array
    def _mock_pad_sequences(sequences, maxlen=None, padding="post", truncating="post"):
        """Mock de pad_sequences qui simule le comportement basique."""
        import numpy as np
        if not sequences:
            return np.zeros((1, maxlen or 100), dtype=int)
        seq = sequences[0]
        if maxlen:
            if len(seq) > maxlen:
                if truncating == "post":
                    seq = seq[:maxlen]
                else:  # pre
                    seq = seq[-maxlen:]
            elif len(seq) < maxlen:
                padding_value = 0
                if padding == "post":
                    seq = seq + [padding_value] * (maxlen - len(seq))
                else:  # pre
                    seq = [padding_value] * (maxlen - len(seq)) + seq
        return np.array([seq], dtype=int)

    _mock_keras_preprocessing_sequence.pad_sequences = _mock_pad_sequences
    _mock_keras_preprocessing.sequence = _mock_keras_preprocessing_sequence
    _mock_keras.preprocessing = _mock_keras_preprocessing
    _mock_tensorflow.keras = _mock_keras

    # Créer un mock pour tf.keras.Model (utilisé dans les type hints)
    _mock_model_class = MagicMock()
    _mock_keras.Model = _mock_model_class

    # Mock pour tf.keras.models.load_model (utilisé dans _load_model)
    _mock_keras_models = MagicMock()
    _mock_keras_models.load_model = MagicMock()  # Sera remplacé par les tests qui mockent _load_model
    _mock_keras.models = _mock_keras_models

    sys.modules["tensorflow"] = _mock_tensorflow
    sys.modules["tensorflow.keras"] = _mock_keras
    sys.modules["tensorflow.keras.models"] = _mock_keras_models
    sys.modules["tensorflow.keras.preprocessing"] = _mock_keras_preprocessing
    sys.modules["tensorflow.keras.preprocessing.sequence"] = _mock_keras_preprocessing_sequence

    # 3. Mocker deep_translator
    _mock_google_translator = MagicMock()

    class _MockGoogleTranslator:
        """Mock de GoogleTranslator pour éviter les appels API réels."""
        def __init__(self, source: str = "fr", target: str = "en"):
            self.source = source
            self.target = target

        def translate(self, text: str) -> str:
            # Retourne le texte original pour les tests (pas de vraie traduction)
            return f"[{self.target}]{text}"

    _mock_google_translator.GoogleTranslator = _MockGoogleTranslator
    sys.modules["deep_translator"] = _mock_google_translator


# Appeler _setup_mocks() immédiatement pour s'assurer que les mocks sont en place
_setup_mocks()

# Maintenant on peut importer config en toute sécurité
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


@pytest.fixture(autouse=True)
def reset_streamlit_session_state() -> None:
    """
    Réinitialise le session_state mocké entre chaque test pour éviter les effets de bord.
    """
    # Récupérer le mock depuis sys.modules pour éviter les problèmes de scope
    mock_st = sys.modules.get("streamlit")
    if mock_st and hasattr(mock_st, "session_state"):
        mock_st.session_state.language = "fr"
        mock_st.session_state.translations_cache = {}


# Hook pytest pour s'assurer que les mocks sont en place avant la collecte des tests
def pytest_configure(config: pytest.Config) -> None:
    """S'assure que les mocks sont en place avant la collecte des tests."""
    _setup_mocks()


