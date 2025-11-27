from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src import config
from src.utils import seq_pad_and_trunc


def _get_sequence_and_length(result: Any) -> tuple[np.ndarray, int]:
    """Helper pour extraire le tableau numpy et sa longueur."""
    array = np.asarray(result)
    assert array.ndim == 2
    _, length = array.shape
    return array, length


def test_seq_pad_and_trunc_empty_text(dummy_tokenizer: Any) -> None:
    """Un texte vide doit produire une séquence de zéros de la bonne longueur."""
    result = seq_pad_and_trunc(
        sentence="",
        tokenizer=dummy_tokenizer,
        maxlen=config.DEFAULT_MAX_SEQUENCE_LENGTH,
    )
    array, length = _get_sequence_and_length(result)

    assert length == config.DEFAULT_MAX_SEQUENCE_LENGTH
    # Tout doit être à zéro pour un texte vide
    assert np.all(array == 0)


def test_seq_pad_and_trunc_short_text(dummy_tokenizer: Any) -> None:
    """Un texte court doit être padé en fin de séquence."""
    sentence = "hello world"
    result = seq_pad_and_trunc(
        sentence=sentence,
        tokenizer=dummy_tokenizer,
        maxlen=5,
    )
    array, length = _get_sequence_and_length(result)

    assert length == 5
    # DummyTokenizer encode la longueur des mots : [5, 5] puis padding à 0
    assert np.array_equal(array[0, :2], np.array([5, 5]))
    assert np.array_equal(array[0, 2:], np.zeros(3, dtype=int))


def test_seq_pad_and_trunc_long_text_truncation(dummy_tokenizer: Any) -> None:
    """Un texte plus long que maxlen doit être tronqué correctement."""
    # 10 tokens -> sera tronqué à maxlen=4
    sentence = "a b c d e f g h i j"
    result = seq_pad_and_trunc(
        sentence=sentence,
        tokenizer=dummy_tokenizer,
        maxlen=4,
    )
    array, length = _get_sequence_and_length(result)

    assert length == 4
    # Comme tous les tokens sont de longueur 1, tout doit être à 1
    assert np.array_equal(array[0], np.ones(4, dtype=int))


@pytest.mark.parametrize("padding,truncating", [("post", "post"), ("pre", "pre")])
def test_seq_pad_and_trunc_padding_and_truncating_modes(
    dummy_tokenizer: Any,
    padding: str,
    truncating: str,
) -> None:
    """Vérifie que les paramètres de padding/truncating sont bien propagés."""
    sentence = "one two three"
    result = seq_pad_and_trunc(
        sentence=sentence,
        tokenizer=dummy_tokenizer,
        padding=padding,
        truncating=truncating,
        maxlen=5,
    )
    array, length = _get_sequence_and_length(result)

    assert length == 5
    # On ne vérifie pas le détail du comportement interne de Keras,
    # seulement que l'appel ne lève pas d'exception et retourne le bon shape.


