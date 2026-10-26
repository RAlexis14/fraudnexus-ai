"""ONNX extraction contract tests without the candidate model artifacts."""
import numpy as np
import pytest
from api.main import FraudEngine


def test_extract_array():
    x = np.array([[-1.0, 1.0], [2.0, -2.0]], dtype=np.float32)
    out = FraudEngine._to_two_columns(x, 2)
    assert out.shape == (2, 2)
    assert out[0, 1] == 1.0


def test_extract_zipmap():
    x = [{0: -1.0, 1: 1.0}, {0: 2.0, 1: -2.0}]
    out = FraudEngine._to_two_columns(x, 2)
    assert out.shape == (2, 2)
    assert out[1, 1] == -2.0


def test_reject_bad_shape():
    with pytest.raises(RuntimeError):
        FraudEngine._to_two_columns(np.array([1, 2]), 2)
