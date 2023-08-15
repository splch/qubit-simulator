import numpy as np
from qubit_simulator import Gates


def test_create_inverse_gate():
    X = Gates.X
    X_inv = Gates.create_inverse_gate(X)
    assert np.allclose(X @ X_inv, np.eye(2))


def test_create_controlled_gate():
    # Test for a controlled-X gate
    controlled_X = Gates.create_controlled_gate(Gates.X, 0, 1, 2)
    expected_result = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.allclose(controlled_X, expected_result)
