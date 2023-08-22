import pytest
import numpy as np
from qubit_simulator import Gates


def test_create_inverse_gate():
    random_matrix = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    random_unitary_gate, _ = np.linalg.qr(random_matrix)
    inverse_gate = Gates.create_inverse_gate(random_unitary_gate)
    assert np.allclose(random_unitary_gate @ inverse_gate, np.eye(2))


def test_create_controlled_gate():
    # Test for a controlled-X gate
    controlled_X = Gates.create_controlled_gate(Gates.X, 0, 1, 2)
    expected_result = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.allclose(controlled_X, expected_result)


def test_non_unitary_gate():
    # Define a non-unitary matrix (conjugate transpose is not equal to its inverse)
    non_unitary_gate = np.array([[2, 0], [0, 0.5]])
    with pytest.raises(ValueError):
        Gates._validate_gate(non_unitary_gate)


def test_create_controlled_gate_invalid_qubits():
    # Define a scenario where the control and target qubits are out of range
    with pytest.raises(IndexError):
        Gates.create_controlled_gate(
            Gates.X, control_qubit=4, target_qubit=5, num_qubits=3
        )
