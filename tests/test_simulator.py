import pytest
import numpy as np
from qubit_simulator import QubitSimulator
from qubit_simulator import gates


def test_hadamard_gate():
    simulator = QubitSimulator(1)
    simulator.H(0)
    # After applying the Hadamard gate, the state should be an equal superposition
    assert np.allclose(simulator.state_vector, [0.70710678, 0.70710678])


def test_t_gate():
    simulator = QubitSimulator(1)
    simulator.X(0)  # Set the initial state to |1⟩
    simulator.T(0)
    # After applying the T gate, the state should have a phase shift of pi/4
    assert np.allclose(simulator.state_vector, [0, 0.70710678 + 0.70710678j])


def test_u_gate():
    theta = np.pi / 2
    phi = np.pi / 4
    lambda_ = np.pi / 3
    simulator = QubitSimulator(1)
    simulator.U(0, theta, phi, lambda_)
    U_gate = np.array(
        [
            [np.cos(theta / 2), -np.exp(1j * lambda_) * np.sin(theta / 2)],
            [
                np.exp(1j * phi) * np.sin(theta / 2),
                np.exp(1j * lambda_ + 1j * phi) * np.cos(theta / 2),
            ],
        ],
        dtype=complex,
    )
    expected_state_vector = np.dot(U_gate, [1, 0])  # Initial state is |0⟩
    assert np.allclose(simulator.state_vector, expected_state_vector)


def test_controlled_u_gate():
    theta = np.pi / 2
    phi = np.pi / 4
    lambda_ = np.pi / 3
    controlled_U_gate = np.eye(4, dtype=complex)
    controlled_U_gate[2:4, 2:4] = gates.U(theta, phi, lambda_)
    simulator = QubitSimulator(2)
    simulator.X(0)  # Set control qubit to |1⟩
    # Apply controlled-U gate manually by dot product with the state vector
    simulator.state_vector = np.dot(controlled_U_gate, simulator.state_vector)
    expected_state_vector = np.dot(
        controlled_U_gate, [0, 0, 1, 0]
    )  # Initial state is |10⟩
    assert np.allclose(simulator.state_vector, expected_state_vector)


def test_cnot_gate():
    simulator = QubitSimulator(2)
    simulator.state_vector = [0, 0, 0, 1]  # Set the initial state to |11⟩
    simulator.CNOT(0, 1)
    # After applying the CNOT gate, the state should be |10⟩ (big-endian)
    assert np.allclose(simulator.state_vector, [0, 0, 1, 0])


def test_target_control():
    simulator = QubitSimulator(3)
    simulator.X(0)
    simulator.X(2)  # Set the initial state to |101⟩
    simulator.CNOT(control=2, target=0)
    # After applying the CNOT gate, the state should be |001⟩
    assert np.allclose(simulator.state_vector, [0, 1, 0, 0, 0, 0, 0, 0])


def test_measure():
    simulator = QubitSimulator(1)
    simulator.X(0)
    # After applying the X gate, the state should be |1⟩
    result = simulator.Measure()
    assert result == ["1"]


def test_run():
    simulator = QubitSimulator(1)
    # Running the simulation 10 times should produce 10 results
    results = simulator.run(10)
    assert len(results) == 10
    assert set(results) == {"0"}


def test_bell_state():
    simulator = QubitSimulator(2)
    simulator.H(0)
    simulator.CNOT(0, 1)
    # After applying the Hadamard and CNOT gates, the state should be a Bell state
    assert np.allclose(simulator.state_vector, [0.70710678, 0, 0, 0.70710678])


def test_ghz_state():
    simulator = QubitSimulator(3)
    simulator.H(0)
    simulator.CNOT(0, 1)
    simulator.CNOT(0, 2)
    # After applying the Hadamard and CNOT gates, the state should be a GHZ state
    assert np.allclose(
        simulator.state_vector, [0.70710678, 0, 0, 0, 0, 0, 0, 0.70710678]
    )
