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


def test_custom_2qubit_gate():
    custom_gate = np.kron(gates.X, gates.H)
    simulator = QubitSimulator(2)
    simulator._apply_gate([gates.X, gates.H], [0, 1])
    # After applying the custom 2-qubit gate, the state should be transformed according to the custom_gate
    assert np.allclose(simulator.state_vector, np.dot(custom_gate, [1, 0, 0, 0]))


def test_qft():
    simulator = QubitSimulator(3)
    # Applying QFT manually
    for k in range(3):
        simulator.H(k)
        for j in range(k + 1, 3):
            phase = np.pi / (2 ** (j - k))
            phase_shift = np.array([[1, 0], [0, np.exp(1j * phase)]])
            simulator._apply_gate([phase_shift], [j], [k])
    # After applying the Quantum Fourier Transform, the state should be in the QFT state
    assert np.allclose(simulator.state_vector, [1 / np.sqrt(8)] * 8)
