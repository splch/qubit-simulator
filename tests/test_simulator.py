import pytest
import numpy as np
from qubit_simulator import QubitSimulator


def test_hadamard_gate():
    simulator = QubitSimulator(1)
    simulator.H(0)
    # After applying the Hadamard gate, the state should be an equal superposition
    assert np.allclose(simulator.state_vector, [0.70710678, 0.70710678])


def test_t_gate():
    simulator = QubitSimulator(1)
    simulator.X(0) # Set the initial state to |1>
    simulator.T(0)
    # After applying the T gate, the state should have a phase shift of pi/4
    assert np.allclose(simulator.state_vector, [0, 0.70710678 + 0.70710678j])


def test_cnot_gate():
    simulator = QubitSimulator(2)
    simulator.state_vector = [0, 0, 0, 1]  # Set the initial state to |11>
    simulator.CNOT(0, 1)
    # After applying the CNOT gate, the state should be |10>
    assert np.allclose(simulator.state_vector, [0, 0, 1, 0])


def test_measure():
    simulator = QubitSimulator(1)
    simulator.X(0)
    # After applying the X gate, the state should be |1>
    result = simulator.Measure()
    assert result == "1"


def test_run():
    simulator = QubitSimulator(1)
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
    # The state should be a GHZ state
    expected_state = np.array([0.70710678, 0, 0, 0, 0, 0, 0, 0.70710678])
    assert np.allclose(simulator.state_vector, expected_state)
