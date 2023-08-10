import numpy as np
from qubit_simulator import QubitSimulator, gates


def test_x_gate():
    simulator = QubitSimulator(1)
    simulator.X(0)
    # After applying the Pauli-X gate, the state should be |1⟩
    assert np.allclose(simulator.state_vector, [0, 1])


def test_h_gate():
    simulator = QubitSimulator(1)
    simulator.H(0)
    # After applying the Hadamard gate, the state should be an equal superposition
    assert np.allclose(simulator.state_vector, [0.70710678, 0.70710678])


def test_t_gate():
    simulator = QubitSimulator(1)
    simulator.X(0)  # Set the initial state to |1⟩
    simulator.T(0)
    # After applying the π/8 gate, the state should have a phase shift of π/4
    assert np.allclose(simulator.state_vector, [0, 0.70710678 + 0.70710678j])


def test_u_gate():
    theta = np.pi / 2
    phi = np.pi / 4
    lambda_ = np.pi / 3
    simulator = QubitSimulator(1)
    simulator.U(0, theta, phi, lambda_)
    U = gates.U(theta, phi, lambda_)
    expected_state_vector = np.dot(U, [1, 0])  # Initial state is |0⟩
    assert np.allclose(simulator.state_vector, expected_state_vector)


def test_controlled_u_gate():
    theta = np.pi / 2
    phi = np.pi / 4
    lambda_ = np.pi / 3
    simulator = QubitSimulator(2)
    simulator.X(0)  # Set the control qubit to |1⟩
    simulator.CU(0, 1, theta, phi, lambda_)
    U = gates.U(theta, phi, lambda_)
    expected_state_vector = np.kron(
        [0, 1], np.dot(U, [1, 0])
    )  # Expected state is |1⟩ ⊗ U|0⟩
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
    assert results == {"0": 10}


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
