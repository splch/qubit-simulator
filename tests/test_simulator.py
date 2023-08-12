import pytest
import numpy as np
from qubit_simulator import QubitSimulator, gates


def test_initial_state():
    simulator = QubitSimulator(3)
    assert np.allclose(simulator.state_vector, [1, 0, 0, 0, 0, 0, 0, 0])


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


def test_cnot_gate():
    simulator = QubitSimulator(2)
    simulator.state_vector = [0, 0, 0, 1]  # Set the initial state to |11⟩
    simulator.CNOT(0, 1)
    # After applying the CNOT gate, the state should be |10⟩ (big-endian)
    assert np.allclose(simulator.state_vector, [0, 0, 1, 0])


def test_u_gate():
    theta = np.pi / 4
    phi = np.pi / 3
    lambda_ = np.pi / 2
    simulator = QubitSimulator(1)
    simulator.U(0, theta, phi, lambda_)
    # Expected result obtained from the U matrix using the given parameters
    expected_result = gates.U(theta, phi, lambda_) @ [1, 0]
    assert np.allclose(simulator.state_vector, expected_result)


def test_cu_gate():
    theta = np.pi / 4
    phi = np.pi / 3
    lambda_ = np.pi / 2
    simulator = QubitSimulator(2)
    simulator.X(0)  # Set the control qubit to |1⟩
    simulator.CU(0, 1, theta, phi, lambda_)
    # Initial state |10⟩
    initial_state = np.array([0, 0, 1, 0], dtype=complex)
    # Apply U gate to the target qubit
    expected_result = np.kron(np.eye(2), gates.U(theta, phi, lambda_)) @ initial_state
    assert np.allclose(simulator.state_vector, expected_result)


def test_swap_gate():
    simulator = QubitSimulator(2)
    simulator.state_vector = np.array(
        [0, 1, 0, 0], dtype=complex
    )  # Set the initial state to |01⟩
    simulator.SWAP(0, 1)
    # After swapping, the state should be |10⟩
    expected_state = np.array([0, 0, 1, 0], dtype=complex)
    assert np.allclose(simulator.state_vector, expected_state)


def test_target_control():
    simulator = QubitSimulator(3)
    simulator.X(0)
    simulator.X(2)  # Set the initial state to |101⟩
    simulator.CNOT(control_qubit=2, target_qubit=0)
    # After applying the CNOT gate, the state should be |001⟩
    assert np.allclose(simulator.state_vector, [0, 1, 0, 0, 0, 0, 0, 0])


def test_measure():
    simulator = QubitSimulator(1)
    simulator.X(0)
    # After applying the X gate, the state should be |1⟩
    result = simulator.Measure()
    assert result == ["1"]


def test_measure_multiple_shots():
    shots = 100
    simulator = QubitSimulator(1)
    simulator.X(0)
    results = simulator.Measure(shots=shots)
    # Measure 100 1s
    assert results.count("1") == shots


def test_measure_custom_basis():
    simulator = QubitSimulator(1)
    # Define the transformation matrix for the Pauli-X basis
    X_basis = np.array([[0, 1], [1, 0]])
    # Apply the X gate to the qubit, transforming it to |1⟩
    simulator.X(0)
    # Measure in the X basis, which should result in the state |0⟩ in the X basis
    result = simulator.run(shots=10, basis=X_basis)
    assert set(result) == {"0"}


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


def test_gate_reversibility():
    theta = np.pi / 2
    phi = np.pi / 4
    lambda_ = np.pi / 3
    simulator = QubitSimulator(1)
    simulator.U(0, theta, phi, lambda_)
    simulator.H(0)
    simulator.X(0)
    simulator.X(0)
    simulator.H(0)
    # Apply U inverse
    U_inv = np.conjugate(gates.U(theta, phi, lambda_).T)
    simulator._apply_gate("U", U_inv, 0)
    assert np.allclose(simulator.state_vector, [1, 0])


def test_measure_probabilities():
    shots = 10000
    simulator = QubitSimulator(1)
    simulator.H(0)
    results = simulator.run(shots=shots)
    assert abs(results.get("0", 0) - results.get("1", 0)) < shots / 4


def test_str():
    simulator = QubitSimulator(3)
    simulator.H(0)
    simulator.CNOT(0, 2)
    expected_string = "---------\n| H | C |\n|   |   |\n|   | X |\n---------"
    assert str(simulator) == expected_string


@pytest.mark.parametrize("num_qubits", [1, 2, 5])
def test_qft(num_qubits):
    def apply_qft(simulator):
        num_qubits = simulator.num_qubits
        for target_qubit in range(num_qubits):
            simulator.H(target_qubit)
            for control_qubit in range(target_qubit + 1, num_qubits):
                phase_angle = 2 * np.pi / (2 ** (control_qubit - target_qubit + 1))
                simulator.CU(target_qubit, control_qubit, 0, 0, phase_angle)
        # Swap qubits to match the desired output order
        for i in range(num_qubits // 2):
            j = num_qubits - i - 1
            simulator.SWAP(i, j)

    simulator = QubitSimulator(num_qubits)
    # Create a random initial state vector and normalize it
    random_state = np.random.rand(2**num_qubits) + 1j * np.random.rand(
        2**num_qubits
    )
    random_state /= np.linalg.norm(random_state)
    # Set the random state as the initial state in the simulator
    simulator.state_vector = random_state.copy()
    # Apply QFT in the simulator
    apply_qft(simulator)
    # Compute the expected result using NumPy's FFT and normalize
    fft_result = np.fft.fft(random_state) / np.sqrt(2**num_qubits)
    # Compare the state vectors
    assert np.allclose(
        sorted(simulator.state_vector, key=lambda x: (x.real, x.imag)),
        sorted(fft_result, key=lambda x: (x.real, x.imag)),
    )
