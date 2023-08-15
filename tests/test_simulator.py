import pytest
import numpy as np
from qubit_simulator import QubitSimulator, Gates


def test_initial_state():
    simulator = QubitSimulator(3)
    assert np.allclose(simulator.state_vector, [1, 0, 0, 0, 0, 0, 0, 0])


def test_zero_qubits():
    simulator = QubitSimulator(0)
    assert len(simulator.state_vector) == 1
    assert simulator.state_vector[0] == 1


def test_negative_qubits():
    with pytest.raises(ValueError):
        QubitSimulator(-1)  # Negative number of qubits is not allowed


def test_x_gate():
    simulator = QubitSimulator(1)
    simulator.x(0)
    # After applying the Pauli-X gate, the state should be |1⟩
    assert np.allclose(simulator.state_vector, [0, 1])


def test_h_gate():
    simulator = QubitSimulator(1)
    simulator.h(0)
    # After applying the Hadamard gate, the state should be an equal superposition
    assert np.allclose(simulator.state_vector, [0.70710678, 0.70710678])


def test_t_gate():
    simulator = QubitSimulator(1)
    simulator.x(0)  # Set the initial state to |1⟩
    simulator.t(0)
    # After applying the π/8 gate, the state should have a phase shift of π/4
    assert np.allclose(simulator.state_vector, [0, 0.70710678 + 0.70710678j])


def test_u_gate():
    theta = np.pi / 4
    phi = np.pi / 3
    lambda_ = np.pi / 2
    simulator = QubitSimulator(1)
    simulator.u(0, theta, phi, lambda_)
    # Expected result obtained from the U matrix using the given parameters
    expected_result = Gates.U(theta, phi, lambda_) @ [1, 0]
    assert np.allclose(simulator.state_vector, expected_result)


def test_cx_gate():
    simulator = QubitSimulator(2)
    simulator.state_vector = [0, 0, 0, 1]  # Set the initial state to |11⟩
    simulator.cx(0, 1)
    # After applying the CNOT gate, the state should be |10⟩ (big-endian)
    assert np.allclose(simulator.state_vector, [0, 0, 1, 0])


def test_cu_gate():
    theta = np.pi / 4
    phi = np.pi / 3
    lambda_ = np.pi / 2
    simulator = QubitSimulator(2)
    simulator.x(0)  # Set the control qubit to |1⟩
    simulator.cu(0, 1, theta, phi, lambda_)
    # Initial state |10⟩
    initial_state = np.array([0, 0, 1, 0], dtype=complex)
    # Apply U gate to the target qubit
    expected_result = np.kron(np.eye(2), Gates.U(theta, phi, lambda_)) @ initial_state
    assert np.allclose(simulator.state_vector, expected_result)


def test_cu_gate_no_effect():
    theta = np.pi / 4
    phi = np.pi / 3
    lambda_ = np.pi / 2
    simulator = QubitSimulator(2)
    # Control qubit is |0⟩, so the CU gate should have no effect
    simulator.cu(0, 1, theta, phi, lambda_)
    assert np.allclose(simulator.state_vector, [1, 0, 0, 0])


def test_target_control():
    simulator = QubitSimulator(3)
    simulator.x(0)
    simulator.x(2)  # Set the initial state to |101⟩
    simulator.cx(control_qubit=2, target_qubit=0)
    # After applying the CNOT gate, the state should be |001⟩
    assert np.allclose(simulator.state_vector, [0, 1, 0, 0, 0, 0, 0, 0])


def test_measure():
    simulator = QubitSimulator(1)
    simulator.x(0)
    # After applying the X gate, the state should be |1⟩
    result = simulator.measure()
    assert result == ["1"]


def test_measure_multiple_shots():
    shots = 100
    simulator = QubitSimulator(1)
    simulator.x(0)
    results = simulator.measure(shots=shots)
    # Measure 100 1s
    assert results.count("1") == shots


def test_non_unitary_gate():
    simulator = QubitSimulator(1)
    # Define a non-unitary matrix (conjugate transpose is not equal to its inverse)
    non_unitary_gate = np.array([[2, 0], [0, 0.5]])
    # Directly apply the non-unitary gate to the simulator
    simulator._apply_gate("NON_UNITARY", non_unitary_gate, 0)
    # Expected result after applying the non-unitary gate
    expected_result = non_unitary_gate @ [1, 0]
    assert np.allclose(simulator.state_vector, expected_result)
    with pytest.raises(ValueError):
        simulator.run()


@pytest.mark.parametrize("shots", [-1, -10])
def test_invalid_measurement_shots(shots):
    simulator = QubitSimulator(1)
    with pytest.raises(ValueError):
        simulator.run(shots=shots)  # Negative shots are invalid


def test_measure_without_gates():
    simulator = QubitSimulator(2)
    results = simulator.run(shots=100)
    assert results == {"00": 100}


def test_measure_custom_basis():
    simulator = QubitSimulator(1)
    # Define the transformation matrix for the Pauli-X basis
    X_basis = np.array([[0, 1], [1, 0]])
    # Apply the X gate to the qubit, transforming it to |1⟩
    simulator.x(0)
    # Measure in the X basis, which should result in the state |0⟩ in the X basis
    result = simulator.run(shots=10, basis=X_basis)
    assert set(result) == {"0"}


def test_invalid_basis_transformation():
    simulator = QubitSimulator(1)
    # Define an invalid basis transformation (not unitary)
    invalid_basis = np.array([[1, 2], [2, 1]])
    with pytest.raises(ValueError):
        simulator.run(basis=invalid_basis)


def test_measure_probabilities():
    shots = 1000
    simulator = QubitSimulator(1)
    simulator.h(0)
    results = simulator.run(shots=shots)
    assert abs(results.get("0", 0) - results.get("1", 0)) < shots / 4


def test_invalid_qubit_index():
    simulator = QubitSimulator(1)
    with pytest.raises(IndexError):
        simulator.h(2)  # Index out of range


def test_invalid_control_and_target_index():
    simulator = QubitSimulator(1)
    with pytest.raises(IndexError):
        simulator.cx(1, 0)  # Control qubit cannot be out of range


def test_run():
    simulator = QubitSimulator(1)
    # Running the simulation 10 times should produce 10 results
    results = simulator.run(10)
    assert results == {"0": 10}


def test_gate_reversibility():
    theta = np.pi / 2
    phi = np.pi / 4
    lambda_ = np.pi / 3
    simulator = QubitSimulator(1)
    simulator.u(0, theta, phi, lambda_)
    simulator.h(0)
    simulator.x(0)
    simulator.x(0)
    simulator.h(0)
    # Apply U inverse
    simulator.u(0, theta, phi, lambda_, inverse=True)
    assert np.allclose(simulator.state_vector, [1, 0])


def test_random_unitary_gate_inverse():
    simulator = QubitSimulator(1)
    # Generate a random unitary matrix using QR decomposition
    random_matrix = np.random.rand(2, 2) + 1j * np.random.rand(2, 2)
    random_unitary_gate, _ = np.linalg.qr(random_matrix)
    simulator._apply_gate("RANDOM_UNITARY", random_unitary_gate, 0)
    simulator._apply_gate("RANDOM_UNITARY_INV", random_unitary_gate.conj().T, 0)
    # The final state should be the same as the initial state
    assert np.allclose(simulator.state_vector, [1, 0])


@pytest.mark.parametrize(
    "num_qubits, expected_string",
    [
        (0, "-\n-"),
        (
            3,
            "-----------------------------------\n"
            "| H |   |   |          @          |\n"
            "|   | X | X | U(1.05, 0.63, 0.45) |\n"
            "|   |   | @ |                     |\n"
            "-----------------------------------",
        ),
    ],
)
def test_circuit_string(num_qubits, expected_string):
    simulator = QubitSimulator(num_qubits)
    if num_qubits == 3:
        simulator.h(0)
        simulator.x(1)
        simulator.cx(2, 1)
        simulator.cu(0, 1, np.pi / 3, np.pi / 5, np.pi / 7)
    assert str(simulator) == expected_string


def test_complex_circuit():
    simulator = QubitSimulator(3)
    simulator.h(0)
    simulator.u(1, np.pi / 4, np.pi / 4, np.pi / 2)
    simulator.cx(2, 0)
    simulator.cu(1, 2, np.pi / 2, np.pi / 4, np.pi / 8)
    simulator.x(0)
    simulator.run(shots=10)
    # This test verifies the process rather than the final state, so no assertion is needed


def test_bell_state():
    simulator = QubitSimulator(2)
    simulator.h(0)
    simulator.cx(0, 1)
    # After applying the Hadamard and CNOT gates, the state should be a Bell state
    assert np.allclose(simulator.state_vector, [0.70710678, 0, 0, 0.70710678])


def test_ghz_state():
    simulator = QubitSimulator(3)
    simulator.h(0)
    simulator.cx(0, 1)
    simulator.cx(0, 2)
    # After applying the Hadamard and CNOT gates, the state should be a GHZ state
    assert np.allclose(
        simulator.state_vector, [0.70710678, 0, 0, 0, 0, 0, 0, 0.70710678]
    )


@pytest.mark.parametrize("num_qubits", [1, 2, 5])
def test_qft(num_qubits):
    def apply_qft(simulator):
        num_qubits = simulator.num_qubits
        for target_qubit in range(num_qubits):
            simulator.h(target_qubit)
            for control_qubit in range(target_qubit + 1, num_qubits):
                phase_angle = 2 * np.pi / (2 ** (control_qubit - target_qubit + 1))
                simulator.cu(target_qubit, control_qubit, 0, 0, phase_angle)
        # Swap qubits to match the desired output order
        for i in range(num_qubits // 2):
            j = num_qubits - i - 1
            simulator.cx(i, j)
            simulator.cx(j, i)
            simulator.cx(i, j)

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
