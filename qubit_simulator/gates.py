import numpy as np


class gates:
    # Hadamard (H) gate
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    # π/8 (T) gate
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])
    # Pauli-X (NOT) gate
    X = np.array([[0, 1], [1, 0]])

    # Generic (U) gate
    @staticmethod
    def U(theta, phi, lambda_):
        return np.array(
            [
                [np.cos(theta), -np.exp(1j * lambda_) * np.sin(theta)],
                [
                    np.exp(1j * phi) * np.sin(theta),
                    np.exp(1j * (phi + lambda_)) * np.cos(theta),
                ],
            ]
        )

    @staticmethod
    def create_controlled_gate(gate, control_qubit, target_qubit, num_qubits):
        controlled_gate = np.eye(2**num_qubits, dtype=complex)
        for basis in range(2**num_qubits):
            basis_binary = format(basis, f"0{num_qubits}b")
            if basis_binary[control_qubit] == "1":
                target_state = int(
                    basis_binary[:target_qubit]
                    + str(1 - int(basis_binary[target_qubit]))
                    + basis_binary[target_qubit + 1 :],
                    2,
                )
                controlled_gate[basis, basis] = gate[
                    int(basis_binary[target_qubit]), int(basis_binary[target_qubit])
                ]
                controlled_gate[basis, target_state] = gate[
                    int(basis_binary[target_qubit]), 1 - int(basis_binary[target_qubit])
                ]
                controlled_gate[target_state, basis] = gate[
                    1 - int(basis_binary[target_qubit]), int(basis_binary[target_qubit])
                ]
                controlled_gate[target_state, target_state] = gate[
                    1 - int(basis_binary[target_qubit]),
                    1 - int(basis_binary[target_qubit]),
                ]
        return controlled_gate
