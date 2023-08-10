import numpy as np
from . import gates


class QubitSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1

    def _apply_gate(self, gate, targets, control=None):
        op_matrix = 1
        target_idx = 0
        for qubit in range(self.num_qubits):
            current_gate = gate[target_idx] if qubit in targets else gates.I
            op_matrix = np.kron(op_matrix, current_gate)
            if qubit in targets:
                target_idx += 1
        if control is not None:
            controlled_op_matrix = np.eye(2**self.num_qubits, dtype=complex)
            for state in range(2**self.num_qubits):
                binary_state = format(state, f"0{self.num_qubits}b")
                if all(binary_state[c] == "1" for c in control):
                    controlled_op_matrix[state, :] = np.dot(
                        op_matrix, controlled_op_matrix[state, :]
                    )
            op_matrix = controlled_op_matrix
        self.state_vector = np.dot(op_matrix, self.state_vector)

    def X(self, target):
        self._apply_gate([gates.X], [target])

    def H(self, target):
        self._apply_gate([gates.H], [target])

    def T(self, target):
        self._apply_gate([gates.T], [target])

    def U(self, target, theta, phi, lambda_):
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
        self._apply_gate([U_gate], [target])

    def CNOT(self, control, target):
        self._apply_gate([gates.X], [target], [control])

    def Measure(self, shots=1):
        probabilities = np.abs(self.state_vector) ** 2
        result = np.random.choice(2**self.num_qubits, p=probabilities, size=shots)
        return [format(r, f"0{self.num_qubits}b") for r in result]

    def run(self, num_shots=100):
        return self.Measure(shots=num_shots)
