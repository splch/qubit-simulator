import numpy as np
from . import gates


class QubitSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits)
        self.state_vector[0] = 1

    def _apply_gate(self, gate, target_qubit, control_qubit=None):
        # Create an operation matrix for applying the gate
        if control_qubit is None:
            # Single-qubit gate
            op_matrix = 1
            for qubit in range(self.num_qubits):
                current_gate = gate if qubit == target_qubit else gates.I
                op_matrix = np.kron(op_matrix, current_gate)
        else:
            # Controlled gate
            op_matrix = np.eye(2**self.num_qubits)
            for state in range(2**self.num_qubits):
                binary_state = format(state, f"0{self.num_qubits}b")
                if binary_state[control_qubit] == '1':
                    submatrix = 1
                    for qubit in range(self.num_qubits):
                        current_gate = gate if qubit == target_qubit else gates.I
                        submatrix = np.kron(submatrix, current_gate)
                    op_matrix[state, :] = np.dot(submatrix, op_matrix[state, :])
        # Apply the operation matrix to the state vector
        self.state_vector = np.dot(op_matrix, self.state_vector)

    def H(self, target_qubit):
        self._apply_gate(gates.H, target_qubit)

    def T(self, target_qubit):
        self._apply_gate(gates.T, target_qubit)

    def X(self, target_qubit):
        self._apply_gate(gates.X, target_qubit)

    def CNOT(self, control_qubit, target_qubit):
        self._apply_gate(gates.X, target_qubit, control_qubit)

    def Measure(self):
        probabilities = np.abs(self.state_vector) ** 2
        result = np.random.choice(2**self.num_qubits, p=probabilities)
        return format(result, f"0{self.num_qubits}b")

    def run(self, num_shots=100):
        results = [self.Measure() for _ in range(num_shots)]
        return results
