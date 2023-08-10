import numpy as np
from . import gates


class QubitSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1

    def _apply_gate(self, gate, target_qubits, control_qubit=None):
        op_matrix = 1
        target_idx = 0
        for qubit in range(self.num_qubits):
            current_gate = gate[target_idx] if qubit in target_qubits else gates.I
            op_matrix = np.kron(op_matrix, current_gate)
            if qubit in target_qubits:
                target_idx += 1
        if control_qubit is not None:
            controlled_op_matrix = np.eye(2**self.num_qubits, dtype=complex)
            for state in range(2**self.num_qubits):
                binary_state = format(state, f"0{self.num_qubits}b")
                if all(binary_state[c] == "1" for c in control_qubit):
                    controlled_op_matrix[state, :] = np.dot(
                        op_matrix, controlled_op_matrix[state, :]
                    )
            op_matrix = controlled_op_matrix
        self.state_vector = np.dot(op_matrix, self.state_vector)

    def X(self, target_qubit):
        self._apply_gate([gates.X], [target_qubit])

    def H(self, target_qubit):
        self._apply_gate([gates.H], [target_qubit])

    def S(self, target_qubit):
        self._apply_gate([gates.S], [target_qubit])

    def T(self, target_qubit):
        self._apply_gate([gates.T], [target_qubit])

    def CNOT(self, control_qubit, target_qubit):
        self._apply_gate([gates.X], [target_qubit], [control_qubit])

    def Measure(self):
        probabilities = np.abs(self.state_vector) ** 2
        result = np.random.choice(2**self.num_qubits, p=probabilities)
        return format(result, f"0{self.num_qubits}b")

    def run(self, num_shots=100):
        results = [self.Measure() for _ in range(num_shots)]
        return results
