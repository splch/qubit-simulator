import numpy as np
from . import gates


class QubitSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits)
        self.state_vector[0] = 1

    def _apply_gate(self, gate, target_qubit, control_qubit=None):
        if control_qubit is not None:  # Handle controlled gates
            # Create the controlled gate matrix
            control_gate = np.kron(
                gates.I
                - np.outer(gates.I[:, control_qubit], gates.I[control_qubit, :]),
                gate,
            )
            control_gate += np.kron(
                np.outer(gates.I[:, control_qubit], gates.I[control_qubit, :]), gates.I
            )
            # Create the operator using Kronecker products
            operator = 1
            for qubit in range(self.num_qubits):
                if qubit == control_qubit or qubit == target_qubit:
                    continue
                operator = np.kron(operator, gates.I)
            operator = np.kron(operator, control_gate)
        else:  # Handle single qubit gates
            operator = 1
            for qubit in range(self.num_qubits):
                if qubit == target_qubit:
                    operator = np.kron(operator, gate)
                else:
                    operator = np.kron(operator, gates.I)
        # Apply the operator to the state vector
        self.state_vector = np.dot(operator, self.state_vector)

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
