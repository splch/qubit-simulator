import numpy as np
import collections
from .gates import gates


class QubitSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1

    def _apply_gate(self, gate, target_qubit, control_qubit=None):
        # If there's a control qubit, create the controlled gate using the provided function
        if control_qubit is not None:
            operator = gates.create_controlled_gate(
                gate, control_qubit, target_qubit, self.num_qubits
            )
        # Otherwise, build the operator for the target qubit and use the identity for other qubits
        else:
            operator = 1
            for qubit in range(self.num_qubits):
                operator = np.kron(
                    operator,
                    gate if qubit == target_qubit else np.eye(2, dtype=complex),
                )
        # Apply the operator to the state vector
        self.state_vector = operator @ self.state_vector

    def H(self, target_qubit):
        self._apply_gate(gates.H, target_qubit)

    def T(self, target_qubit):
        self._apply_gate(gates.T, target_qubit)

    def X(self, target_qubit):
        self._apply_gate(gates.X, target_qubit)

    def CNOT(self, control_qubit, target_qubit):
        self._apply_gate(gates.X, target_qubit, control_qubit)

    def U(self, target_qubit, theta, phi, lambda_):
        U = gates.U(theta, phi, lambda_)
        self._apply_gate(U, target_qubit)

    def CU(self, control_qubit, target_qubit, theta, phi, lambda_):
        U = gates.U(theta, phi, lambda_)
        self._apply_gate(U, target_qubit, control_qubit)

    def Measure(self, shots=1):
        probabilities = np.abs(self.state_vector) ** 2
        result = np.random.choice(2**self.num_qubits, p=probabilities, size=shots)
        return [format(r, f"0{self.num_qubits}b") for r in result]

    def run(self, shots=100):
        results = self.Measure(shots)
        return collections.Counter(results)
