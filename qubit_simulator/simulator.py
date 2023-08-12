import numpy as np
import collections
from .gates import gates


class QubitSimulator:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1
        self.circuit = []

    def _apply_gate(self, gate_name, gate, target_qubit, control_qubit=None):
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
        self.circuit.append((gate_name, target_qubit, control_qubit))

    def H(self, target_qubit):
        self._apply_gate("H", gates.H, target_qubit)

    def T(self, target_qubit):
        self._apply_gate("T", gates.T, target_qubit)

    def X(self, target_qubit):
        self._apply_gate("X", gates.X, target_qubit)

    def CNOT(self, control_qubit, target_qubit):
        self._apply_gate("X", gates.X, target_qubit, control_qubit)

    def U(self, target_qubit, theta, phi, lambda_):
        U = gates.U(theta, phi, lambda_)
        self._apply_gate("U", U, target_qubit)

    def CU(self, control_qubit, target_qubit, theta, phi, lambda_):
        U = gates.U(theta, phi, lambda_)
        self._apply_gate("U", U, target_qubit, control_qubit)

    def SWAP(self, qubit1, qubit2):
        self.CNOT(qubit1, qubit2)
        self.CNOT(qubit2, qubit1)
        self.CNOT(qubit1, qubit2)

    def Measure(self, shots=1, basis=None):
        state_vector = self.state_vector
        if basis is not None:
            state_vector = basis @ self.state_vector
        probabilities = np.abs(state_vector) ** 2
        result = np.random.choice(2**self.num_qubits, p=probabilities, size=shots)
        return [format(r, f"0{self.num_qubits}b") for r in result]

    def run(self, shots=100, basis=None):
        results = self.Measure(shots, basis)
        return dict(collections.Counter(results))

    def __str__(self):
        string = "-" * (len(self.circuit) * 4 + 1) + "\n"
        qubit_lines = ["|"] * self.num_qubits
        for gate_name, target_qubit, control_qubit in self.circuit:
            for i in range(self.num_qubits):
                if control_qubit == i:
                    qubit_lines[i] += " C |"
                elif target_qubit == i:
                    qubit_lines[i] += f" {gate_name} |"
                else:
                    qubit_lines[i] += "   |"
        string += "\n".join(qubit_lines)
        string += "\n" + "-" * (len(self.circuit) * 4 + 1)
        return string
