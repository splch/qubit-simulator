import numpy as np
from collections import Counter
from typing import Optional, List, Tuple, Dict
from .gates import Gates


class QubitSimulator:
    """
    A class that represents a quantum simulator.
    """

    def __init__(self, num_qubits: int):
        """
        Initialize the simulator with given number of qubits.

        :param num_qubits: Number of qubits.
        """
        self.num_qubits = num_qubits
        self.state_vector = np.zeros(2**num_qubits, dtype=complex)
        self.state_vector[0] = 1
        self.circuit: List[Tuple[str, int, Optional[int]]] = []

    def _validate_qubit_index(
        self, target_qubit: int, control_qubit: Optional[int] = None
    ):
        """
        Validates the qubit indices.

        :param target_qubit: Index of the target qubit to validate.
        :param control_qubit: Index of the control qubit to validate.
        :raises IndexError: If the qubit index is out of range.
        :raises ValueError: If the control qubit and target qubit are the same.
        """
        if target_qubit < 0 or target_qubit >= self.num_qubits:
            raise IndexError(f"Target qubit index {target_qubit} out of range.")
        if control_qubit is not None:
            if control_qubit < 0 or control_qubit >= self.num_qubits:
                raise IndexError(f"Control qubit index {control_qubit} out of range.")

    def _apply_gate(
        self,
        gate_name: str,
        gate: np.ndarray,
        target_qubit: int,
        control_qubit: Optional[int] = None,
    ):
        """
        Applies the given gate to the target qubit.

        :param gate_name: Name of the gate.
        :param gate: Matrix representing the gate.
        :param target_qubit: Index of the target qubit.
        :param control_qubit: Index of the control qubit (if controlled gate).
        """
        # Validate the target and control qubit indices
        self._validate_qubit_index(target_qubit, control_qubit)
        if control_qubit is not None:
            operator = Gates.create_controlled_gate(
                gate, control_qubit, target_qubit, self.num_qubits
            )
        else:
            operator = np.eye(1)
            for qubit in range(self.num_qubits):
                operator = np.kron(
                    operator,
                    gate if qubit == target_qubit else np.eye(2),
                )
        self.state_vector = operator @ self.state_vector
        self.circuit.append((gate_name, target_qubit, control_qubit))

    def h(self, target_qubit: int):
        """
        Applies Hadamard gate to the target qubit.

        :param target_qubit: Index of the target qubit.
        """
        self._apply_gate("H", Gates.H, target_qubit)

    def t(self, target_qubit: int):
        """
        Applies π/8 gate to the target qubit.

        :param target_qubit: Index of the target qubit.
        """
        self._apply_gate("T", Gates.T, target_qubit)

    def x(self, target_qubit: int):
        """
        Applies Not gate to the target qubit.

        :param target_qubit: Index of the target qubit.
        """
        self._apply_gate("X", Gates.X, target_qubit)

    def cx(self, control_qubit: int, target_qubit: int):
        """
        Applies Controlled-Not gate to the target qubit.

        :param control_qubit: Index of the control qubit.
        :param target_qubit: Index of the target qubit.
        """
        self._apply_gate("X", Gates.X, target_qubit, control_qubit)

    def u(self, target_qubit: int, theta: float, phi: float, lambda_: float):
        """
        Applies Generic gate to the target qubit.

        :param target_qubit: Index of the target qubit.
        :param theta: Angle theta.
        :param phi: Angle phi.
        :param lambda_: Angle lambda.
        """
        U = Gates.U(theta, phi, lambda_)
        self._apply_gate("U", U, target_qubit)

    def cu(
        self,
        control_qubit: int,
        target_qubit: int,
        theta: float,
        phi: float,
        lambda_: float,
    ):
        """
        Applies Controlled-Generic gate to the target qubit.

        :param control_qubit: Index of the control qubit.
        :param target_qubit: Index of the target qubit.
        :param theta: Angle theta.
        :param phi: Angle phi.
        :param lambda_: Angle lambda.
        """
        U = Gates.U(theta, phi, lambda_)
        self._apply_gate("U", U, target_qubit, control_qubit)

    def measure(self, shots: int = 1, basis: Optional[np.ndarray] = None) -> List[str]:
        """
        Measures the state of the qubits.

        :param shots: Number of measurements.
        :param basis: Optional basis transformation.
        :return: List of measurement results.
        """
        state_vector = self.state_vector
        if basis is not None:
            state_vector = basis @ self.state_vector
        probabilities = np.abs(state_vector) ** 2
        result = np.random.choice(2**self.num_qubits, p=probabilities, size=shots)
        return [format(r, f"0{self.num_qubits}b") for r in result]

    def run(
        self, shots: int = 100, basis: Optional[np.ndarray] = None
    ) -> Dict[str, int]:
        """
        Runs the simulation and returns measurement results.

        :param shots: Number of measurements.
        :param basis: Optional basis transformation.
        :return: Dictionary of measurement results.
        """
        results = self.measure(shots, basis)
        return dict(Counter(results))

    def __str__(self) -> str:
        """
        Returns a string representation of the circuit.

        :return: String representing the circuit.
        """
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
