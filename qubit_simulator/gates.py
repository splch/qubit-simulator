import numpy as np


class Gates:
    """
    A utility class containing standard quantum gate matrices (as NumPy arrays)
    and static methods for generating parametric gates, controlled gates,
    and multi-qubit gate matrices.
    """

    # Single-qubit gates (2x2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.diag([1, 1j]).astype(complex)
    T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)

    @staticmethod
    def U(theta: float, phi: float, lam: float) -> np.ndarray:
        """
        Create a single-qubit U(θ, φ, λ) gate matrix according to the standard
        Euler-angle parameterization.

        Args:
            theta (float): Rotation angle.
            phi (float): Phase angle for the rotation axis.
            lam (float): Additional phase angle.

        Returns:
            np.ndarray: A 2x2 complex matrix representing U(θ, φ, λ).
        """
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        e = np.exp
        return np.array(
            [[c, -e(1j * lam) * s], [e(1j * phi) * s, e(1j * (phi + lam)) * c]],
            dtype=complex,
        )

    # Two-qubit gates (4x4)
    @staticmethod
    def SWAP_matrix() -> np.ndarray:
        """
        Return the 4x4 matrix for the SWAP gate, which exchanges the states of two qubits.

        Returns:
            np.ndarray: A 4x4 complex SWAP gate matrix.
        """
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )

    @staticmethod
    def iSWAP_matrix() -> np.ndarray:
        """
        Return the 4x4 matrix for the iSWAP gate, which swaps the two qubits and
        introduces a phase of i for the |01> <-> |10> transitions.

        Returns:
            np.ndarray: A 4x4 complex iSWAP gate matrix.
        """
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
        )

    # Three-qubit gates (8x8)
    @staticmethod
    def Toffoli_matrix() -> np.ndarray:
        """
        Return the 8x8 matrix for the Toffoli (CCX) gate, which flips the target bit
        only if both control bits are 1.

        Returns:
            np.ndarray: An 8x8 complex matrix for the Toffoli gate.
        """
        m = np.eye(8, dtype=complex)
        # Swap the last two diagonal elements to apply the X on the 111 state
        m[[6, 7], [6, 7]] = 0
        m[6, 7] = 1
        m[7, 6] = 1
        return m

    @staticmethod
    def Fredkin_matrix() -> np.ndarray:
        """
        Return the 8x8 matrix for the Fredkin (CSWAP) gate, which swaps the two target
        bits if the control bit is 1.

        Returns:
            np.ndarray: An 8x8 complex matrix for the Fredkin gate.
        """
        m = np.eye(8, dtype=complex)
        # Swap indices 5 (101) and 6 (110) to reflect the conditional swap
        m[[5, 6], [5, 6]] = 0
        m[5, 6] = 1
        m[6, 5] = 1
        return m

    @staticmethod
    def inverse_gate(U: np.ndarray) -> np.ndarray:
        """
        Return the inverse of a given unitary gate U.

        Args:
            U (np.ndarray): A unitary matrix.

        Returns:
            np.ndarray: The inverse (conjugate transpose) of U.
        """
        return U.conjugate().T

    @staticmethod
    def controlled_gate(U: np.ndarray) -> np.ndarray:
        """
        Create a 4x4 controlled version of a 2x2 gate U. The resulting matrix
        applies U only when the control qubit is |1>.

        Args:
            U (np.ndarray): A 2x2 single-qubit gate matrix.

        Returns:
            np.ndarray: A 4x4 matrix representing the controlled gate.
        """
        c = np.zeros((4, 4), dtype=complex)
        c[:2, :2] = np.eye(2)
        c[2:, 2:] = U
        return c
