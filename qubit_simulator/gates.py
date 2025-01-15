import numpy as np


class Gates:
    """Minimal collection of common gates and helper methods."""

    # Single-qubit gates (2x2)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    Z = np.array([[1, 0], [0, -1]], complex)
    H = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    S = np.diag([1, 1j]).astype(complex)
    T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)

    @staticmethod
    def U(theta, phi, lam):
        """Parametrized single-qubit gate U(theta,phi,lambda)."""
        return np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lam)) * np.cos(theta / 2),
                ],
            ],
            complex,
        )

    # Two-qubit gates (4x4)
    @staticmethod
    def SWAP_matrix():
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], complex
        )

    @staticmethod
    def iSWAP_matrix():
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], complex
        )

    # Three-qubit gates (8x8)
    @staticmethod
    def Toffoli_matrix():
        # Flip the 3rd qubit if first two are |1>
        m = np.eye(8, complex)
        m[[6, 7], [6, 7]] = 0
        m[6, 7] = 1
        m[7, 6] = 1
        return m

    @staticmethod
    def Fredkin_matrix():
        # Swap the last two qubits if the first is |1>
        m = np.eye(8, complex)
        m[[5, 6], [5, 6]] = 0
        m[5, 6] = 1
        m[6, 5] = 1
        return m

    @staticmethod
    def inverse_gate(U):
        """Returns U^\dagger."""
        return U.conjugate().T

    @staticmethod
    def controlled_gate(U):
        """
        Builds a 4x4 controlled version of a 2x2 gate U:
          |0><0| ⊗ I + |1><1| ⊗ U
        """
        c = np.zeros((4, 4), complex)
        c[:2, :2] = np.eye(2)
        c[2:, 2:] = U
        return c
