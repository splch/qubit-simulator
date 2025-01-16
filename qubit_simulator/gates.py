import numpy as np


class Gates:
    # Single-qubit gates (2x2)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    S = np.diag([1, 1j]).astype(complex)
    T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)

    @staticmethod
    def U(theta: float, phi: float, lam: float) -> np.ndarray:
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        e = np.exp
        return np.array(
            [[c, -e(1j * lam) * s], [e(1j * phi) * s, e(1j * (phi + lam)) * c]],
            dtype=complex,
        )

    # Two-qubit gates (4x4)
    @staticmethod
    def SWAP_matrix() -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )

    @staticmethod
    def iSWAP_matrix() -> np.ndarray:
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
        )

    # Three-qubit gates (8x8)
    @staticmethod
    def Toffoli_matrix() -> np.ndarray:
        m = np.eye(8, dtype=complex)
        m[[6, 7], [6, 7]] = 0
        m[6, 7] = 1
        m[7, 6] = 1
        return m

    @staticmethod
    def Fredkin_matrix() -> np.ndarray:
        m = np.eye(8, dtype=complex)
        m[[5, 6], [5, 6]] = 0
        m[5, 6] = 1
        m[6, 5] = 1
        return m

    @staticmethod
    def inverse_gate(U: np.ndarray) -> np.ndarray:
        return U.conjugate().T

    @staticmethod
    def controlled_gate(U: np.ndarray) -> np.ndarray:
        c = np.zeros((4, 4), dtype=complex)
        c[:2, :2] = np.eye(2)
        c[2:, 2:] = U
        return c
