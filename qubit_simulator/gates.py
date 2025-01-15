import numpy as np


class Gates:
    """Collection of common quantum gate matrices and helper methods."""

    # --- Single-qubit gates (2x2) ---
    X = np.array([[0, 1], [1, 0]], dtype=complex)

    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)

    Z = np.array([[1, 0], [0, -1]], dtype=complex)

    H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)

    S = np.array([[1, 0], [0, 1j]], dtype=complex)  # Phase = pi/2

    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)  # Phase = pi/4

    @staticmethod
    def U(theta, phi, lam):
        """
        General single-qubit U gate parametrized by
        U(theta, phi, lambda):
            U|0> = cos(theta/2)|0> + e^{i*lambda} sin(theta/2)|1>
            U|1> = e^{i*phi}(-sin(theta/2))|0> + e^{i(phi+lambda)}cos(theta/2)|1>
        """
        return np.array(
            [
                [np.cos(theta / 2), -np.exp(1j * lam) * np.sin(theta / 2)],
                [
                    np.exp(1j * phi) * np.sin(theta / 2),
                    np.exp(1j * (phi + lam)) * np.cos(theta / 2),
                ],
            ],
            dtype=complex,
        )

    # --- Two-qubit gates (4x4) ---
    @staticmethod
    def SWAP_matrix():
        """
        4x4 SWAP gate: swaps the states of two qubits.
        """
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )

    @staticmethod
    def iSWAP_matrix():
        """
        4x4 iSWAP gate: swaps the states of two qubits and
        adds a phase of i to the |01> -> |10> and |10> -> |01> transitions.
        """
        return np.array(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
        )

    # --- Three-qubit gates (8x8) ---
    @staticmethod
    def Toffoli_matrix():
        """
        8x8 Toffoli (CCNOT) gate:
        flips the 3rd qubit (target) if first two qubits (controls) are |1>.
        Ordering: qubits = (control1, control2, target).
        Basis: |000>, |001>, |010>, |011>, |100>, |101>, |110>, |111>.
        """
        mat = np.eye(8, dtype=complex)
        # The only flip is between |110> (6) and |111> (7).
        mat[6, 6] = 0
        mat[7, 7] = 0
        mat[6, 7] = 1
        mat[7, 6] = 1
        return mat

    @staticmethod
    def Fredkin_matrix():
        """
        8x8 Fredkin (CSWAP) gate:
        swaps the last two qubits if the first qubit is |1>.
        Ordering: (control, target1, target2).
        """
        mat = np.eye(8, dtype=complex)
        # Indices in binary:
        # |101> (5) <-> |110> (6) get swapped if control=1 => states |101> <-> |110>.
        mat[5, 5], mat[6, 6] = 0, 0
        mat[5, 6], mat[6, 5] = 1, 1
        return mat

    @staticmethod
    def inverse_gate(U):
        """
        Returns the inverse (adjoint) of gate U.
        For a unitary U, the inverse is U^\dagger (the conjugate transpose).
        """
        return U.conjugate().T

    @staticmethod
    def controlled_gate(U):
        """
        Convert a single-qubit gate U (2x2) into its controlled 2-qubit version (4x4).
        Top-left block = I (when control is 0), bottom-right block = U (when control is 1).
        """
        # We assume U is 2x2.
        controlled = np.zeros((4, 4), dtype=complex)
        # The "control = 0" subspace gets Identity
        controlled[0:2, 0:2] = np.eye(2, dtype=complex)
        # The "control = 1" subspace gets U
        controlled[2:4, 2:4] = U
        return controlled
