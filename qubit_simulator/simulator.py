import numpy as np
from .gates import Gates


class QubitSimulator:
    """
    Simple statevector simulator using tensor operations to apply
    any k-qubit gate by appropriately reshaping both the gate and state.
    """

    def __init__(self, num_qubits):
        self.n = num_qubits
        # Statevector of length 2^n, start in |0...0>
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0

    def _apply_gate(self, U, qubits):
        """
        1) Reshape U into a tensor of shape (2,...,2, 2,...,2) with k inputs & k outputs
        2) Reshape self.state into (2,2,...,2)
        3) Move the 'qubits' axes to the front
        4) Use tensordot to contract
        5) Move the axes back and flatten
        """
        k = len(qubits)  # number of qubits this gate acts on
        shapeU = (2,) * k + (2,) * k  # e.g. for 2-qubit gate => (2,2, 2,2)
        U_reshaped = U.reshape(shapeU)  # from (2^k,2^k) to (2,...,2,2,...,2)

        st = self.state.reshape([2] * self.n)
        # Move the targeted qubits' axes to the front, so we contract over them
        st = np.moveaxis(st, qubits, range(k))

        # tensordot over the last k dims of U with the first k dims of st
        #   - The last k axes of U_reshaped are the "input" axes
        #   - The first k axes of st are the qubits we apply the gate to
        st_out = np.tensordot(U_reshaped, st, axes=(range(k, 2 * k), range(k)))

        # st_out now has k "output" axes in front, plus the other (n-k) axes
        # Move the front k axes back to their original positions
        st_out = np.moveaxis(st_out, range(k), qubits)

        # Flatten back to 1D
        self.state = st_out.ravel()

    # -- Single-qubit gates --
    def x(self, q):
        self._apply_gate(Gates.X, [q])

    def y(self, q):
        self._apply_gate(Gates.Y, [q])

    def z(self, q):
        self._apply_gate(Gates.Z, [q])

    def h(self, q):
        self._apply_gate(Gates.H, [q])

    def s(self, q):
        self._apply_gate(Gates.S, [q])

    def t(self, q):
        self._apply_gate(Gates.T, [q])

    def u(self, theta, phi, lam, q):
        self._apply_gate(Gates.U(theta, phi, lam), [q])

    # -- Two-qubit gates --
    def cx(self, control, target):
        self._apply_gate(Gates.controlled_gate(Gates.X), [control, target])

    def cu(self, theta, phi, lam, control, target):
        self._apply_gate(
            Gates.controlled_gate(Gates.U(theta, phi, lam)), [control, target]
        )

    def swap(self, q1, q2):
        self._apply_gate(Gates.SWAP_matrix(), [q1, q2])

    def iswap(self, q1, q2):
        self._apply_gate(Gates.iSWAP_matrix(), [q1, q2])

    # -- Three-qubit gates --
    def toffoli(self, c1, c2, t):
        self._apply_gate(Gates.Toffoli_matrix(), [c1, c2, t])

    def fredkin(self, c, t1, t2):
        self._apply_gate(Gates.Fredkin_matrix(), [c, t1, t2])

    # -- Simulation & measurement --
    def run(self, shots=1024):
        """
        Measure all qubits in the computational basis 'shots' times.
        Returns a dict of {'bitstring': count}.
        """
        probs = np.abs(self.state) ** 2
        outcomes = np.random.choice(2**self.n, p=probs, size=shots)
        counts = {}
        for out in outcomes:
            bitstring = format(out, "0{}b".format(self.n))
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
