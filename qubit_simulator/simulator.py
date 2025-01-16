import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from .gates import Gates


class QubitSimulator:
    """
    Simple statevector simulator using tensor operations to apply
    any k-qubit gate by appropriately reshaping both the gate and state.
    """

    def __init__(self, num_qubits: int):
        self.n = num_qubits
        # Statevector of length 2^n, start in |0...0>
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0

    def __sizeof__(self):
        return self.state.nbytes + 8  # 8 bytes for the int n

    def reset(self):
        self.state = np.zeros(2**self.n, dtype=complex)
        self.state[0] = 1.0

    def _apply_gate(self, U: np.ndarray, qubits: list):
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

    # Single-qubit gates
    def x(self, q: int):
        self._apply_gate(Gates.X, [q])

    def y(self, q: int):
        self._apply_gate(Gates.Y, [q])

    def z(self, q: int):
        self._apply_gate(Gates.Z, [q])

    def h(self, q: int):
        self._apply_gate(Gates.H, [q])

    def s(self, q: int):
        self._apply_gate(Gates.S, [q])

    def t(self, q: int):
        self._apply_gate(Gates.T, [q])

    def u(self, theta: float, phi: float, lam: float, q: int):
        self._apply_gate(Gates.U(theta, phi, lam), [q])

    # Two-qubit gates
    def cx(self, control: int, target: int):
        self._apply_gate(Gates.controlled_gate(Gates.X), [control, target])

    def cu(self, theta: float, phi: float, lam: float, control: int, target: int):
        self._apply_gate(
            Gates.controlled_gate(Gates.U(theta, phi, lam)), [control, target]
        )

    def swap(self, q1: int, q2: int):
        self._apply_gate(Gates.SWAP_matrix(), [q1, q2])

    def iswap(self, q1: int, q2: int):
        self._apply_gate(Gates.iSWAP_matrix(), [q1, q2])

    # Three-qubit gates
    def toffoli(self, c1: int, c2: int, t: int):
        self._apply_gate(Gates.Toffoli_matrix(), [c1, c2, t])

    def fredkin(self, c: int, t1: int, t2: int):
        self._apply_gate(Gates.Fredkin_matrix(), [c, t1, t2])

    # Simulation & measurement
    def run(self, shots: int = 100) -> dict[str, int]:
        # Compute base counts by multiplying probabilities and truncating
        float_counts = shots * (np.abs(self.state) ** 2)
        base_counts = float_counts.astype(int)
        remainder = shots - base_counts.sum()
        if remainder:
            # Distribute leftover shots to states with the largest fractional parts
            fractional_parts = float_counts - base_counts
            base_counts[np.argsort(fractional_parts)[-remainder:]] += 1
        # Return only those states that actually occurred
        return {f"{i:0{self.n}b}": int(c) for i, c in enumerate(base_counts) if c}

    def plot_state(self):
        mag, ph = np.abs(self.state), np.angle(self.state)
        cols = hsv_to_rgb(
            np.column_stack(
                ((ph % (2 * np.pi)) / (2 * np.pi), np.ones(len(ph)), np.ones(len(ph)))
            )
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(mag)), mag, color=cols)
        ax.set(
            xlabel="Basis state (decimal)",
            ylabel="Amplitude magnitude",
            title=f"{self.n}-Qubit State",
        )
        cb = plt.colorbar(plt.cm.ScalarMappable(cmap="hsv"), ax=ax)
        cb.set_label("Phase (radians mod 2π)")
        plt.show()
