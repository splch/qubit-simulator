import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import Circle, Rectangle
from .gates import Gates


class QubitSimulator:
    """
    Statevector simulator using tensor operations
    to apply any k-qubit gate by appropriately reshaping both
    the gate and state.
    """

    def __init__(self, num_qubits: int):
        self.n = num_qubits
        # Initialize statevector of length 2^n to |0...0>
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0
        self._circuit = []

    def __sizeof__(self):
        return self.state.nbytes + sum(op.__sizeof__() for op in self._circuit) + 8 * 3

    def reset(self):
        """Reset the simulator to the all-|0> state."""
        self.state = np.zeros(2**self.n, dtype=complex)
        self.state[0] = 1.0
        self._circuit.clear()

    def _apply_gate(self, U: np.ndarray, qubits: list[int]):
        """
        Apply the gate U on the specified qubits using tensor operations.
        """
        k = len(qubits)  # number of qubits this gate acts on
        shapeU = (2,) * k + (2,) * k  # e.g. for 2-qubit gate => (2,2, 2,2)
        U_reshaped = U.reshape(shapeU)  # from (2^k,2^k) to (2,...,2,2,...,2)
        # Reshape state from (2^n,) -> (2, 2, ..., 2)
        st = self.state.reshape([2] * self.n)
        # Move the targeted qubits' axes to the front, so we contract over them
        st = np.moveaxis(st, qubits, range(k))
        # Tensordot over the last k dims of U with the first k dims of st
        #   - The last k axes of U_reshaped are the "input" axes
        #   - The first k axes of st are the qubits we apply the gate to
        st_out = np.tensordot(U_reshaped, st, axes=(range(k, 2 * k), range(k)))
        # st_out has k "output" axes in front, plus the other (n-k) axes
        # Move the front k axes back to their original positions
        st_out = np.moveaxis(st_out, range(k), qubits)
        # Flatten back to 1D
        self.state = st_out.ravel()

    # Single-qubit gates
    def x(self, q: int):
        self._apply_gate(Gates.X, [q])
        self._circuit.append(("X", [q]))

    def y(self, q: int):
        self._apply_gate(Gates.Y, [q])
        self._circuit.append(("Y", [q]))

    def z(self, q: int):
        self._apply_gate(Gates.Z, [q])
        self._circuit.append(("Z", [q]))

    def h(self, q: int):
        self._apply_gate(Gates.H, [q])
        self._circuit.append(("H", [q]))

    def s(self, q: int):
        self._apply_gate(Gates.S, [q])
        self._circuit.append(("S", [q]))

    def t(self, q: int):
        self._apply_gate(Gates.T, [q])
        self._circuit.append(("T", [q]))

    def u(self, theta: float, phi: float, lam: float, q: int):
        self._apply_gate(Gates.U(theta, phi, lam), [q])
        self._circuit.append(("U", [q], (theta, phi, lam)))

    # Two-qubit gates
    def cx(self, control: int, target: int):
        self._apply_gate(Gates.controlled_gate(Gates.X), [control, target])
        self._circuit.append(("CX", [control, target]))

    def cu(self, theta: float, phi: float, lam: float, control: int, target: int):
        self._apply_gate(
            Gates.controlled_gate(Gates.U(theta, phi, lam)), [control, target]
        )
        self._circuit.append(("CU", [control, target], (theta, phi, lam)))

    def swap(self, q1: int, q2: int):
        self._apply_gate(Gates.SWAP_matrix(), [q1, q2])
        self._circuit.append(("SWAP", [q1, q2]))

    def iswap(self, q1: int, q2: int):
        self._apply_gate(Gates.iSWAP_matrix(), [q1, q2])
        self._circuit.append(("iSWAP", [q1, q2]))

    # Three-qubit gates
    def toffoli(self, c1: int, c2: int, t: int):
        self._apply_gate(Gates.Toffoli_matrix(), [c1, c2, t])
        self._circuit.append(("TOFFOLI", [c1, c2, t]))

    def fredkin(self, c: int, t1: int, t2: int):
        self._apply_gate(Gates.Fredkin_matrix(), [c, t1, t2])
        self._circuit.append(("FREDKIN", [c, t1, t2]))

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
        """
        Plot the magnitudes and phases of the statevector.
        """
        mag = np.abs(self.state)
        ph = np.angle(self.state)  # in range [-pi, pi]
        colors = hsv_to_rgb(
            np.column_stack(
                (((ph % (2 * np.pi)) / (2 * np.pi)), np.ones(len(ph)), np.ones(len(ph)))
            )
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(len(mag)), mag, color=colors)
        ax.set(
            xlabel="Basis state (decimal)",
            ylabel="Amplitude magnitude",
            title=f"{self.n}-Qubit State",
        )
        # Create a colorbar for phase
        sm = plt.cm.ScalarMappable(cmap="hsv")
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Phase (radians mod 2π)")
        plt.tight_layout()
        plt.show()

    def draw(self, ax: plt.Axes = None, figsize: tuple[int, int] = None):
        """
        Draw a simple circuit diagram of the operations that were applied.
        """
        if ax is None:
            if not figsize:
                figsize = (max(8, len(self._circuit)), self.n + 1)
            fig, ax = plt.subplots(figsize=figsize)
        # Draw horizontal lines for each qubit wire
        for q in range(self.n):
            ax.hlines(q, -0.5, len(self._circuit) - 0.5, color="k")

        def cC(x, y):
            ax.add_patch(Circle((x, y), 0.08, fc="k", zorder=3))

        def xT(x, y):
            ax.add_patch(Circle((x, y), 0.18, fc="w", ec="k", zorder=3))
            ax.plot(
                [x - 0.1, x + 0.1],
                [y - 0.1, y + 0.1],
                "k",
                [x - 0.1, x + 0.1],
                [y + 0.1, y - 0.1],
                "k",
                zorder=4,
            )

        def box(x, y, t):
            ax.add_patch(
                Rectangle(
                    (x - 0.3, y - 0.3), 0.6, 0.6, fc="lightblue", ec="k", zorder=3
                )
            )
            ax.text(x, y, t, ha="center", va="center", zorder=4)

        # Render each gate in the circuit
        for i, (gate_name, qubits, *pars) in enumerate(self._circuit):
            if gate_name in "XYZHST":
                box(i, qubits[0], gate_name)
            elif gate_name == "U":
                theta, phi, lam = pars[0]
                box(i, qubits[0], f"U\n({theta:.2g},{phi:.2g},{lam:.2g})")
            elif gate_name in ("CX", "CU"):
                ax.vlines(i, *sorted(qubits), color="k")
                cC(i, qubits[0])
                if gate_name == "CX":
                    xT(i, qubits[1])
                else:
                    theta, phi, lam = pars[0]
                    box(i, qubits[1], f"U\n({theta:.2g},{phi:.2g},{lam:.2g})")
            elif gate_name in ("SWAP", "iSWAP"):
                ax.vlines(i, *sorted(qubits), color="k")
                xT(i, qubits[0])
                xT(i, qubits[1])
                if gate_name == "iSWAP":
                    ax.text(i, sum(qubits) / 2, "i", ha="center", va="center", zorder=4)
            elif gate_name == "TOFFOLI":
                ax.vlines(i, min(qubits), max(qubits), color="k")
                cC(i, qubits[0])
                cC(i, qubits[1])
                xT(i, qubits[2])
            elif gate_name == "FREDKIN":
                ax.vlines(i, min(qubits), max(qubits), color="k")
                cC(i, qubits[0])
                xT(i, qubits[1])
                xT(i, qubits[2])
            else:
                # Default fallback if a new gate name is added
                box(i, qubits[0], gate_name)
        # Label qubits
        for q in range(self.n):
            ax.text(-1, q, f"q{q}", ha="right", va="center")
        ax.set_xlim(-1, len(self._circuit))
        ax.set_ylim(-0.5, self.n - 0.5)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title("Circuit Diagram")
        plt.tight_layout()
        plt.show()
