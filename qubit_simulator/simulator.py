import numpy as np
from .gates import Gates


class QubitSimulator:
    def __init__(self, num_qubits: int):
        self.n = num_qubits
        self._state = np.zeros(2**num_qubits, complex)
        self._state[0] = 1
        self._circuit = []

    def __sizeof__(self):
        # 24 = ~8*3 overhead for some ints
        return self._state.nbytes + sum(op.__sizeof__() for op in self._circuit) + 24

    def __repr__(self):
        return f"sim = QubitSimulator(num_qubits={self.n})\n" + "\n".join(
            f"sim.{g_name.lower()}({', '.join(map(str, (*pars[0], *qubits) if pars else qubits))})"
            for g_name, qubits, *pars in self._circuit
        )

    def reset(self):
        self._state[:] = 0
        self._state[0] = 1
        self._circuit.clear()

    def _apply_gate(self, U: np.ndarray, qubits: list):
        k = len(qubits)
        st = np.moveaxis(self._state.reshape((2,) * self.n), qubits, range(k))
        out = np.tensordot(U.reshape((2,) * (2 * k)), st, (range(k, 2 * k), range(k)))
        self._state = np.moveaxis(out, range(k), qubits).ravel()

    def apply_gate(
        self, gate: np.ndarray, qubits: list, name: str = "G", params: tuple = None
    ):
        self._apply_gate(gate, qubits)
        self._circuit.append((name, qubits, params) if params else (name, qubits))

    # Single-qubit gates
    def x(self, qubit: int):
        self.apply_gate(Gates.X, [qubit], "X")

    def y(self, qubit: int):
        self.apply_gate(Gates.Y, [qubit], "Y")

    def z(self, qubit: int):
        self.apply_gate(Gates.Z, [qubit], "Z")

    def h(self, qubit: int):
        self.apply_gate(Gates.H, [qubit], "H")

    def s(self, qubit: int):
        self.apply_gate(Gates.S, [qubit], "S")

    def t(self, qubit: int):
        self.apply_gate(Gates.T, [qubit], "T")

    def u(self, theta: float, phi: float, lam: float, qubit: int):
        self.apply_gate(Gates.U(theta, phi, lam), [qubit], "U", (theta, phi, lam))

    # Two-qubit gates
    def cx(self, control: int, target: int):
        self.apply_gate(Gates.controlled_gate(Gates.X), [control, target], "CX")

    def cu(self, theta: float, phi: float, lam: float, control: int, target: int):
        self.apply_gate(
            Gates.controlled_gate(Gates.U(theta, phi, lam)),
            [control, target],
            "CU",
            (theta, phi, lam),
        )

    def swap(self, qubit1: int, qubit2: int):
        self.apply_gate(Gates.SWAP_matrix(), [qubit1, qubit2], "SWAP")

    def iswap(self, qubit1: int, qubit2: int):
        self.apply_gate(Gates.iSWAP_matrix(), [qubit1, qubit2], "iSWAP")

    # Three-qubit gates
    def toffoli(self, control1: int, control2: int, target: int):
        self.apply_gate(Gates.Toffoli_matrix(), [control1, control2, target], "TOFFOLI")

    def fredkin(self, control: int, target1: int, target2: int):
        self.apply_gate(Gates.Fredkin_matrix(), [control, target1, target2], "FREDKIN")

    def run(self, shots: int = 100) -> dict:
        float_counts = shots * (np.abs(self._state) ** 2)
        base_counts = float_counts.astype(int)
        remainder = shots - base_counts.sum()
        if remainder:
            frac = float_counts - base_counts
            base_counts[np.argsort(frac)[-remainder:]] += 1
        return {f"{i:0{self.n}b}": int(c) for i, c in enumerate(base_counts) if c}

    def state(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import hsv_to_rgb, Normalize
        from matplotlib import cm

        mag = np.abs(self._state)
        ph = np.angle(self._state)  # Phase in range [-π, π]
        colors = hsv_to_rgb(
            np.column_stack(
                ((ph % (2 * np.pi)) / (2 * np.pi), np.ones_like(ph), np.ones_like(ph))
            )
        )
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar([f"{i:0{self.n}b}" for i in range(len(mag))], mag, color=colors)
        ax.set(
            xlabel="Basis state",
            ylabel="Amplitude magnitude",
            title=f"{self.n}-Qubit State",
        )
        ax.tick_params(axis="x", labelrotation=67)
        sm = cm.ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap="hsv")
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Phase (radians)")
        cbar.ax.set(
            yticks=[-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            yticklabels=["-π", "-π/2", "0", "π/2", "π"],
        )
        plt.tight_layout()
        plt.show()

    def draw(self):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from matplotlib import cm
        import hashlib

        fig, ax = plt.subplots(figsize=(max(8, len(self._circuit)), self.n + 1))
        for q in range(self.n):
            ax.hlines(q, -0.5, len(self._circuit) - 0.5, color="k")
        color = lambda name: cm.tab20(
            (int(hashlib.md5(name.encode()).hexdigest(), 36) % 313) / 313
        )
        cC = lambda x, y: ax.add_patch(Circle((x, y), 0.08, fc="k", zorder=3))
        xT = lambda x, y: (
            ax.add_patch(Circle((x, y), 0.18, fc="w", ec="k", zorder=3)),
            ax.plot(
                [x - 0.1, x + 0.1],
                [y - 0.1, y + 0.1],
                "k",
                [x - 0.1, x + 0.1],
                [y + 0.1, y - 0.1],
                "k",
                zorder=4,
            ),
        )
        box = lambda x, y, t, col: (
            ax.add_patch(
                Rectangle((x - 0.3, y - 0.3), 0.6, 0.6, fc=col, ec="k", zorder=3)
            ),
            ax.text(x, y, t, ha="center", va="center", zorder=4),
        )
        for i, gate_info in enumerate(self._circuit):
            if len(gate_info) == 3:
                g_name, qubits, pars = gate_info
                col = color(g_name + str(pars))
            else:
                g_name, qubits = gate_info
                pars = None
                col = color(g_name)
            if g_name in "XYZHST":
                box(i, qubits[0], g_name, col)
            elif g_name == "U":
                box(
                    i, qubits[0], f"U\n({pars[0]:.2g},{pars[1]:.2g},{pars[2]:.2g})", col
                )
            elif g_name in ("CX", "CU"):
                ax.vlines(i, min(qubits), max(qubits), color="k")
                cC(i, qubits[0])
                if g_name == "CX":
                    xT(i, qubits[1])
                else:
                    box(
                        i,
                        qubits[1],
                        f"U\n({pars[0]:.2g},{pars[1]:.2g},{pars[2]:.2g})",
                        col,
                    )
            elif g_name in ("SWAP", "iSWAP"):
                ax.vlines(i, min(qubits), max(qubits), color="k")
                xT(i, qubits[0])
                xT(i, qubits[1])
                if g_name == "iSWAP":
                    ax.text(i, sum(qubits) / 2, "i", ha="center", va="center", zorder=4)
            elif g_name == "TOFFOLI":
                ax.vlines(i, min(qubits), max(qubits), color="k")
                cC(i, qubits[0])
                cC(i, qubits[1])
                xT(i, qubits[2])
            elif g_name == "FREDKIN":
                ax.vlines(i, min(qubits), max(qubits), color="k")
                cC(i, qubits[0])
                xT(i, qubits[1])
                xT(i, qubits[2])
            else:
                box(i, qubits[0], g_name, col)
        for q in range(self.n):
            ax.text(-1, q, f"q{q}", ha="right", va="center")
        ax.set_xlim(-1, len(self._circuit))
        ax.set_ylim(-0.5, self.n - 0.5)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"{self.n}-Qubit Circuit")
        plt.tight_layout()
        plt.show()
