import numpy as np
from .gates import Gates


class QubitSimulator:
    """
    Simulates an n-qubit quantum system, including state evolution under various
    quantum gates and the ability to measure outcomes over many shots. Also
    provides a record (circuit) of all applied gates, and utilities to visualize
    the circuit or the final state.
    """

    def __init__(self, num_qubits: int):
        """
        Initialize the QubitSimulator with a specified number of qubits.

        Args:
            num_qubits (int): The number of qubits in the system.

        Attributes:
            n (int): The number of qubits.
            _state (np.ndarray): A 1D complex NumPy array of length 2^n holding
                the state vector. Defaults to |0...0> at index 0.
            _circuit (list): A list of tuples describing each gate application.
                Tuples are typically (gate_name, [qubit_indices], params?).
        """
        self.n = num_qubits
        # Allocate space for the state vector; initially, all amplitude in |0...0>.
        self._state = np.zeros(2**num_qubits, complex)
        self._state[0] = 1
        # Keep track of the gates applied as a simple list of tuples.
        self._circuit = []

    def __sizeof__(self):
        """
        Return an approximate size of this simulator object in bytes.
        """
        return (
            self._state.__sizeof__()
            + self.n.__sizeof__()
            + sum(op.__sizeof__() for op in self._circuit)
            + object.__sizeof__(self)
        )

    def __repr__(self):
        """
        Produce a string that, when printed, displays the simulator's initialization
        plus each gate call in a Pythonic syntax. Useful for debugging or replicating
        the circuit programmatically.
        """
        return f"sim = QubitSimulator(num_qubits={self.n})\n" + "\n".join(
            f"sim.{g_name.lower()}({', '.join(map(str, (*pars[0], *qubits) if pars else qubits))})"
            for g_name, qubits, *pars in self._circuit
        )

    def reset(self):
        """
        Reset the simulator's state back to |0...0> and clear the recorded circuit.
        """
        # Zero out the state vector
        self._state[:] = 0
        # Re-initialize to |0...0>
        self._state[0] = 1
        # Clear the gate application history
        self._circuit.clear()

    def _apply_gate(self, U: np.ndarray, qubits: list):
        """
        Internal helper method to apply a unitary gate U to specified qubits
        in the state vector.

        Args:
            U (np.ndarray): A 2^k x 2^k gate matrix for k target qubits.
            qubits (list): Indices of qubits to which the gate is applied.
        """
        k = len(qubits)
        # Reshape state into an n-dimensional tensor of shape (2, 2, ..., 2).
        st = np.moveaxis(self._state.reshape((2,) * self.n), qubits, range(k))
        # Perform tensor contraction of U with the selected qubits.
        out = np.tensordot(U.reshape((2,) * (2 * k)), st, (range(k, 2 * k), range(k)))
        # Restore dimension ordering and flatten back to a 1D state vector.
        self._state = np.moveaxis(out, range(k), qubits).ravel()

    def apply_gate(
        self, gate: np.ndarray, qubits: list, name: str = "G", params: tuple = None
    ):
        """
        Apply a gate to specified qubits and record it in the circuit.

        Args:
            gate (np.ndarray): The gate matrix to apply (of shape 2^k x 2^k).
            qubits (list): The target qubit indices.
            name (str, optional): A short label for the gate (e.g. 'X', 'H', 'U').
            params (tuple, optional): Parameter values if this is a parametric gate.
        """
        # Update the simulator state
        self._apply_gate(gate, qubits)
        # Record the operation in the circuit for future reference or visualization
        self._circuit.append((name, qubits, params) if params else (name, qubits))

    # Single-qubit gates
    def x(self, qubit: int):
        """
        Apply the Pauli-X gate to a single qubit.
        """
        self.apply_gate(Gates.X, [qubit], "X")

    def y(self, qubit: int):
        """
        Apply the Pauli-Y gate to a single qubit.
        """
        self.apply_gate(Gates.Y, [qubit], "Y")

    def z(self, qubit: int):
        """
        Apply the Pauli-Z gate to a single qubit.
        """
        self.apply_gate(Gates.Z, [qubit], "Z")

    def h(self, qubit: int):
        """
        Apply the Hadamard gate to a single qubit.
        """
        self.apply_gate(Gates.H, [qubit], "H")

    def s(self, qubit: int):
        """
        Apply the S (phase) gate to a single qubit.
        """
        self.apply_gate(Gates.S, [qubit], "S")

    def t(self, qubit: int):
        """
        Apply the T (π/4 phase) gate to a single qubit.
        """
        self.apply_gate(Gates.T, [qubit], "T")

    def u(self, theta: float, phi: float, lam: float, qubit: int):
        """
        Apply the generic single-qubit U(θ, φ, λ) gate to a single qubit.

        Args:
            theta (float): Rotation angle around the Bloch sphere's axis.
            phi (float): Phase angle for the rotation axis.
            lam (float): Additional phase angle.
            qubit (int): The target qubit.
        """
        self.apply_gate(Gates.U(theta, phi, lam), [qubit], "U", (theta, phi, lam))

    # Two-qubit gates
    def cx(self, control: int, target: int):
        """
        Apply a controlled-X (CNOT) gate with one control and one target qubit.

        Args:
            control (int): The index of the control qubit.
            target (int): The index of the target qubit.
        """
        self.apply_gate(Gates.controlled_gate(Gates.X), [control, target], "CX")

    def cu(self, theta: float, phi: float, lam: float, control: int, target: int):
        """
        Apply a controlled-U(θ, φ, λ) gate with one control and one target qubit.

        Args:
            theta (float): Rotation angle for the single-qubit U gate.
            phi (float): Phase angle for the rotation axis.
            lam (float): Additional phase angle.
            control (int): The index of the control qubit.
            target (int): The index of the target qubit.
        """
        self.apply_gate(
            Gates.controlled_gate(Gates.U(theta, phi, lam)),
            [control, target],
            "CU",
            (theta, phi, lam),
        )

    def swap(self, qubit1: int, qubit2: int):
        """
        Apply a SWAP gate to exchange the states of two qubits.

        Args:
            qubit1 (int): The index of the first qubit.
            qubit2 (int): The index of the second qubit.
        """
        self.apply_gate(Gates.SWAP_matrix(), [qubit1, qubit2], "SWAP")

    def iswap(self, qubit1: int, qubit2: int):
        """
        Apply an iSWAP gate, which swaps the two qubits and introduces a phase of i.

        Args:
            qubit1 (int): The index of the first qubit.
            qubit2 (int): The index of the second qubit.
        """
        self.apply_gate(Gates.iSWAP_matrix(), [qubit1, qubit2], "iSWAP")

    # Three-qubit gates
    def toffoli(self, control1: int, control2: int, target: int):
        """
        Apply a Toffoli (CCX) gate with two control qubits and one target qubit.

        Args:
            control1 (int): The index of the first control qubit.
            control2 (int): The index of the second control qubit.
            target (int): The index of the target qubit.
        """
        self.apply_gate(Gates.Toffoli_matrix(), [control1, control2, target], "TOFFOLI")

    def fredkin(self, control: int, target1: int, target2: int):
        """
        Apply a Fredkin (CSWAP) gate with one control qubit and two target qubits.

        Args:
            control (int): The index of the control qubit.
            target1 (int): The index of the first target qubit.
            target2 (int): The index of the second target qubit.
        """
        self.apply_gate(Gates.Fredkin_matrix(), [control, target1, target2], "FREDKIN")

    # Measurement and visualization
    def run(self, shots: int = 100) -> dict:
        """
        Simulate measurement in the computational basis for a specified number
        of shots. The final result is a dictionary mapping 'basis_state' -> count.

        Args:
            shots (int, optional): The number of measurements to simulate. Defaults to 100.

        Returns:
            dict: Keys are binary strings (e.g. '010') representing measured states,
                  values are integer counts out of 'shots' for how often each state appeared.
        """
        # Expected counts = probabilities * shots, which may not be integer.
        float_counts = shots * (np.abs(self._state) ** 2)
        base_counts = float_counts.astype(int)
        remainder = shots - base_counts.sum()

        # Distribute leftover shots to states with largest fractional part.
        if remainder:
            frac = float_counts - base_counts
            base_counts[np.argsort(frac)[-remainder:]] += 1

        return {f"{i:0{self.n}b}": int(c) for i, c in enumerate(base_counts) if c}

    def state(self):
        """
        Plot the magnitude and phase of the current state vector as a bar chart.
        Each bar is labeled by its basis state, and colored by its phase.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import hsv_to_rgb, Normalize
        from matplotlib import cm

        mag = np.abs(self._state)  # Magnitudes of the amplitudes
        ph = np.angle(self._state)  # Phases in [-π, π]

        # Convert phase to an HSV color, then to RGB
        colors = hsv_to_rgb(
            np.column_stack(
                ((ph % (2 * np.pi)) / (2 * np.pi), np.ones_like(ph), np.ones_like(ph))
            )
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        # Label each bar by the basis state index in binary
        ax.bar([f"{i:0{self.n}b}" for i in range(len(mag))], mag, color=colors)
        ax.set(
            xlabel="Basis state",
            ylabel="Amplitude magnitude",
            title=f"{self.n}-Qubit State",
        )
        ax.tick_params(axis="x", labelrotation=67)

        # Add a color bar to show phase in radians
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
        """
        Draw a simple schematic of the circuit showing each qubit as a horizontal line
        and each applied gate at the appropriate location. Control qubits use a filled
        circle, targets for an X gate use a circle with an 'X', parametric gates use
        text boxes, etc.
        """
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle, Rectangle
        from matplotlib import cm
        import hashlib

        # Create a figure wide enough to hold all gates, plus some vertical space for qubits.
        fig, ax = plt.subplots(figsize=(max(8, len(self._circuit)), self.n + 1))

        # Draw horizontal lines for each qubit at y = 0,1,2,...,n-1
        for q in range(self.n):
            ax.hlines(q, -0.5, len(self._circuit) - 0.5, color="k")

        # A function to consistently map gate names to color
        color = lambda name: cm.tab20(
            (int(hashlib.md5(name.encode()).hexdigest(), 36) % 313) / 313
        )

        # Helper lambdas to draw small shapes:
        cC = lambda x, y: ax.add_patch(
            Circle((x, y), 0.08, fc="k", zorder=3)
        )  # control
        xT = lambda x, y: (
            ax.add_patch(
                Circle((x, y), 0.18, fc="w", ec="k", zorder=3)
            ),  # target circle
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

        # Iterate through each gate in the circuit and draw it
        for i, gate_info in enumerate(self._circuit):
            if len(gate_info) == 3:
                g_name, qubits, pars = gate_info
                col = color(g_name + str(pars))
            else:
                g_name, qubits = gate_info
                pars = None
                col = color(g_name)

            # Single-qubit gates
            if g_name in "XYZHST":
                box(i, qubits[0], g_name, col)

            elif g_name == "U":
                # Parametric single-qubit gate
                box(
                    i, qubits[0], f"U\n({pars[0]:.2g},{pars[1]:.2g},{pars[2]:.2g})", col
                )

            # Controlled gates
            elif g_name in ("CX", "CU"):
                # Draw vertical line between control and target qubit lines
                ax.vlines(i, min(qubits), max(qubits), color="k")
                cC(i, qubits[0])  # control circle
                if g_name == "CX":
                    xT(i, qubits[1])  # target X symbol
                else:
                    # Parametric target
                    box(
                        i,
                        qubits[1],
                        f"U\n({pars[0]:.2g},{pars[1]:.2g},{pars[2]:.2g})",
                        col,
                    )

            elif g_name in ("SWAP", "iSWAP"):
                # Draw vertical line
                ax.vlines(i, min(qubits), max(qubits), color="k")
                # Two X targets for each qubit
                xT(i, qubits[0])
                xT(i, qubits[1])
                if g_name == "iSWAP":
                    # Label the center with an 'i'
                    ax.text(i, sum(qubits) / 2, "i", ha="center", va="center", zorder=4)

            elif g_name == "TOFFOLI":
                # Two controls (circles), one target (X)
                ax.vlines(i, min(qubits), max(qubits), color="k")
                cC(i, qubits[0])
                cC(i, qubits[1])
                xT(i, qubits[2])

            elif g_name == "FREDKIN":
                # Control qubit plus two X's for the swap
                ax.vlines(i, min(qubits), max(qubits), color="k")
                cC(i, qubits[0])
                xT(i, qubits[1])
                xT(i, qubits[2])

            else:
                # Fallback for gates not explicitly handled above
                box(i, qubits[0], g_name, col)

        # Label qubits on the left
        for q in range(self.n):
            ax.text(-1, q, f"q{q}", ha="right", va="center")

        # Configure axis/appearance
        ax.set_xlim(-1, len(self._circuit))
        ax.set_ylim(-0.5, self.n - 0.5)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(f"{self.n}-Qubit Circuit")
        plt.tight_layout()
        plt.show()
