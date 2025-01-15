import numpy as np
from .gates import Gates


class QubitSimulator:
    """
    A simple statevector simulator for a given number of qubits.
    State is stored in a 1D NumPy array of length 2^n.
    """

    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        # Start in |0...0> state
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1.0

    def _apply_single_qubit_gate(self, gate, qubit):
        """
        Apply a 2x2 single-qubit gate to the specified qubit.
        """
        # We'll iterate in blocks of size 2^(qubit).
        # Within each block, we do a 2-element transform.
        step = 2**qubit
        for start in range(0, 2**self.num_qubits, 2 * step):
            for k in range(step):
                idx0 = start + k
                idx1 = start + k + step
                # Apply the 2x2 gate
                new0 = gate[0, 0] * self.state[idx0] + gate[0, 1] * self.state[idx1]
                new1 = gate[1, 0] * self.state[idx0] + gate[1, 1] * self.state[idx1]
                self.state[idx0] = new0
                self.state[idx1] = new1

    def _apply_two_qubit_gate(self, gate4x4, qubit0, qubit1):
        """
        Apply a 4x4 two-qubit gate where
        - qubit0 is the MOST significant bit
        - qubit1 is the LEAST significant bit
        in the 2-qubit subspace index.
        """
        new_state = np.zeros_like(self.state, dtype=complex)
        size = 2 ** self.num_qubits

        for i in range(size):
            # Extract the bits for each qubit
            b0 = (i >> qubit0) & 1  # MSB
            b1 = (i >> qubit1) & 1  # LSB

            # sub_i in [0..3], MSB = b0, LSB = b1
            sub_i = 2 * b0 + b1

            # Loop over all possible final subspace states
            for sub_f in range(4):
                # Decode final bits
                b0_f = (sub_f >> 1) & 1  # final MSB
                b1_f = (sub_f >> 0) & 1  # final LSB

                # Build final index j by replacing bits qubit0, qubit1 in i
                j = i
                # Clear the old bits
                j &= ~(1 << qubit0)
                j &= ~(1 << qubit1)
                # Set the new bits
                j |= (b0_f << qubit0)
                j |= (b1_f << qubit1)

                # Accumulate amplitude
                new_state[j] += gate4x4[sub_f, sub_i] * self.state[i]

        self.state = new_state

    def _apply_three_qubit_gate(self, gate8x8, qubits):
        """
        Apply an 8x8 three-qubit gate where:
        qubits[0] = MSB,
        qubits[1] = middle,
        qubits[2] = LSB
        """
        q0, q1, q2 = qubits
        new_state = np.zeros_like(self.state, dtype=complex)
        size = 2 ** self.num_qubits

        for i in range(size):
            b0 = (i >> q0) & 1  # MSB
            b1 = (i >> q1) & 1  # middle
            b2 = (i >> q2) & 1  # LSB

            # sub_i in [0..7]
            sub_i = 4 * b0 + 2 * b1 + b2

            # Loop over all possible final subspace states
            for sub_f in range(8):
                b0_f = (sub_f >> 2) & 1
                b1_f = (sub_f >> 1) & 1
                b2_f = (sub_f >> 0) & 1

                j = i
                # Clear old bits
                j &= ~(1 << q0)
                j &= ~(1 << q1)
                j &= ~(1 << q2)
                # Set new bits
                j |= (b0_f << q0)
                j |= (b1_f << q1)
                j |= (b2_f << q2)

                # Accumulate
                new_state[j] += gate8x8[sub_f, sub_i] * self.state[i]

        self.state = new_state

    # ------------------------------------------------------------------------
    # Public methods for applying gates by name
    # ------------------------------------------------------------------------
    def x(self, qubit):
        self._apply_single_qubit_gate(Gates.X, qubit)

    def y(self, qubit):
        self._apply_single_qubit_gate(Gates.Y, qubit)

    def z(self, qubit):
        self._apply_single_qubit_gate(Gates.Z, qubit)

    def h(self, qubit):
        self._apply_single_qubit_gate(Gates.H, qubit)

    def s(self, qubit):
        self._apply_single_qubit_gate(Gates.S, qubit)

    def t(self, qubit):
        self._apply_single_qubit_gate(Gates.T, qubit)

    def u(self, theta, phi, lam, qubit):
        mat = Gates.U(theta, phi, lam)
        self._apply_single_qubit_gate(mat, qubit)

    def cx(self, control, target):
        mat = Gates.controlled_gate(Gates.X)
        self._apply_two_qubit_gate(mat, control, target)

    def cu(self, theta, phi, lam, control, target):
        mat = Gates.controlled_gate(Gates.U(theta, phi, lam))
        self._apply_two_qubit_gate(mat, control, target)

    def swap(self, qubit1, qubit2):
        mat = Gates.SWAP_matrix()
        self._apply_two_qubit_gate(mat, qubit1, qubit2)

    def iswap(self, qubit1, qubit2):
        mat = Gates.iSWAP_matrix()
        self._apply_two_qubit_gate(mat, qubit1, qubit2)

    def toffoli(self, c1, c2, t):
        mat = Gates.Toffoli_matrix()
        self._apply_three_qubit_gate(mat, [c1, c2, t])

    def fredkin(self, c, t1, t2):
        mat = Gates.Fredkin_matrix()
        self._apply_three_qubit_gate(mat, [c, t1, t2])

    # ------------------------------------------------------------------------
    # Simulation & measurement
    # ------------------------------------------------------------------------
    def run(self, shots=1024):
        """
        Measure all qubits in the computational basis, repeated 'shots' times.
        Returns a dictionary of {'bitstring': count}.
        """
        probs = np.abs(self.state) ** 2
        outcomes = np.random.choice(2**self.num_qubits, size=shots, p=probs)
        counts = {}
        for out in outcomes:
            # Convert integer to bitstring (e.g., for 2 qubits -> '00', '01', '10', '11')
            bitstring = format(out, "0{}b".format(self.num_qubits))
            counts[bitstring] = counts.get(bitstring, 0) + 1
        return counts
