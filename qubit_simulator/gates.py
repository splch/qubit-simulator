import numpy as np

# Pauli-X (NOT) gate
X = np.array(
    [[0, 1], [1, 0]],
    dtype=complex,
)

# Hadamard (H) gate
H = (1 / np.sqrt(2)) * np.array(
    [[1, 1], [1, -1]],
    dtype=complex,
)

# π/8 (T) gate
T = np.array(
    [[1, 0], [0, np.exp(1j * np.pi / 4)]],
    dtype=complex,
)

# Generic gate
U = lambda theta, phi, lambda_: np.array(
    [
        [np.cos(theta / 2), -np.exp(1j * lambda_) * np.sin(theta / 2)],
        [
            np.exp(1j * phi) * np.sin(theta / 2),
            np.exp(1j * lambda_ + 1j * phi) * np.cos(theta / 2),
        ],
    ],
    dtype=complex,
)
