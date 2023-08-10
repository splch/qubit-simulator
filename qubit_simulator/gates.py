import numpy as np

# Pauli-X (NOT) gate
X = np.array([[0, 1], [1, 0]])

# Hadamard (H) gate
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

# Phase (S) gate
S = np.array([[1, 0], [0, 1j]])

# π/8 (T) gate
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

# Identity gate
I = np.array([[1, 0], [0, 1]])

# Zero gate
O = np.zeros((2, 2))
