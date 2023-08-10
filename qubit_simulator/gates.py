import numpy as np


# Identity gate
I = np.array([[1, 0], [0, 1]])

# Zero gate
O = np.zeros((2, 2))

# Hadamard gate
H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])

# T gate
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

# X gate
X = np.array([[0, 1], [1, 0]])
