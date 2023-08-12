# Qubit Simulator

Qubit Simulator is a simple and lightweight library that provides a quantum simulator for simulating qubits and quantum gates. It supports basic quantum operations and gates such as Hadamard, Pauli-X, Controlled-Not, and generic unitary transformations.

## Installation

Install Qubit Simulator via pip:

```bash
pip install qubit_simulator
```

## Usage

### Initializing the Simulator

Create a simulator with a specified number of qubits:

```python
from qubit_simulator import QubitSimulator

simulator = QubitSimulator(3)
```

### Applying Gates

Apply various quantum gates to the qubits:

```python
simulator.h(0)      # Hadamard gate
simulator.x(1)      # Pauli-X gate
simulator.cx(0, 1)  # Controlled-Not gate
```

### Custom Gates

Define and apply custom gates using angles:

```python
import numpy as np

theta, phi, lambda_ = np.pi/4, np.pi/3, np.pi/2
simulator.u(0, theta, phi, lambda_)  # Generic gate
```

### Measurements

Measure the state of the qubits:

```python
results = simulator.run(shots=100)
```

> {'100': 24, '110': 30, '000': 29, '010': 17}

### Circuit Representation

Get a string representation of the circuit:

```python
print(str(simulator))
```

```plaintext
-----------------
| H |   | C | U |
|   | X | X |   |
|   |   |   |   |
-----------------
```

## Testing

Tests are included in the package to verify its functionality.

## License

This project is licensed under the MIT License.
