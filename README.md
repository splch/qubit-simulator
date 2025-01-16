# Qubit Simulator

Qubit Simulator is a simple and lightweight library that provides a quantum statevector simulator for simulating qubits and quantum gates. It supports basic quantum operations and gates using NumPy.

## Installation

Install Qubit Simulator via pip:

```bash
pip install qubit-simulator
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
simulator.t(1)      # π/8 gate
simulator.cx(0, 2)  # Controlled-Not gate
```

### Custom Gates

Define and apply custom gates using angles:

```python
simulator.u(0.3, 0.4, 0.5, 2)  # Generic gate
```

### Measurements

Measure the state of the qubits:

```python
print(simulator.run(shots=100))
```

```plaintext
{'000': 46, '001': 4, '100': 4, '101': 46}
```

## Testing

Tests are included in the package to verify its functionality and provide more advanced examples:

```shell
python3 -m pytest tests/
```

## License

This project is licensed under the MIT License.
