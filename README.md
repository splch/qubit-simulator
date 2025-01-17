# Qubit Simulator

A simple yet flexible statevector-based quantum circuit simulator for Python. It supports common single-, two-, and three-qubit gates (including parameterized gates), measurement (shot-based sampling), state resetting, and basic circuit visualization.

## Features

- **Statevector Simulation**: Maintains a complex-valued statevector of size ( 2^n ).
- **Common Gates**: X, Y, Z, H, S, T, plus multi-qubit gates like CNOT, SWAP, Toffoli, Fredkin, etc.
- **Parameterized Gates**: General single-qubit rotation ( U(θ, φ, λ) ).
- **Controlled Gates**: Automatically construct controlled versions of single-qubit gates.
- **Circuit Visualization**: Generate a diagram of applied operations with `.draw()`.
- **Measurement**: Returns shot-based measurement outcomes from the final state.
- **Lightweight**: Only requires [NumPy](https://numpy.org). For plotting, install optional [matplotlib](https://matplotlib.org).

## Installation

Install Qubit Simulator via pip:

```bash
pip install qubit-simulator[visualization]
```

## Usage

### Initializing the Simulator

Create a simulator with a specified number of qubits:

```python
from qubit_simulator import QubitSimulator

sim = QubitSimulator(num_qubits=2)
```

### Applying Gates

Apply various quantum gates to the qubits:

```python
sim.h(0)      # Hadamard gate
sim.t(0)      # π/8 gate
sim.cx(0, 1)  # Controlled-Not gate
```

### Custom Gates

Define and apply custom gates using angles:

```python
sim.u(1.2, 3.4, 5.6, 1)  # Arbitrary single-qubit gate
```

### Circuit Drawing

Get a drawing of the circuit:

```python
sim.draw()
```

![Circuit Drawing](https://github.com/user-attachments/assets/2e6dbbc3-39e0-4d4f-8c43-c6f2ba83e121)

### Measurements

Measure the state of the qubits:

```python
print(sim.run(shots=100))
```

```plaintext
{'000': 49, '001': 1, '100': 1, '101': 49}
```

### Statevector Plot

Show the amplitude and phase of all quantum states:

```python
sim.state()
```

![Statevector Bar Chart](https://github.com/user-attachments/assets/f883b77f-1dc5-4236-8aed-0d67f8305e12)

## Testing

Tests are included in the package to verify its functionality and provide more advanced examples:

```shell
python3 -m pytest tests/
```

## License

This project is licensed under the MIT License.
