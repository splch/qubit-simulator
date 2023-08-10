# QubitSimulator Library

## Overview

`QubitSimulator` is a simple and lightweight quantum computing simulator library implemented in Python. It provides basic functionalities for simulating quantum circuits with commonly used quantum gates, including Hadamard, Phase, $\frac{\pi}{8}$, and Controlled-NOT gates. You can also perform measurements on qubits and run the quantum circuit for multiple shots.

## Features

- [x] Universal gate set (CNOT, H, S, T)
- [ ] 

## Installation

```shell
pip install qubit_simulator
```

## Usage

### Initialization

Initialize the QubitSimulator with the number of qubits:

```python
simulator = QubitSimulator(2)
```

### Applying Gates

Apply the H, T, and CNOT gates to the specified qubits:

```python
simulator.H(0)
simulator.T(1)
simulator.CNOT(0, 1)
```

### Running the Circuit

Run the quantum circuit for a specified number of shots:

```python
simulator.run(num_shots=10)
```

> ['00', '11', '11', '00', '11', '11', '11', '00', '00', '11']

### Measurements

Perform a measurement on the qubits:

```python
simulator.measure()
```

> '11'

## Tests

```shell
python -m pytest
```

## License

This library is open-source and available under the MIT License.
