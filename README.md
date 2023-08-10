# QubitSimulator Library

## Overview

`QubitSimulator` is a simple and lightweight quantum computing simulator library implemented in Python. It provides basic functionalities for simulating quantum circuits with commonly used quantum gates, including Hadamard (H), T, and Controlled-NOT (CNOT) gates. You can also perform measurements on qubits and run the quantum circuit for multiple shots.

## Features

- **Hadamard (H) Gate:** Creates a superposition of states.
- **T Gate:** Applies a phase of \( \frac{\pi}{4} \).
- **Controlled-NOT (CNOT) Gate:** Applies a NOT operation to the target qubit if the control qubit is in state |1⟩.
- **Measure:** Measures the qubits and returns the result.
- **Run Method:** Executes the quantum circuit and returns the results for multiple shots.

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

Comprehensive tests are included to verify the correct implementation of the simulator. Refer to the test code for details on testing the functionality.

## License

This library is open-source and available under the MIT License.
