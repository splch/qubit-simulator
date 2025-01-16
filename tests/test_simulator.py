import pytest
import numpy as np
from qubit_simulator import QubitSimulator, Gates


# 1. Initialization Tests
def test_initial_state_vector_length():
    sim = QubitSimulator(num_qubits=3)
    assert len(sim.state) == 2**3, "State vector length should be 2^n."


def test_initial_state_is_zero_state():
    sim = QubitSimulator(num_qubits=3)
    expected = np.zeros(2**3, dtype=complex)
    expected[0] = 1.0
    assert np.allclose(sim.state, expected), "Initial state should be |000>."


# 2. Single-Qubit Gate Tests
def test_x_gate_on_single_qubit():
    sim = QubitSimulator(num_qubits=1)
    sim.x(0)
    expected = np.array([0, 1], dtype=complex)
    assert np.allclose(sim.state, expected), "X gate did not produce |1> from |0>."


def test_y_gate_on_single_qubit():
    sim = QubitSimulator(num_qubits=1)
    sim.y(0)
    expected = np.array([0, 1j], dtype=complex)  # i|1>
    assert np.allclose(sim.state, expected), "Y gate did not produce i|1> from |0>."


def test_z_gate_on_single_qubit():
    sim = QubitSimulator(num_qubits=1)
    sim.x(0)  # now |1>
    sim.z(0)  # => -|1>
    expected = np.array([0, -1], dtype=complex)
    assert np.allclose(sim.state, expected), "Z gate did not produce -|1>."


def test_h_gate_on_single_qubit():
    sim = QubitSimulator(num_qubits=1)
    sim.h(0)
    expected = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    assert np.allclose(
        sim.state, expected
    ), "H gate should produce (|0> + |1>)/sqrt(2)."


def test_h_gate_twice():
    sim = QubitSimulator(num_qubits=1)
    sim.h(0)
    sim.h(0)
    expected = np.array([1, 0], dtype=complex)
    assert np.allclose(sim.state, expected), "H applied twice should return to |0>."


def test_s_gate():
    sim = QubitSimulator(num_qubits=1)
    sim.x(0)  # |1>
    sim.s(0)  # => i|1>
    expected = np.array([0, 1j], dtype=complex)
    assert np.allclose(sim.state, expected), "S gate did not apply phase i to |1>."


def test_t_gate():
    sim = QubitSimulator(num_qubits=1)
    sim.x(0)  # |1>
    sim.t(0)  # => e^{i pi/4}|1>
    phase = np.exp(1j * np.pi / 4)
    expected = np.array([0, phase], dtype=complex)
    assert np.allclose(sim.state, expected), "T gate did not produce e^{i pi/4}|1>."


def test_u_gate():
    sim = QubitSimulator(num_qubits=1)
    sim.u(theta=np.pi, phi=0, lam=0, q=0)  # acts like X on |0> (up to global phase)
    expected = np.array([0, 1], dtype=complex)
    assert np.allclose(sim.state, expected), "U(π,0,0) not matching X action on |0>."


# 3. Two-Qubit Gate Tests
def test_cnot_control_zero():
    sim = QubitSimulator(num_qubits=2)
    # |00>, apply CNOT => still |00>
    sim.cx(0, 1)
    expected = np.array([1, 0, 0, 0], dtype=complex)
    assert np.allclose(sim.state, expected), "CNOT changed target when control was |0>."


def test_cnot_control_one():
    sim = QubitSimulator(num_qubits=2)
    sim.x(0)  # now |10>
    sim.cx(0, 1)  # => |11>
    expected = np.array([0, 0, 0, 1], dtype=complex)
    assert np.allclose(
        sim.state, expected
    ), "CNOT did not flip target when control was |1>."


def test_swap_gate():
    sim = QubitSimulator(num_qubits=2)
    sim.x(1)  # => |01>
    sim.swap(0, 1)  # => |10>
    expected = np.array([0, 0, 1, 0], dtype=complex)
    assert np.allclose(sim.state, expected), "SWAP did not swap |01> to |10>."


def test_iswap_gate():
    sim = QubitSimulator(num_qubits=2)
    sim.x(1)  # => |01>
    sim.iswap(0, 1)  # => i|10>
    expected = np.array([0, 0, 1j, 0], dtype=complex)
    assert np.allclose(sim.state, expected), "iSWAP did not produce i|10> from |01>."


def test_controlled_u_gate():
    sim = QubitSimulator(num_qubits=2)
    # control=0 => not activated => remains |00>
    sim.cu(theta=np.pi, phi=0, lam=0, control=0, target=1)
    expected = np.array([1, 0, 0, 0], dtype=complex)
    assert np.allclose(
        sim.state, expected
    ), "Controlled-U acted even though control was |0>."

    # Now put control in |1> => |10>, apply CU => acts like X on target => |11>
    sim = QubitSimulator(num_qubits=2)
    sim.x(0)  # => |10>
    sim.cu(theta=np.pi, phi=0, lam=0, control=0, target=1)
    expected = np.array([0, 0, 0, 1], dtype=complex)
    assert np.allclose(sim.state, expected), "Controlled-U didn't act like X on |10>."


# 4. Three-Qubit Gate Tests
def test_toffoli_gate():
    sim = QubitSimulator(num_qubits=3)
    sim.x(0)
    sim.x(1)  # => |110>
    sim.toffoli(0, 1, 2)  # => should flip target => |111>
    expected = np.zeros(8, dtype=complex)
    expected[7] = 1
    assert np.allclose(sim.state, expected), "Toffoli did not flip target for |110>."


def test_toffoli_partial_control():
    sim = QubitSimulator(num_qubits=3)
    sim.x(0)  # => |100>, only 1st ctrl is 1 => no flip => remains |100>
    sim.toffoli(0, 1, 2)
    expected = np.zeros(8, dtype=complex)
    expected[4] = 1
    assert np.allclose(
        sim.state, expected
    ), "Toffoli flipped target even though second ctrl was 0."


def test_fredkin_gate():
    # Fredkin = CSWAP: if ctrl=1 => swap qubits 1 & 2
    sim = QubitSimulator(num_qubits=3)
    sim.x(0)  # qubit0=1
    sim.x(2)  # => |101>
    sim.fredkin(0, 1, 2)  # => swap qubits 1 & 2 => |110>
    expected = np.zeros(8, dtype=complex)
    expected[6] = 1
    assert np.allclose(
        sim.state, expected
    ), "Fredkin did not swap qubits 1 & 2 correctly."


# 5. Simulator-Specific Tests (Measurement, Norm, etc.)
def test_apply_gate_then_inverse():
    """
    Use the simulator to apply a gate then its inverse,
    verifying the final state returns to |0>.
    """
    sim = QubitSimulator(num_qubits=1)
    sim.h(0)
    sim._apply_gate(Gates.inverse_gate(Gates.H), [0])
    expected = np.array([1, 0], dtype=complex)
    assert np.allclose(
        sim.state, expected
    ), "Applying H then H^† did not return to |0>."


def test_measurement_counts_no_superposition():
    sim = QubitSimulator(num_qubits=2)
    # => |00> by default
    shots = 200
    counts = sim.run(shots=shots)
    assert (
        counts.get("00", 0) == shots
    ), "All measurements should be '00' in a definite state."


@pytest.mark.parametrize("shots", [1000, 2000])
def test_measurement_distribution(shots):
    sim = QubitSimulator(num_qubits=1)
    sim.h(0)
    counts = sim.run(shots=shots)
    zero_counts = counts.get("0", 0)
    one_counts = counts.get("1", 0)
    # Expect roughly half 0, half 1
    assert abs(zero_counts - shots / 2) < 5 * np.sqrt(
        shots
    ), "Distribution off from 50% for '0'."
    assert abs(one_counts - shots / 2) < 5 * np.sqrt(
        shots
    ), "Distribution off from 50% for '1'."


def test_bell_state():
    sim = QubitSimulator(num_qubits=2)
    sim.h(0)
    sim.cx(0, 1)
    expected = np.array([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)], dtype=complex)
    assert np.allclose(sim.state, expected), "Bell state not correct."


def test_state_norm_is_one():
    sim = QubitSimulator(num_qubits=2)
    sim.h(0)
    sim.s(0)
    sim.cx(0, 1)
    norm = np.sum(np.abs(sim.state) ** 2)
    assert np.isclose(norm, 1.0), "State norm deviated from 1."

def test_reset():
    sim = QubitSimulator(num_qubits=2)
    sim.h(0)
    sim.s(0)
    sim.cx(0, 1)
    sim.reset()
    expected = np.array([1, 0, 0, 0], dtype=complex)
    assert np.allclose(sim.state, expected), "Reset did not return to |00>."

def test_sizeof():
    sim = QubitSimulator(num_qubits=2)
    assert sim.__sizeof__() == 2**2 * 16 + 24, "Size of simulator is not 2^2 * 16 + 24 bytes."
