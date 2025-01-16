import numpy as np
from qubit_simulator import Gates


def test_gate_shapes():
    """Check that the gates have the correct matrix dimensions."""
    # Single-qubit gates => 2x2
    for gate in [Gates.X, Gates.Y, Gates.Z, Gates.H, Gates.S, Gates.T]:
        assert gate.shape == (2, 2), f"Gate {gate} should be 2x2."

    # SWAP, iSWAP => 4x4
    assert Gates.SWAP_matrix().shape == (4, 4), "SWAP should be 4x4."
    assert Gates.iSWAP_matrix().shape == (4, 4), "iSWAP should be 4x4."

    # Toffoli, Fredkin => 8x8
    assert Gates.Toffoli_matrix().shape == (8, 8), "Toffoli should be 8x8."
    assert Gates.Fredkin_matrix().shape == (8, 8), "Fredkin should be 8x8."


def test_is_unitary_single_qubit():
    """Check that single-qubit gates are unitary (U * U^† = I)."""
    for gate in [Gates.X, Gates.Y, Gates.Z, Gates.H, Gates.S, Gates.T]:
        U_dagger = gate.conjugate().T
        identity = np.dot(gate, U_dagger)
        assert np.allclose(identity, np.eye(2, dtype=complex)), f"{gate} not unitary."


def test_is_unitary_two_qubit():
    """Check that two-qubit gates are unitary."""
    two_qubit_gates = [Gates.SWAP_matrix(), Gates.iSWAP_matrix()]
    for gate in two_qubit_gates:
        U_dagger = gate.conjugate().T
        identity = np.dot(gate, U_dagger)
        assert np.allclose(identity, np.eye(4, dtype=complex)), f"{gate} not unitary."


def test_is_unitary_three_qubit():
    """Check that three-qubit gates are unitary."""
    three_qubit_gates = [Gates.Toffoli_matrix(), Gates.Fredkin_matrix()]
    for gate in three_qubit_gates:
        U_dagger = gate.conjugate().T
        identity = np.dot(gate, U_dagger)
        assert np.allclose(identity, np.eye(8, dtype=complex)), f"{gate} not unitary."


def test_controlled_gate_shape():
    """Check that controlled_gate(U) is 4x4 if U is 2x2."""
    for gate in [Gates.X, Gates.Y, Gates.Z, Gates.H, Gates.S, Gates.T]:
        c_gate = Gates.controlled_gate(gate)
        assert c_gate.shape == (4, 4), "Controlled version of a 2x2 gate should be 4x4."


def test_controlled_gate_blocks():
    """Check that controlled_gate structure matches block-diagonal form: I (2x2) on top-left, U on bottom-right."""
    U = Gates.X  # any 2x2
    c_gate = Gates.controlled_gate(U)
    # top-left 2x2 should be identity
    assert np.allclose(c_gate[:2, :2], np.eye(2)), "Top-left block should be identity."
    # bottom-right 2x2 should be U
    assert np.allclose(
        c_gate[2:, 2:], U
    ), "Bottom-right block should be the original gate."


def test_inverse_gate_single_qubit():
    """Check that inverse_gate(U) = U^† for single-qubit gates."""
    for gate in [Gates.X, Gates.Y, Gates.Z, Gates.H, Gates.S, Gates.T]:
        U_inv = Gates.inverse_gate(gate)
        U_dagger = gate.conjugate().T
        assert np.allclose(U_inv, U_dagger), "inverse_gate should match U^†."


def test_U_parameterized_gate():
    """
    Check dimension & a known special case for U(θ, φ, λ).
    We'll use angles that produce X up to a global phase.
    """
    import math

    # 1) Basic shape check
    Umat = Gates.U(theta=math.pi / 2, phi=0, lam=0)
    assert Umat.shape == (2, 2), "Parameterized U gate should be 2x2."

    # 2) Produce X up to global phase with U(π, -π/2, π/2)
    U_x = Gates.U(theta=math.pi, phi=-math.pi / 2, lam=math.pi / 2)
    X = Gates.X

    # Compare up to a global phase
    # We'll pick a nonzero element to find the ratio
    ratio = U_x[0, 1] / X[0, 1]  # e.g., compare top-right
    adjusted = U_x * np.conjugate(ratio)
    assert np.allclose(
        adjusted, X, atol=1e-8
    ), "U(π, -π/2, π/2) not matching X up to a global phase."
