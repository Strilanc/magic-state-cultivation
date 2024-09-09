from gen._layers._data import (
    single_qubit_clifford_multiplication_table,
    gate_to_unsigned_pauli_change,
    gate_to_unsigned_pauli_change_inverse,
)


def test_single_qubit_clifford_multiplication_table():
    v = single_qubit_clifford_multiplication_table()
    assert len(v) == 24 * 24


def test_gate_to_unsigned_pauli_change():
    m = gate_to_unsigned_pauli_change()
    assert m["H"] == {"X": "Z", "Z": "X", "Y": "Y"}
    assert m["C_XYZ"] == {"X": "Y", "Y": "Z", "Z": "X"}
    assert m["C_ZYX"] == {"X": "Z", "Z": "Y", "Y": "X"}
    assert m["X*C_ZYX"] == {"X": "Z", "Z": "Y", "Y": "X"}

    m = gate_to_unsigned_pauli_change_inverse()
    assert m["H"] == {"X": "Z", "Z": "X", "Y": "Y"}
    assert m["C_ZYX"] == {"X": "Y", "Y": "Z", "Z": "X"}
    assert m["C_XYZ"] == {"X": "Z", "Z": "Y", "Y": "X"}
    assert m["X*C_XYZ"] == {"X": "Z", "Z": "Y", "Y": "X"}
