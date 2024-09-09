import stim

import gen


def test_builder_init():
    builder = gen.Builder.for_qubits([0, 1j, 3 + 2j])
    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        QUBIT_COORDS(3, 2) 2
        """
    )


def test_append_tick():
    builder = gen.Builder.for_qubits([0])
    builder.append("TICK")
    builder.append("TICK")
    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        TICK
        TICK
        """
    )


def test_append_shift_coords():
    builder = gen.Builder.for_qubits([0])
    builder.append("SHIFT_COORDS", arg=[0, 0, 1])
    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        SHIFT_COORDS(0, 0, 1)
        """
    )


def test_append_measurements():
    builder = gen.Builder.for_qubits(range(6))

    builder.append("MXX", [(2, 3)])
    assert builder.lookup_recs([(2, 3)]) == [0]
    assert builder.lookup_recs([(3, 2)]) == [0]

    builder.append("MYY", [(5, 4)])
    assert builder.lookup_recs([(4, 5)]) == [1]
    assert builder.lookup_recs([(5, 4)]) == [1]

    builder.append("M", [3])
    assert builder.lookup_recs([3]) == [2]


def test_append_measurements_canonical_order():
    builder = gen.Builder.for_qubits(range(6))

    builder.append("MX", [5, 2, 3])
    assert builder.lookup_recs([2]) == [0]
    assert builder.lookup_recs([3]) == [1]
    assert builder.lookup_recs([5]) == [2]

    builder.append("MZZ", [(5, 2), (3, 4)])
    assert builder.lookup_recs([(3, 4)]) == [3]
    assert builder.lookup_recs([(2, 5)]) == [4]

    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 0) 3
        QUBIT_COORDS(4, 0) 4
        QUBIT_COORDS(5, 0) 5
        MX 2 3 5
        MZZ 3 4 2 5
        """
    )


def test_append_mpp():
    builder = gen.Builder.for_qubits([2 + 3j, 5 + 7j, 11 + 13j])

    xxx = gen.PauliMap(xs=[2 + 3j, 5 + 7j, 11 + 13j])
    z_z = gen.PauliMap(zs=[11 + 13j, 2 + 3j])
    builder.append("MPP", [xxx, z_z])
    assert builder.lookup_recs([xxx]) == [0]
    assert builder.lookup_recs([z_z]) == [1]

    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(2, 3) 0
        QUBIT_COORDS(5, 7) 1
        QUBIT_COORDS(11, 13) 2
        MPP X0*X1*X2 Z0*Z2
        """
    )


def test_append_observable_include():
    builder = gen.Builder.for_qubits([2 + 3j, 5 + 7j, 11 + 13j])

    builder.append("R", [5 + 7j])
    builder.append("M", [2 + 3j, 5 + 7j, 11 + 13j], measure_key_func=lambda e: (e, "X"))
    builder.append("OBSERVABLE_INCLUDE", [(5 + 7j, "X")], arg=2)

    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(2, 3) 0
        QUBIT_COORDS(5, 7) 1
        QUBIT_COORDS(11, 13) 2
        R 1
        M 0 1 2
        OBSERVABLE_INCLUDE(2) rec[-2]
    """
    )


def test_append_detector():
    builder = gen.Builder.for_qubits([2 + 3j, 5 + 7j, 11 + 13j])

    builder.append("R", [5 + 7j])
    builder.append("M", [2 + 3j, 5 + 7j, 11 + 13j], measure_key_func=lambda e: (e, "X"))
    builder.append("DETECTOR", [(5 + 7j, "X")], arg=[2, 3, 5])

    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(2, 3) 0
        QUBIT_COORDS(5, 7) 1
        QUBIT_COORDS(11, 13) 2
        R 1
        M 0 1 2
        DETECTOR(2, 3, 5) rec[-2]
    """
    )
