import pytest
import stim

import gen


def test_make_phenom_circuit_for_stabilizer_code():
    patch = gen.Patch(
        [
            gen.Tile(
                bases="Z",
                data_qubits=[0, 1, 1j, 1 + 1j],
                measure_qubit=0.5 + 0.5j,
            ),
            gen.Tile(
                bases="X",
                data_qubits=[0, 1],
                measure_qubit=0.5,
            ),
            gen.Tile(
                bases="X",
                data_qubits=[0 + 1j, 1 + 1j],
                measure_qubit=0.5 + 1j,
            ),
        ]
    )
    obs_x = gen.PauliMap({0: "X", 1j: "X"})
    obs_z = gen.PauliMap({0: "Z", 1: "Z"})

    assert gen.StabilizerCode(
        stabilizers=patch,
        logicals=[(obs_x, obs_z)],
    ).make_phenom_circuit(
        noise=gen.NoiseRule(flip_result=0.125, after={"DEPOLARIZE1": 0.25}),
        rounds=100,
    ) == stim.Circuit(
        """
        QUBIT_COORDS(0, -1) 0
        QUBIT_COORDS(0, 0) 1
        QUBIT_COORDS(0, 1) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        MPP X0*X1*X2 Z0*Z1*Z3 X1*X3 Z1*Z2*Z3*Z4 X2*X4
        OBSERVABLE_INCLUDE(0) rec[-5]
        OBSERVABLE_INCLUDE(1) rec[-4]
        DEPOLARIZE1(0.25) 1 2 3 4
        TICK
        REPEAT 100 {
            MPP(0.125) X1*X3 Z1*Z2*Z3*Z4 X2*X4
            DETECTOR(0.5, 0, 0) rec[-6] rec[-3]
            DETECTOR(0.5, 0.5, 0) rec[-5] rec[-2]
            DETECTOR(0.5, 1, 0) rec[-4] rec[-1]
            DEPOLARIZE1(0.25) 1 2 3 4
            SHIFT_COORDS(0, 0, 1)
            TICK
        }
        MPP X0*X1*X2 Z0*Z1*Z3 X1*X3 Z1*Z2*Z3*Z4 X2*X4
        OBSERVABLE_INCLUDE(0) rec[-5]
        OBSERVABLE_INCLUDE(1) rec[-4]
        DETECTOR(0.5, 0, 0) rec[-8] rec[-3]
        DETECTOR(0.5, 0.5, 0) rec[-7] rec[-2]
        DETECTOR(0.5, 1, 0) rec[-6] rec[-1]
        """
    )


def test_make_code_capacity_circuit_for_stabilizer_code():
    patch = gen.Patch(
        [
            gen.Tile(
                bases="Z",
                data_qubits=[0, 1, 1j, 1 + 1j],
                measure_qubit=0.5 + 0.5j,
            ),
            gen.Tile(
                bases="X",
                data_qubits=[0, 1],
                measure_qubit=0.5,
            ),
            gen.Tile(
                bases="X",
                data_qubits=[0 + 1j, 1 + 1j],
                measure_qubit=0.5 + 1j,
            ),
        ]
    )
    obs_x = gen.PauliMap({0: "X", 1j: "X"})
    obs_z = gen.PauliMap({0: "Z", 1: "Z"})

    assert gen.StabilizerCode(
        stabilizers=patch,
        logicals=[(obs_x, obs_z)],
    ).make_code_capacity_circuit(
        noise=gen.NoiseRule(after={"DEPOLARIZE1": 0.25}),
    ) == stim.Circuit(
        """
        QUBIT_COORDS(0, -1) 0
        QUBIT_COORDS(0, 0) 1
        QUBIT_COORDS(0, 1) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        MPP X0*X1*X2 Z0*Z1*Z3 X1*X3 Z1*Z2*Z3*Z4 X2*X4
        OBSERVABLE_INCLUDE(0) rec[-5]
        OBSERVABLE_INCLUDE(1) rec[-4]
        DEPOLARIZE1(0.25) 1 2 3 4
        TICK
        MPP X0*X1*X2 Z0*Z1*Z3 X1*X3 Z1*Z2*Z3*Z4 X2*X4
        OBSERVABLE_INCLUDE(0) rec[-5]
        OBSERVABLE_INCLUDE(1) rec[-4]
        DETECTOR(0.5, 0, 0) rec[-8] rec[-3]
        DETECTOR(0.5, 0.5, 0) rec[-7] rec[-2]
        DETECTOR(0.5, 1, 0) rec[-6] rec[-1]
        """
    )


def test_from_patch_with_inferred_observables():
    code = gen.StabilizerCode.from_patch_with_inferred_observables(
        gen.Patch(
            [
                gen.Tile(bases="XZZX", data_qubits=[0, 1, 2, 3], measure_qubit=0),
                gen.Tile(bases="XZZX", data_qubits=[1, 2, 3, 4], measure_qubit=1),
                gen.Tile(bases="XZZX", data_qubits=[2, 3, 4, 0], measure_qubit=2),
                gen.Tile(bases="XZZX", data_qubits=[3, 4, 0, 1], measure_qubit=3),
            ]
        )
    )
    code.verify()
    assert len(code.logicals) == 1
    assert len(code.logicals[0]) == 2


def test_verify_distance_is_at_least_3():
    distance_1_code = gen.StabilizerCode(
        stabilizers=gen.Patch(
            [
                gen.Tile(bases="XXXX", data_qubits=[0, 1, 2, 3]),
            ]
        ),
        logicals=[(gen.PauliMap(xs=[0, 1]), gen.PauliMap(zs=[0, 2]))],
    )
    with pytest.raises(ValueError, match="distance 1 error"):
        distance_1_code.verify_distance_is_at_least_2()
    with pytest.raises(ValueError, match="distance 1 error"):
        distance_1_code.verify_distance_is_at_least_3()

    distance_2_code = gen.StabilizerCode(
        stabilizers=gen.Patch(
            [
                gen.Tile(bases="XXXX", data_qubits=[0, 1, 2, 3]),
                gen.Tile(bases="ZZZZ", data_qubits=[0, 1, 2, 3]),
            ]
        ),
        logicals=[(gen.PauliMap(xs=[0, 1]), gen.PauliMap(zs=[0, 2]))],
    )
    distance_2_code.verify_distance_is_at_least_2()
    with pytest.raises(ValueError, match="distance 2 error"):
        distance_2_code.verify_distance_is_at_least_3()

    perfect_code = gen.StabilizerCode(
        stabilizers=gen.Patch(
            [
                gen.Tile(bases="XZZX", data_qubits=[0, 1, 2, 3]),
                gen.Tile(bases="XZZX", data_qubits=[1, 2, 3, 4]),
                gen.Tile(bases="XZZX", data_qubits=[2, 3, 4, 0]),
                gen.Tile(bases="XZZX", data_qubits=[3, 4, 0, 1]),
            ]
        ),
        logicals=[(gen.PauliMap(xs=[0, 1, 2, 3, 4]), gen.PauliMap(zs=[0, 1, 2, 3, 4]))],
    )
    perfect_code.verify_distance_is_at_least_2()
    perfect_code.verify_distance_is_at_least_3()


def test_with_integer_coordinates():
    code = gen.StabilizerCode(
        stabilizers=[
            gen.Tile(
                bases="XXXX",
                data_qubits=[0, 1, 1j, 1 + 1j],
                measure_qubit=1.5 + 0.5j,
            ),
            gen.Tile(
                bases="ZZZZ",
                data_qubits=[0, 1, 1j, 1 + 1j],
            ),
        ],
        logicals=[
            (gen.PauliMap(xs=[0, 1]), gen.PauliMap(zs=[0, 1j])),
            (gen.PauliMap(xs=[0, 1j]), gen.PauliMap(zs=[0, 1])),
        ],
    )
    code.verify()
    code2 = code.with_integer_coordinates()
    assert code2 == gen.StabilizerCode(
        stabilizers=[
            gen.Tile(
                bases="XXXX",
                data_qubits=[0, 1, 2j, 1 + 2j],
                measure_qubit=2 + 1j,
            ),
            gen.Tile(
                bases="ZZZZ",
                data_qubits=[0, 1, 2j, 1 + 2j],
            ),
        ],
        logicals=[
            (gen.PauliMap(xs=[0, 1]), gen.PauliMap(zs=[0, 2j])),
            (gen.PauliMap(xs=[0, 2j]), gen.PauliMap(zs=[0, 1])),
        ],
    )


def test_physical_to_logical():
    code = gen.StabilizerCode(
        stabilizers=[
            gen.Tile(
                bases="XXXX",
                data_qubits=[0, 1, 1j, 1 + 1j],
                measure_qubit=1.5 + 0.5j,
            ),
            gen.Tile(
                bases="ZZZZ",
                data_qubits=[0, 1, 1j, 1 + 1j],
            ),
        ],
        logicals=[
            (gen.PauliMap(xs=[0, 1]), gen.PauliMap(zs=[0, 1j])),
            (gen.PauliMap(xs=[0, 1j]), gen.PauliMap(zs=[0, 1])),
        ],
    )
    assert code.physical_to_logical(stim.PauliString("__")) == gen.PauliMap()
    assert code.physical_to_logical(stim.PauliString("X_")) == gen.PauliMap(
        {"X": [0, 1]}
    )
    assert code.physical_to_logical(stim.PauliString("_X")) == gen.PauliMap(
        {"X": [0, 1j]}
    )
    assert code.physical_to_logical(stim.PauliString("XX")) == gen.PauliMap(
        {"X": [1, 1j]}
    )
    assert code.physical_to_logical(stim.PauliString("Z_")) == gen.PauliMap(
        {"Z": [0, 1j]}
    )
    assert code.physical_to_logical(stim.PauliString("_Z")) == gen.PauliMap(
        {"Z": [0, 1]}
    )
    assert code.physical_to_logical(stim.PauliString("ZZ")) == gen.PauliMap(
        {"Z": [1, 1j]}
    )
    assert code.physical_to_logical(stim.PauliString("Y_")) == gen.PauliMap(
        {0: "Y", 1: "X", 1j: "Z"}
    )
    assert code.physical_to_logical(stim.PauliString("_Y")) == gen.PauliMap(
        {0: "Y", 1: "Z", 1j: "X"}
    )
    assert code.physical_to_logical(stim.PauliString("YY")) == gen.PauliMap(
        {1: "Y", 1j: "Y"}
    )
    assert code.physical_to_logical(stim.PauliString("XZ")) == gen.PauliMap(
        {0: "Y", 1: "Y"}
    )


def test_concat_over():
    a, b, c, d = [0, 1, 1j, 1 + 1j]
    code = gen.StabilizerCode(
        stabilizers=[
            gen.PauliMap(xs=[a, b, c, d]),
            gen.PauliMap(zs=[a, b, c, d]),
        ],
        logicals=[
            (gen.PauliMap(xs=[a, b]), gen.PauliMap(zs=[a, c])),
            (gen.PauliMap(zs=[a, b]), gen.PauliMap(xs=[a, c])),
        ],
    )
    code.verify()
    code2 = code.concat_over(code)
    code2.verify()
    assert code2.find_distance(max_search_weight=8) == 4
    assert len(code2.logicals) == len(code.logicals) * len(code.logicals)
    assert len(code2.stabilizers) == len(code.stabilizers) * len(code.logicals) + len(
        code.stabilizers
    ) * len(code.data_set)
    assert len(code2.data_set) == len(code.data_set) * len(code.data_set)
