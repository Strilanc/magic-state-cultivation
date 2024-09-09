import pytest
import stim

import gen


def test_circuit_with_xz_flipped():
    assert (
        gen.circuit_with_xz_flipped(
            stim.Circuit(
                """
                CX 0 1 2 3
                TICK
                H 0
                TICK
                REPEAT 10 {
                    MXX 0 1
                }
                """
            )
        )
        == stim.Circuit(
            """
            XCZ 0 1 2 3
            TICK
            H 0
            TICK
            REPEAT 10 {
                MZZ 0 1
            }
            """
        )
    )


def test_gates_used_by_circuit():
    assert (
        gen.gates_used_by_circuit(
            stim.Circuit(
                """
                H 0
                TICK
                CX 0 1
                """
            )
        )
        == {"H", "TICK", "CX"}
    )

    assert (
        gen.gates_used_by_circuit(
            stim.Circuit(
                """
                S 0
                XCZ 0 1
                """
            )
        )
        == {"S", "XCZ"}
    )

    assert (
        gen.gates_used_by_circuit(
            stim.Circuit(
                """
                MPP X0*X1 Z2*Z3*Z4 Y0*Z1
                """
            )
        )
        == {"MXX", "MZZZ", "MYZ"}
    )

    assert (
        gen.gates_used_by_circuit(
            stim.Circuit(
                """
                CX rec[-1] 1
                """
            )
        )
        == {"feedback"}
    )

    assert (
        gen.gates_used_by_circuit(
            stim.Circuit(
                """
                CX sweep[1] 1
                """
            )
        )
        == {"sweep"}
    )

    assert (
        gen.gates_used_by_circuit(
            stim.Circuit(
                """
                CX rec[-1] 1 0 1
                """
            )
        )
        == {"feedback", "CX"}
    )


def test_gate_counts_for_circuit():
    assert (
        gen.gate_counts_for_circuit(
            stim.Circuit(
                """
                H 0
                TICK
                CX 0 1
                """
            )
        )
        == {"H": 1, "TICK": 1, "CX": 1}
    )

    assert (
        gen.gate_counts_for_circuit(
            stim.Circuit(
                """
                S 0
                XCZ 0 1
                """
            )
        )
        == {"S": 1, "XCZ": 1}
    )

    assert (
        gen.gate_counts_for_circuit(
            stim.Circuit(
                """
                MPP X0*X1 Z2*Z3*Z4 Y0*Z1
                """
            )
        )
        == {"MXX": 1, "MZZZ": 1, "MYZ": 1}
    )

    assert (
        gen.gate_counts_for_circuit(
            stim.Circuit(
                """
                CX rec[-1] 1
                """
            )
        )
        == {"feedback": 1}
    )

    assert (
        gen.gate_counts_for_circuit(
            stim.Circuit(
                """
                CX sweep[1] 1
                """
            )
        )
        == {"sweep": 1}
    )

    assert (
        gen.gate_counts_for_circuit(
            stim.Circuit(
                """
                CX rec[-1] 1 0 1
                """
            )
        )
        == {"feedback": 1, "CX": 1}
    )

    assert (
        gen.gate_counts_for_circuit(
            stim.Circuit(
                """
                H 0 1
                REPEAT 100 {
                    S 0 1 2
                    CX 0 1 2 3
                }
                """
            )
        )
        == {"H": 2, "S": 300, "CX": 200}
    )


def test_count_measurement_layers():
    assert gen.count_measurement_layers(stim.Circuit()) == 0
    assert (
        gen.count_measurement_layers(
            stim.Circuit(
                """
                M 0 1 2
                """
            )
        )
        == 1
    )
    assert (
        gen.count_measurement_layers(
            stim.Circuit(
                """
                M 0 1
                MX 2
                MR 3
                """
            )
        )
        == 1
    )
    assert (
        gen.count_measurement_layers(
            stim.Circuit(
                """
                M 0 1
                MX 2
                TICK
                MR 3
                """
            )
        )
        == 2
    )
    assert (
        gen.count_measurement_layers(
            stim.Circuit(
                """
                R 0
                CX 0 1
                TICK
                M 0
                """
            )
        )
        == 1
    )
    assert (
        gen.count_measurement_layers(
            stim.Circuit(
                """
                R 0
                CX 0 1
                TICK
                M 0
                DETECTOR rec[-1]
                M 1
                DETECTOR rec[-1]
                OBSERVABLE_INCLUDE(0) rec[-1]
                MPP X0*X1
                DETECTOR rec[-1]
                """
            )
        )
        == 1
    )
    assert (
        gen.count_measurement_layers(
            stim.Circuit.generated(
                "repetition_code:memory",
                distance=3,
                rounds=4,
            )
        )
        == 4
    )
    assert (
        gen.count_measurement_layers(
            stim.Circuit.generated(
                "surface_code:rotated_memory_x",
                distance=3,
                rounds=1000,
            )
        )
        == 1000
    )


def test_verify_distance_is_at_least_23():
    gen.verify_distance_is_at_least_2(
        stim.Circuit(
            """
            R 0
            X_ERROR(0.125) 0
            M 0
            """
        )
    )

    gen.verify_distance_is_at_least_2(
        stim.Circuit(
            """
            R 0
            X_ERROR(0.125) 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """
        )
    )

    gen.verify_distance_is_at_least_2(
        stim.Circuit(
            """
            R 0
            X_ERROR(0.125) 0
            M 0
            DETECTOR rec[-1]
            OBSERVABLE_INCLUDE(0) rec[-1]
            """
        ).detector_error_model()
    )

    with pytest.raises(ValueError, match="distance 1 error"):
        gen.verify_distance_is_at_least_2(
            stim.Circuit(
                """
                R 0
                X_ERROR(0.125) 0
                M 0
                OBSERVABLE_INCLUDE(0) rec[-1]
                """
            )
        )

    with pytest.raises(ValueError, match="distance 1 error"):
        gen.verify_distance_is_at_least_3(
            stim.Circuit(
                """
                R 0
                X_ERROR(0.125) 0
                M 0
                OBSERVABLE_INCLUDE(0) rec[-1]
                """
            )
        )

    gen.verify_distance_is_at_least_2(
        stim.Circuit.generated(
            code_task="repetition_code:memory",
            distance=2,
            rounds=3,
            after_clifford_depolarization=1e-3,
        )
    )

    gen.verify_distance_is_at_least_2(
        stim.Circuit.generated(
            code_task="repetition_code:memory",
            distance=3,
            rounds=3,
            after_clifford_depolarization=1e-3,
        )
    )

    gen.verify_distance_is_at_least_2(
        stim.Circuit.generated(
            code_task="repetition_code:memory",
            distance=9,
            rounds=3,
            after_clifford_depolarization=1e-3,
        )
    )

    gen.verify_distance_is_at_least_3(
        stim.Circuit.generated(
            code_task="repetition_code:memory",
            distance=3,
            rounds=3,
            after_clifford_depolarization=1e-3,
        )
    )

    gen.verify_distance_is_at_least_3(
        stim.Circuit.generated(
            code_task="repetition_code:memory",
            distance=9,
            rounds=3,
            after_clifford_depolarization=1e-3,
        )
    )

    with pytest.raises(ValueError, match="distance 2 error"):
        gen.verify_distance_is_at_least_3(
            stim.Circuit.generated(
                code_task="repetition_code:memory",
                distance=2,
                rounds=3,
                after_clifford_depolarization=1e-3,
            )
        )
