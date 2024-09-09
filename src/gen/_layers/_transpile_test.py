import stim

from gen import transpile_to_z_basis_interaction_circuit


def test_to_cz_circuit_rotation_folding():
    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
    """
            )
        )
        == stim.Circuit(
            """
    """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
        C_XYZ 0
        TICK
        M 0 1 2 3
    """
            )
        )
        == stim.Circuit(
            """
        C_XYZ 0
        TICK
        M 0 1 2 3
    """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
        I 0
        TICK
        C_XYZ 0
        TICK
        M 0 1 2 3
    """
            )
        )
        == stim.Circuit(
            """
        C_XYZ 0
        TICK
        M 0 1 2 3
    """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
        C_XYZ 0
        TICK
        I 0
        TICK
        M 0 1 2 3
    """
            )
        )
        == stim.Circuit(
            """
        C_XYZ 0
        TICK
        M 0 1 2 3
    """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
        H 0
        TICK
        H 0
        TICK
        M 0 1 2 3
    """
            )
        )
        == stim.Circuit(
            """
        M 0 1 2 3
    """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    H 0 1 2 3 4 5
                    TICK
                    I 0
                    H_YZ 1
                    H_XY 2
                    C_XYZ 3
                    C_ZYX 4
                    H 5
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                H 0
                S 4
                SQRT_X_DAG 3
                C_XYZ 1
                Y 1
                C_ZYX 2
                Y 2
                TICK
                M 0 1 2 3
            """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    H_XY 0 1 2 3 4 5
                    TICK
                    I 0
                    H_YZ 1
                    H_XY 2
                    C_XYZ 3
                    C_ZYX 4
                    H 5
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                H_XY 0
                SQRT_X 4
                SQRT_Y_DAG 3
                C_XYZ 5
                Z 5
                C_ZYX 1
                Z 1
                TICK
                M 0 1 2 3
            """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    H_YZ 0 1 2 3 4 5
                    TICK
                    I 0
                    H_YZ 1
                    H_XY 2
                    C_XYZ 3
                    C_ZYX 4
                    H 5
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                H_YZ 0
                SQRT_Y 4
                S_DAG 3
                C_XYZ 2
                X 2
                C_ZYX 5
                X 5
                TICK
                M 0 1 2 3
            """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    C_XYZ 0 1 2 3 4 5
                    TICK
                    I 0
                    H_YZ 1
                    H_XY 2
                    C_XYZ 3
                    C_ZYX 4
                    H 5
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                C_XYZ 0
                C_ZYX 3
                SQRT_X_DAG 2
                SQRT_Y_DAG 1
                S_DAG 5
                TICK
                M 0 1 2 3
            """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    C_ZYX 0 1 2 3 4 5
                    TICK
                    I 0
                    H_YZ 1
                    H_XY 2
                    C_XYZ 3
                    C_ZYX 4
                    H 5
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                C_XYZ 4
                C_ZYX 0
                S 1
                SQRT_X 5
                SQRT_Y 2
                TICK
                M 0 1 2 3
            """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    I 0 1 2 3 4 5
                    TICK
                    I 0
                    H_YZ 1
                    H_XY 2
                    C_XYZ 3
                    C_ZYX 4
                    H 5
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                C_XYZ 3
                C_ZYX 4
                H 5
                H_XY 2
                H_YZ 1
                TICK
                M 0 1 2 3
            """
        )
    )


def test_to_cz_circuit_loop_boundary_folding():
    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
        H 2
        TICK
        CX rec[-1] 2 
        TICK
        S 2
        TICK
        M 0 1 2 3
    """
            )
        )
        == stim.Circuit(
            """
        CZ rec[-1] 2
        C_ZYX 2 
        TICK
        M 0 1 2 3
    """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
        MRX 0
        DETECTOR rec[-1]
        TICK
        M 0
        TICK
        H 0
    """
            )
        )
        == stim.Circuit(
            """
        H 0
        TICK
        M 0
        TICK
        R 0
        DETECTOR rec[-1]
        TICK
        H 0
        TICK
        M 0
    """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
        REPEAT 100 {
            C_XYZ 0
            TICK
            CZ 0 1
            TICK
            H 0
            TICK
        }
        M 0
    """
            )
        )
        == stim.Circuit(
            """
    H 0
        TICK
        REPEAT 100 {
            SQRT_X_DAG 0
            TICK
            CZ 0 1
            TICK
        }
        H 0
        TICK
        M 0
    """
        )
    )


def test_to_cz_circuit_from_cnot():
    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
        CX 0 1 2 3
        TICK
        CX 1 0 2 3
        TICK
        M 0 1 2 3
    """
            )
        )
        == stim.Circuit(
            """
        H 1 3
        TICK
        CZ 0 1 2 3
        TICK
        H 0 1
        TICK
        CZ 0 1 2 3
        TICK
        H 0 3
        TICK
        M 0 1 2 3
    """
        )
    )


def test_to_cz_circuit_from_swap_cnot():
    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    CNOT 0 1
                    TICK
                    SWAP 0 1
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                H 1
                TICK
                CZSWAP 0 1
                TICK
                H 0
                TICK
                M 0 1 2 3
            """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    CNOT 0 1
                    TICK
                    SWAP 1 0
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                H 1
                TICK
                CZSWAP 0 1
                TICK
                H 0
                TICK
                M 0 1 2 3
            """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    SWAP 1 0
                    TICK
                    CNOT 1 0
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                H 1
                TICK
                CZSWAP 0 1
                TICK
                H 0
                TICK
                M 0 1 2 3
            """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    SWAP 0 1
                    TICK
                    CNOT 1 0
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                H 1
                TICK
                CZSWAP 0 1
                TICK
                H 0
                TICK
                M 0 1 2 3
            """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    CNOT 1 0
                    TICK
                    SWAP 0 1
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                H 0
                TICK
                CZSWAP 0 1
                TICK
                H 1
                TICK
                M 0 1 2 3
            """
        )
    )

    assert (
        transpile_to_z_basis_interaction_circuit(
            stim.Circuit(
                """
                    SWAP 0 1
                    TICK
                    CNOT 0 1
                    TICK
                    M 0 1 2 3
                """
            )
        )
        == stim.Circuit(
            """
                H 0
                TICK
                CZSWAP 0 1
                TICK
                H 1
                TICK
                M 0 1 2 3
            """
        )
    )
