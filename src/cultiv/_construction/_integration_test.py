from typing import Any

import chromobius
import pytest
import sinter
import stim

import gen
from ._integration import \
    make_escape_to_big_matchable_code_circuit, make_end2end_cultivation_circuit, \
    make_inject_and_cultivate_circuit, \
    make_idle_matchable_code_circuit, make_escape_to_big_color_code_circuit, make_surface_code_memory_circuit
from cultiv._decoding._chromobius_gap_sampler import CompiledChromobiusGapSampler


@pytest.mark.parametrize('dcolor,dsurface', [(3, 7), (5, 10)])
@pytest.mark.parametrize('basis', ['Y'])
def test_make_escape_to_big_matchable_code_circuit(dcolor: int, dsurface: int, basis: Any):
    circuit = make_escape_to_big_matchable_code_circuit(
        dcolor=dcolor,
        dsurface=dsurface,
        r_growing=dcolor - 1,
        r_end=2,
        basis=basis,
    )
    circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
    circuit.detector_error_model()
    err = circuit.search_for_undetectable_logical_errors(
        dont_explore_edges_increasing_symptom_degree=False,
        dont_explore_edges_with_degree_above=3,
        canonicalize_circuit_errors=True,
        dont_explore_detection_event_sets_with_size_above=3,
    )
    assert len(err) == dcolor


@pytest.mark.parametrize('dcolor,dsurface', [(3, 6)])
@pytest.mark.parametrize('basis', ['X'])
def test_make_inject_only_circuit(dcolor: int, dsurface: int, basis: Any):
    circuit = make_inject_and_cultivate_circuit(
        dcolor=dcolor,
        inject_style='unitary',
        basis=basis,
    )
    circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
    circuit.detector_error_model()
    err = circuit.search_for_undetectable_logical_errors(
        dont_explore_edges_increasing_symptom_degree=False,
        dont_explore_edges_with_degree_above=3,
        canonicalize_circuit_errors=True,
        dont_explore_detection_event_sets_with_size_above=3,
    )
    assert len(err) == min(4, dcolor)


@pytest.mark.parametrize('dcolor,dsurface', [(3, 6), (5, 10)])
@pytest.mark.parametrize('basis', ['Y'])
def test_make_end2end_circuit(dcolor: int, dsurface: int, basis: Any):
    circuit = make_end2end_cultivation_circuit(
        dcolor=dcolor,
        dsurface=dsurface,
        r_growing=dcolor - 1,
        r_end=2,
        basis=basis,
        inject_style='unitary',
    )
    circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
    circuit.detector_error_model()
    err = circuit.search_for_undetectable_logical_errors(
        dont_explore_edges_increasing_symptom_degree=False,
        dont_explore_edges_with_degree_above=3,
        canonicalize_circuit_errors=True,
        dont_explore_detection_event_sets_with_size_above=3,
    )
    assert len(err) == min(4, dcolor)


@pytest.mark.parametrize('dcolor,dsurface,rounds', [(3, 6, 4), (5, 10, 4)])
@pytest.mark.parametrize('basis', ['X', 'Y', 'Z'])
def test_make_matchable_idle_circuit(dcolor: int, dsurface: int, basis: Any, rounds: int):
    circuit = make_idle_matchable_code_circuit(dcolor=dcolor, dsurface=dsurface, basis=basis, rounds=rounds)
    circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
    assert circuit.detector_error_model(decompose_errors=True) is not None
    err = circuit.shortest_graphlike_error()
    assert len(err) == dsurface


def test_make_matchable_idle_circuit_exact():
    assert make_idle_matchable_code_circuit(dcolor=3, dsurface=6, basis='Y', rounds=100) == stim.Circuit("""
        QUBIT_COORDS(0, -1) 0
        QUBIT_COORDS(0, 0) 1
        QUBIT_COORDS(1, -1) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        QUBIT_COORDS(1, 2) 5
        QUBIT_COORDS(1, 3) 6
        QUBIT_COORDS(2, -3) 7
        QUBIT_COORDS(2, -2) 8
        QUBIT_COORDS(2, -1) 9
        QUBIT_COORDS(2, 0) 10
        QUBIT_COORDS(2, 1) 11
        QUBIT_COORDS(2, 2) 12
        QUBIT_COORDS(2, 3) 13
        QUBIT_COORDS(3, -3) 14
        QUBIT_COORDS(3, -2) 15
        QUBIT_COORDS(3, -1) 16
        QUBIT_COORDS(3, 0) 17
        QUBIT_COORDS(3, 1) 18
        QUBIT_COORDS(3, 2) 19
        QUBIT_COORDS(3, 3) 20
        QUBIT_COORDS(3, 4) 21
        QUBIT_COORDS(3, 5) 22
        QUBIT_COORDS(4, -5) 23
        QUBIT_COORDS(4, -4) 24
        QUBIT_COORDS(4, -3) 25
        QUBIT_COORDS(4, -2) 26
        QUBIT_COORDS(4, -1) 27
        QUBIT_COORDS(4, 0) 28
        QUBIT_COORDS(4, 1) 29
        QUBIT_COORDS(4, 2) 30
        QUBIT_COORDS(4, 3) 31
        QUBIT_COORDS(4, 4) 32
        QUBIT_COORDS(4, 5) 33
        QUBIT_COORDS(5, -5) 34
        QUBIT_COORDS(5, -4) 35
        QUBIT_COORDS(5, -3) 36
        QUBIT_COORDS(5, -2) 37
        QUBIT_COORDS(5, -1) 38
        QUBIT_COORDS(5, 0) 39
        QUBIT_COORDS(5, 1) 40
        QUBIT_COORDS(5, 2) 41
        QUBIT_COORDS(5, 3) 42
        QUBIT_COORDS(5, 4) 43
        QUBIT_COORDS(5, 5) 44
        QUBIT_COORDS(6, -4) 45
        QUBIT_COORDS(6, -3) 46
        QUBIT_COORDS(6, -2) 47
        QUBIT_COORDS(6, -1) 48
        QUBIT_COORDS(6, 0) 49
        QUBIT_COORDS(6, 1) 50
        QUBIT_COORDS(6, 2) 51
        QUBIT_COORDS(6, 3) 52
        QUBIT_COORDS(6, 4) 53
        QUBIT_COORDS(6, 5) 54
        QUBIT_COORDS(7, -4) 55
        QUBIT_COORDS(7, -3) 56
        QUBIT_COORDS(7, -2) 57
        QUBIT_COORDS(7, -1) 58
        QUBIT_COORDS(7, 0) 59
        QUBIT_COORDS(7, 1) 60
        QUBIT_COORDS(7, 2) 61
        QUBIT_COORDS(7, 3) 62
        QUBIT_COORDS(8, -2) 63
        QUBIT_COORDS(8, -1) 64
        QUBIT_COORDS(8, 0) 65
        QUBIT_COORDS(8, 1) 66
        QUBIT_COORDS(8, 2) 67
        QUBIT_COORDS(8, 3) 68
        QUBIT_COORDS(9, -2) 69
        QUBIT_COORDS(9, -1) 70
        QUBIT_COORDS(9, 0) 71
        QUBIT_COORDS(9, 1) 72
        QUBIT_COORDS(10, 0) 73
        QUBIT_COORDS(10, 1) 74
        MPP X34*Z44*X45*Z53*X56*Z62*X63*Z67*X70*Z72*Y73 X1*X2 X4*X11*X13*X19 X8*X14 X16*X17*X26*X28*X38 X21*X30*X32*X42 X40*X49*X51*X60 Z1*Z2*Z4*Z11*Z16*Z17 Z13*Z19*Z21*Z30 X14*X24*X26*X36 Z28*Z38*Z40*Z49 Z32*Z42*Z44*Z53 Z45*Z56 Z51*Z60*Z62*Z67 X58*X63*X65*X70 X72*X73 X2*X8*X16 Z4*Z13 X11*X17*X19*X28*X30*X40 Z21*Z32 X24*X34 Z26*Z36*Z38*Z47 X42*X51*X53*X62 Z49*Z58*Z60*Z65 Z63*Z70 Z8*Z14*Z16*Z26 Z11*Z19 Z17*Z28 Z24*Z34*Z36*Z45 Z30*Z40*Z42*Z51 X38*X47*X49*X58 X44*X53 X60*X65*X67*X72 X36*X45*X47*X56 X62*X67 Z65*Z70*Z72*Z73 Z47*Z56*Z58*Z63
        OBSERVABLE_INCLUDE(0) rec[-37]
        TICK
        RX 0 3 5 7 9 18 20 22 23 25 27 46 48 50 52 54 64 66 68 74
        R 6 10 12 15 29 31 33 35 37 39 41 43 55 57 59 61 69 71
        TICK
        CX 0 1 3 10 5 12 7 8 9 16 18 29 20 31 22 33 23 24 27 38 54 53 56 55 68 67 70 69 74 73
        TICK
        CX 0 2 7 14 12 19 18 11 23 34 29 40 31 42 45 55 54 44 63 69 68 62 74 72
        TICK
        CX 5 6 9 8 12 13 16 17 18 19 20 21 27 28 29 30 31 32
        TICK
        CX 5 4 9 2 12 11 13 6 14 15 18 17 25 24 27 26 29 28 31 30 34 35 36 37 38 39 40 41 42 43 46 45 48 47 50 49 52 51 56 57 58 59 60 61 64 63 66 65 70 71
        TICK
        CX 1 3 4 5 13 20 17 10 19 12 25 14 26 15 27 16 45 35 46 36 47 37 48 38 49 39 50 40 51 41 52 42 53 43 63 57 64 58 65 59 66 60 67 61 73 71
        TICK
        CX 4 3 5 6 8 15 11 10 16 17 21 20 24 35 25 36 26 37 28 39 30 41 32 43 46 56 47 57 48 58 49 59 50 60 51 61 52 62 64 70 65 71 66 72
        TICK
        CX 2 3 4 5 11 12 17 18 19 20 21 22 28 29 30 31 32 33
        TICK
        CX 3 10 5 12 16 15 18 29 20 31 22 33 25 26 36 35 38 37 40 39 42 41 44 43 46 47 48 49 50 51 52 53 58 57 60 59 62 61 64 65 66 67 72 71
        TICK
        MX 0 3 5 7 9 18 20 22 23 25 27 46 48 50 52 54 64 66 68 74
        M 6 10 12 15 29 31 33 35 37 39 41 43 55 57 59 61 69 71
        DETECTOR(1, 0, 0, 0, 6) rec[-37]
        DETECTOR(3, 5, 0, 0, 6) rec[-31]
        DETECTOR(0, -0.5, 0, 0, 6) rec[-74] rec[-38]
        DETECTOR(1, 1.5, 0, 0, 6) rec[-73] rec[-36]
        DETECTOR(1, 2, 0, 3, 7) rec[-58] rec[-18]
        DETECTOR(2, -2.5, 0, 0, 6) rec[-72] rec[-35]
        DETECTOR(1.5, -1, 0, 0, 6) rec[-59] rec[-34]
        DETECTOR(1, 0, 1, 3, 7) rec[-68] rec[-17]
        DETECTOR(2, 1.5, 0, 3, 7) rec[-49] rec[-16]
        DETECTOR(2.5, -2, 0, 3, 7) rec[-50] rec[-15]
        DETECTOR(2.5, 1, 0, 0, 6) rec[-57] rec[-33]
        DETECTOR(3, 3.5, 0, 0, 6) rec[-70] rec[-32]
        DETECTOR(4, -4.5, 0, 0, 6) rec[-55] rec[-30]
        DETECTOR(3.5, -3, 0, 0, 6) rec[-66] rec[-29]
        DETECTOR(3.5, -1, 0, 0, 6) rec[-71] rec[-28]
        DETECTOR(3.5, 0.5, 0, 3, 7) rec[-48] rec[-14]
        DETECTOR(3, 3, 0, 3, 7) rec[-67] rec[-13]
        DETECTOR(3.5, 4.5, 0, 3, 7) rec[-56] rec[-12]
        DETECTOR(4.5, -4, 0, 3, 7) rec[-47] rec[-11]
        DETECTOR(4.5, -2, 0, 3, 7) rec[-54] rec[-10]
        DETECTOR(4.5, 0, 0, 3, 7) rec[-65] rec[-9]
        DETECTOR(4.5, 2, 0, 3, 7) rec[-46] rec[-8]
        DETECTOR(4.5, 4, 0, 3, 7) rec[-64] rec[-7]
        DETECTOR(5.5, -3, 0, 0, 6) rec[-42] rec[-27]
        DETECTOR(5.5, -1, 0, 0, 6) rec[-45] rec[-26]
        DETECTOR(5.5, 1, 0, 0, 6) rec[-69] rec[-25]
        DETECTOR(5.5, 3, 0, 0, 6) rec[-53] rec[-24]
        DETECTOR(5.5, 5, 0, 0, 6) rec[-44] rec[-23]
        DETECTOR(6.5, -4, 0, 3, 7) rec[-63] rec[-6]
        DETECTOR(6.5, -2, 0, 3, 7) rec[-39] rec[-5]
        DETECTOR(6.5, 0, 0, 3, 7) rec[-52] rec[-4]
        DETECTOR(6.5, 2, 0, 3, 7) rec[-62] rec[-3]
        DETECTOR(7.5, -1, 0, 0, 6) rec[-61] rec[-22]
        DETECTOR(7.5, 1, 0, 0, 6) rec[-43] rec[-21]
        DETECTOR(7.5, 3, 0, 0, 6) rec[-41] rec[-20]
        DETECTOR(8.5, -2, 0, 3, 7) rec[-51] rec[-2]
        DETECTOR(8.5, 0, 0, 3, 7) rec[-40] rec[-1]
        DETECTOR(9.5, 1, 0, 0, 6) rec[-60] rec[-19]
        SHIFT_COORDS(0, 0, 2)
        TICK
        REPEAT 99 {
            RX 0 3 5 7 9 18 20 22 23 25 27 46 48 50 52 54 64 66 68 74
            R 6 10 12 15 29 31 33 35 37 39 41 43 55 57 59 61 69 71
            TICK
            CX 0 1 3 10 5 12 7 8 9 16 18 29 20 31 22 33 23 24 27 38 54 53 56 55 68 67 70 69 74 73
            TICK
            CX 0 2 7 14 12 19 18 11 23 34 29 40 31 42 45 55 54 44 63 69 68 62 74 72
            TICK
            CX 5 6 9 8 12 13 16 17 18 19 20 21 27 28 29 30 31 32
            TICK
            CX 5 4 9 2 12 11 13 6 14 15 18 17 25 24 27 26 29 28 31 30 34 35 36 37 38 39 40 41 42 43 46 45 48 47 50 49 52 51 56 57 58 59 60 61 64 63 66 65 70 71
            TICK
            CX 1 3 4 5 13 20 17 10 19 12 25 14 26 15 27 16 45 35 46 36 47 37 48 38 49 39 50 40 51 41 52 42 53 43 63 57 64 58 65 59 66 60 67 61 73 71
            TICK
            CX 4 3 5 6 8 15 11 10 16 17 21 20 24 35 25 36 26 37 28 39 30 41 32 43 46 56 47 57 48 58 49 59 50 60 51 61 52 62 64 70 65 71 66 72
            TICK
            CX 2 3 4 5 11 12 17 18 19 20 21 22 28 29 30 31 32 33
            TICK
            CX 3 10 5 12 16 15 18 29 20 31 22 33 25 26 36 35 38 37 40 39 42 41 44 43 46 47 48 49 50 51 52 53 58 57 60 59 62 61 64 65 66 67 72 71
            TICK
            MX 0 3 5 7 9 18 20 22 23 25 27 46 48 50 52 54 64 66 68 74
            M 6 10 12 15 29 31 33 35 37 39 41 43 55 57 59 61 69 71
            DETECTOR(1, 0, 0, 0, 6) rec[-37]
            DETECTOR(3, 5, 0, 0, 6) rec[-31]
            DETECTOR(0, -1, 0, 0, 6) rec[-76] rec[-38]
            DETECTOR(1, 2, 0, 0, 6) rec[-74] rec[-36]
            DETECTOR(1, 3, 0, 3, 7) rec[-56] rec[-18]
            DETECTOR(2, -3, 0, 0, 6) rec[-73] rec[-35]
            DETECTOR(2, -1, 0, 0, 6) rec[-72] rec[-34]
            DETECTOR(2, 0, 0, 3, 7) rec[-55] rec[-17]
            DETECTOR(2, 2, 0, 3, 7) rec[-54] rec[-16]
            DETECTOR(3, -2, 0, 3, 7) rec[-53] rec[-15]
            DETECTOR(3, 1, 0, 0, 6) rec[-70] rec[-33]
            DETECTOR(3, 3, 0, 0, 6) rec[-32]
            DETECTOR(4, -5, 0, 0, 6) rec[-68] rec[-30]
            DETECTOR(4, -3, 0, 0, 6) rec[-67] rec[-29]
            DETECTOR(4, -1, 0, 0, 6) rec[-71] rec[-66] rec[-28]
            DETECTOR(4, 1, 0, 3, 7) rec[-52] rec[-14]
            DETECTOR(4, 3, 0, 3, 7) rec[-51] rec[-13]
            DETECTOR(4, 5, 0, 3, 7) rec[-50] rec[-12]
            DETECTOR(5, -4, 0, 3, 7) rec[-49] rec[-11]
            DETECTOR(5, -2, 0, 3, 7) rec[-48] rec[-10]
            DETECTOR(5, 0, 0, 3, 7) rec[-47] rec[-9]
            DETECTOR(5, 2, 0, 3, 7) rec[-46] rec[-8]
            DETECTOR(5, 4, 0, 3, 7) rec[-45] rec[-7]
            DETECTOR(6, -3, 0, 0, 6) rec[-65] rec[-27]
            DETECTOR(6, -1, 0, 0, 6) rec[-64] rec[-26]
            DETECTOR(6, 1, 0, 0, 6) rec[-63] rec[-25]
            DETECTOR(6, 3, 0, 0, 6) rec[-62] rec[-24]
            DETECTOR(6, 5, 0, 0, 6) rec[-61] rec[-23]
            DETECTOR(7, -4, 0, 3, 7) rec[-44] rec[-6]
            DETECTOR(7, -2, 0, 3, 7) rec[-43] rec[-5]
            DETECTOR(7, 0, 0, 3, 7) rec[-42] rec[-4]
            DETECTOR(7, 2, 0, 3, 7) rec[-41] rec[-3]
            DETECTOR(8, -1, 0, 0, 6) rec[-60] rec[-22]
            DETECTOR(8, 1, 0, 0, 6) rec[-59] rec[-21]
            DETECTOR(8, 3, 0, 0, 6) rec[-58] rec[-20]
            DETECTOR(9, -2, 0, 3, 7) rec[-40] rec[-2]
            DETECTOR(9, 0, 0, 3, 7) rec[-39] rec[-1]
            DETECTOR(10, 1, 0, 0, 6) rec[-57] rec[-19]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }
        MPP X34*Z44*X45*Z53*X56*Z62*X63*Z67*X70*Z72*Y73 X1*X2 X4*X11*X13*X19 X8*X14 X16*X17*X26*X28*X38 X21*X30*X32*X42 X40*X49*X51*X60 Z1*Z2*Z4*Z11*Z16*Z17 Z13*Z19*Z21*Z30 X14*X24*X26*X36 Z28*Z38*Z40*Z49 Z32*Z42*Z44*Z53 Z45*Z56 Z51*Z60*Z62*Z67 X58*X63*X65*X70 X72*X73 X2*X8*X16 Z4*Z13 X11*X17*X19*X28*X30*X40 Z21*Z32 X24*X34 Z26*Z36*Z38*Z47 X42*X51*X53*X62 Z49*Z58*Z60*Z65 Z63*Z70 Z8*Z14*Z16*Z26 Z11*Z19 Z17*Z28 Z24*Z34*Z36*Z45 Z30*Z40*Z42*Z51 X38*X47*X49*X58 X44*X53 X60*X65*X67*X72 X36*X45*X47*X56 X62*X67 Z65*Z70*Z72*Z73 Z47*Z56*Z58*Z63
        OBSERVABLE_INCLUDE(0) rec[-37]
        DETECTOR(0, -0.5, 0, 0, 6) rec[-75] rec[-36]
        DETECTOR(1, 0, 0, 3, 7) rec[-54] rec[-30]
        DETECTOR(1.5, -1, 0, 0, 6) rec[-71] rec[-21]
        DETECTOR(1, 1.5, 0, 0, 6) rec[-73] rec[-35]
        DETECTOR(1, 2, 0, 3, 7) rec[-55] rec[-20]
        DETECTOR(2, -2.5, 0, 0, 6) rec[-72] rec[-34]
        DETECTOR(2.5, -2, 0, 3, 7) rec[-52] rec[-12]
        DETECTOR(2.5, 1, 0, 0, 6) rec[-69] rec[-19]
        DETECTOR(2, 1.5, 0, 3, 7) rec[-53] rec[-11]
        DETECTOR(3, 3, 0, 3, 7) rec[-50] rec[-29]
        DETECTOR(3.5, -3, 0, 0, 6) rec[-66] rec[-28]
        DETECTOR(3.5, -1, 0, 0, 6) rec[-70] rec[-65] rec[-33]
        DETECTOR(3.5, 0.5, 0, 3, 7) rec[-51] rec[-10]
        DETECTOR(3, 3.5, 0, 0, 6) rec[-32]
        DETECTOR(3.5, 4.5, 0, 3, 7) rec[-49] rec[-18]
        DETECTOR(4, -4.5, 0, 0, 6) rec[-67] rec[-17]
        DETECTOR(4.5, -4, 0, 3, 7) rec[-48] rec[-9]
        DETECTOR(4.5, -2, 0, 3, 7) rec[-47] rec[-16]
        DETECTOR(4.5, 0, 0, 3, 7) rec[-46] rec[-27]
        DETECTOR(4.5, 2, 0, 3, 7) rec[-45] rec[-8]
        DETECTOR(4.5, 4, 0, 3, 7) rec[-44] rec[-26]
        DETECTOR(5.5, -3, 0, 0, 6) rec[-64] rec[-4]
        DETECTOR(5.5, -1, 0, 0, 6) rec[-63] rec[-7]
        DETECTOR(5.5, 1, 0, 0, 6) rec[-62] rec[-31]
        DETECTOR(5.5, 3, 0, 0, 6) rec[-61] rec[-15]
        DETECTOR(5.5, 5, 0, 0, 6) rec[-60] rec[-6]
        DETECTOR(6.5, -4, 0, 3, 7) rec[-43] rec[-25]
        DETECTOR(6.5, -2, 0, 3, 7) rec[-42] rec[-1]
        DETECTOR(6.5, 0, 0, 3, 7) rec[-41] rec[-14]
        DETECTOR(6.5, 2, 0, 3, 7) rec[-40] rec[-24]
        DETECTOR(7.5, -1, 0, 0, 6) rec[-59] rec[-23]
        DETECTOR(7.5, 1, 0, 0, 6) rec[-58] rec[-5]
        DETECTOR(7.5, 3, 0, 0, 6) rec[-57] rec[-3]
        DETECTOR(8.5, -2, 0, 3, 7) rec[-39] rec[-13]
        DETECTOR(8.5, 0, 0, 3, 7) rec[-38] rec[-2]
        DETECTOR(9.5, 1, 0, 0, 6) rec[-56] rec[-22]
    """)


@pytest.mark.parametrize('basis', ['X', 'Y', 'Z'])
def test_make_growth_only_circuit_8(basis: Any):
    circuit = make_escape_to_big_color_code_circuit(
        start_width=5,
        end_width=11,
        rounds=3,
        basis=basis,
    )
    noisy_circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
    err = noisy_circuit.search_for_undetectable_logical_errors(
        dont_explore_edges_with_degree_above=3,
        dont_explore_edges_increasing_symptom_degree=False,
        dont_explore_detection_event_sets_with_size_above=3,
        canonicalize_circuit_errors=True,
    )
    d = len(err)
    assert d == 5

    dem = noisy_circuit.detector_error_model()
    chromobius.compile_decoder_for_dem(dem)
    if basis == 'X' or basis == 'Z':
        # Successfully configures decoder.
        decoder = CompiledChromobiusGapSampler(sinter.Task(
            circuit=noisy_circuit,
            detector_error_model=dem,
            json_metadata={},
        ))
        assert decoder is not None


def test_make_surface_code_memory_circuit_x():
    circuit = make_surface_code_memory_circuit(dsurface=3, rounds=4, basis='X')
    circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit(circuit)
    assert len(circuit.shortest_graphlike_error()) == 3
    assert circuit == stim.Circuit("""
        QUBIT_COORDS(0, -1) 0
        QUBIT_COORDS(0, 0) 1
        QUBIT_COORDS(1, -1) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        QUBIT_COORDS(1, 2) 5
        QUBIT_COORDS(2, -2) 6
        QUBIT_COORDS(2, -1) 7
        QUBIT_COORDS(2, 0) 8
        QUBIT_COORDS(2, 1) 9
        QUBIT_COORDS(2, 2) 10
        QUBIT_COORDS(3, -2) 11
        QUBIT_COORDS(3, -1) 12
        QUBIT_COORDS(3, 0) 13
        QUBIT_COORDS(3, 1) 14
        QUBIT_COORDS(4, 0) 15
        QUBIT_COORDS(4, 1) 16
        RX 1 2 4 6 8 10 12 14 15 0 7 9 16
        R 3 5 11 13
        X_ERROR(0.001) 3 5 11 13
        Z_ERROR(0.001) 1 2 4 6 8 10 12 14 15 0 7 9 16
        TICK
        CX 0 1 4 3 7 8 9 10 12 11 14 13
        DEPOLARIZE2(0.001) 0 1 4 3 7 8 9 10 12 11 14 13
        DEPOLARIZE1(0.001) 2 5 6 15 16
        TICK
        CX 0 2 1 3 6 11 7 12 8 13 9 14
        DEPOLARIZE2(0.001) 0 2 1 3 6 11 7 12 8 13 9 14
        DEPOLARIZE1(0.001) 4 5 10 15 16
        TICK
        CX 7 2 8 3 9 4 10 5 15 13 16 14
        DEPOLARIZE2(0.001) 7 2 8 3 9 4 10 5 15 13 16 14
        DEPOLARIZE1(0.001) 0 1 6 11 12
        TICK
        CX 2 3 4 5 7 6 9 8 12 13 16 15
        DEPOLARIZE2(0.001) 2 3 4 5 7 6 9 8 12 13 16 15
        DEPOLARIZE1(0.001) 0 1 10 11 14
        TICK
        MX(0.001) 0 7 9 16
        M(0.001) 3 5 11 13
        DETECTOR(0, -1, 0) rec[-8]
        DETECTOR(2, -1, 0) rec[-7]
        DETECTOR(2, 1, 0) rec[-6]
        DETECTOR(4, 1, 0) rec[-5]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.001) 0 7 9 16 3 5 11 13 1 2 4 6 8 10 12 14 15
        TICK
        REPEAT 3 {
            RX 0 7 9 16
            R 3 5 11 13
            X_ERROR(0.001) 3 5 11 13
            Z_ERROR(0.001) 0 7 9 16
            DEPOLARIZE1(0.001) 1 2 4 6 8 10 12 14 15
            TICK
            CX 0 1 4 3 7 8 9 10 12 11 14 13
            DEPOLARIZE2(0.001) 0 1 4 3 7 8 9 10 12 11 14 13
            DEPOLARIZE1(0.001) 2 5 6 15 16
            TICK
            CX 0 2 1 3 6 11 7 12 8 13 9 14
            DEPOLARIZE2(0.001) 0 2 1 3 6 11 7 12 8 13 9 14
            DEPOLARIZE1(0.001) 4 5 10 15 16
            TICK
            CX 7 2 8 3 9 4 10 5 15 13 16 14
            DEPOLARIZE2(0.001) 7 2 8 3 9 4 10 5 15 13 16 14
            DEPOLARIZE1(0.001) 0 1 6 11 12
            TICK
            CX 2 3 4 5 7 6 9 8 12 13 16 15
            DEPOLARIZE2(0.001) 2 3 4 5 7 6 9 8 12 13 16 15
            DEPOLARIZE1(0.001) 0 1 10 11 14
            TICK
            MX(0.001) 0 7 9 16
            M(0.001) 3 5 11 13
            DETECTOR(0, -1, 0) rec[-16] rec[-8]
            DETECTOR(1, 0, 0) rec[-12] rec[-4]
            DETECTOR(1, 2, 0) rec[-11] rec[-3]
            DETECTOR(2, -1, 0) rec[-15] rec[-7]
            DETECTOR(2, 1, 0) rec[-14] rec[-6]
            DETECTOR(3, -2, 0) rec[-10] rec[-2]
            DETECTOR(3, 0, 0) rec[-9] rec[-1]
            DETECTOR(4, 1, 0) rec[-13] rec[-5]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE1(0.001) 0 7 9 16 3 5 11 13 1 2 4 6 8 10 12 14 15
            TICK
        }
        MX(0.001) 15 14 12 10 8 6 4 2 1
        DETECTOR(0, -1, 0) rec[-17] rec[-2] rec[-1]
        DETECTOR(2, -1, 0) rec[-16] rec[-7] rec[-5] rec[-4] rec[-2]
        DETECTOR(2, 1, 0) rec[-15] rec[-8] rec[-6] rec[-5] rec[-3]
        DETECTOR(4, 1, 0) rec[-14] rec[-9] rec[-8]
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-3] rec[-1]
        DEPOLARIZE1(0.001) 15 14 12 10 8 6 4 2 1 0 3 5 7 9 11 13 16
    """)


def test_make_surface_code_memory_circuit_y():
    circuit = make_surface_code_memory_circuit(dsurface=3, rounds=4, basis='Y')
    circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
    assert len(circuit.shortest_graphlike_error()) == 3
    assert circuit == stim.Circuit("""
        QUBIT_COORDS(0, -1) 0
        QUBIT_COORDS(0, 0) 1
        QUBIT_COORDS(1, -1) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        QUBIT_COORDS(1, 2) 5
        QUBIT_COORDS(2, -2) 6
        QUBIT_COORDS(2, -1) 7
        QUBIT_COORDS(2, 0) 8
        QUBIT_COORDS(2, 1) 9
        QUBIT_COORDS(2, 2) 10
        QUBIT_COORDS(3, -2) 11
        QUBIT_COORDS(3, -1) 12
        QUBIT_COORDS(3, 0) 13
        QUBIT_COORDS(3, 1) 14
        QUBIT_COORDS(4, 0) 15
        QUBIT_COORDS(4, 1) 16
        MPP Y1*Z2*X4*Z6*X10 X1*X2 Z1*Z2*Z4*Z8 Z4*Z10 X2*X6*X8*X12 X4*X8*X10*X14 Z6*Z12 Z8*Z12*Z14*Z15 X14*X15
        OBSERVABLE_INCLUDE(0) rec[-9]
        TICK
        RX 0 7 9 16
        R 3 5 11 13
        X_ERROR(0.001) 3 5 11 13
        Z_ERROR(0.001) 0 7 9 16
        DEPOLARIZE1(0.001) 1 2 4 6 8 10 12 14 15
        TICK
        CX 0 1 4 3 7 8 9 10 12 11 14 13
        DEPOLARIZE2(0.001) 0 1 4 3 7 8 9 10 12 11 14 13
        DEPOLARIZE1(0.001) 2 5 6 15 16
        TICK
        CX 0 2 1 3 6 11 7 12 8 13 9 14
        DEPOLARIZE2(0.001) 0 2 1 3 6 11 7 12 8 13 9 14
        DEPOLARIZE1(0.001) 4 5 10 15 16
        TICK
        CX 7 2 8 3 9 4 10 5 15 13 16 14
        DEPOLARIZE2(0.001) 7 2 8 3 9 4 10 5 15 13 16 14
        DEPOLARIZE1(0.001) 0 1 6 11 12
        TICK
        CX 2 3 4 5 7 6 9 8 12 13 16 15
        DEPOLARIZE2(0.001) 2 3 4 5 7 6 9 8 12 13 16 15
        DEPOLARIZE1(0.001) 0 1 10 11 14
        TICK
        MX(0.001) 0 7 9 16
        M(0.001) 3 5 11 13
        DETECTOR(0, -1, 0) rec[-16] rec[-8]
        DETECTOR(1, 0, 0) rec[-15] rec[-4]
        DETECTOR(1, 2, 0) rec[-14] rec[-3]
        DETECTOR(2, -1, 0) rec[-13] rec[-7]
        DETECTOR(2, 1, 0) rec[-12] rec[-6]
        DETECTOR(3, -2, 0) rec[-11] rec[-2]
        DETECTOR(3, 0, 0) rec[-10] rec[-1]
        DETECTOR(4, 1, 0) rec[-9] rec[-5]
        SHIFT_COORDS(0, 0, 1)
        DEPOLARIZE1(0.001) 0 7 9 16 3 5 11 13 1 2 4 6 8 10 12 14 15
        TICK
        REPEAT 3 {
            RX 0 7 9 16
            R 3 5 11 13
            X_ERROR(0.001) 3 5 11 13
            Z_ERROR(0.001) 0 7 9 16
            DEPOLARIZE1(0.001) 1 2 4 6 8 10 12 14 15
            TICK
            CX 0 1 4 3 7 8 9 10 12 11 14 13
            DEPOLARIZE2(0.001) 0 1 4 3 7 8 9 10 12 11 14 13
            DEPOLARIZE1(0.001) 2 5 6 15 16
            TICK
            CX 0 2 1 3 6 11 7 12 8 13 9 14
            DEPOLARIZE2(0.001) 0 2 1 3 6 11 7 12 8 13 9 14
            DEPOLARIZE1(0.001) 4 5 10 15 16
            TICK
            CX 7 2 8 3 9 4 10 5 15 13 16 14
            DEPOLARIZE2(0.001) 7 2 8 3 9 4 10 5 15 13 16 14
            DEPOLARIZE1(0.001) 0 1 6 11 12
            TICK
            CX 2 3 4 5 7 6 9 8 12 13 16 15
            DEPOLARIZE2(0.001) 2 3 4 5 7 6 9 8 12 13 16 15
            DEPOLARIZE1(0.001) 0 1 10 11 14
            TICK
            MX(0.001) 0 7 9 16
            M(0.001) 3 5 11 13
            DETECTOR(0, -1, 0) rec[-16] rec[-8]
            DETECTOR(1, 0, 0) rec[-12] rec[-4]
            DETECTOR(1, 2, 0) rec[-11] rec[-3]
            DETECTOR(2, -1, 0) rec[-15] rec[-7]
            DETECTOR(2, 1, 0) rec[-14] rec[-6]
            DETECTOR(3, -2, 0) rec[-10] rec[-2]
            DETECTOR(3, 0, 0) rec[-9] rec[-1]
            DETECTOR(4, 1, 0) rec[-13] rec[-5]
            SHIFT_COORDS(0, 0, 1)
            DEPOLARIZE1(0.001) 0 7 9 16 3 5 11 13 1 2 4 6 8 10 12 14 15
            TICK
        }
        MPP X15*X14 Z15*Z14*Z12*Z8 Z12*Z6 X14*X10*X8*X4 X12*X8*X6*X2 Z10*Z4 Z8*Z4*Z2*Z1 X2*X1 X10*Z6*X4*Z2*Y1
        OBSERVABLE_INCLUDE(0) rec[-1]
        DETECTOR(0, -1, 0) rec[-17] rec[-2]
        DETECTOR(1, 0, 0) rec[-13] rec[-3]
        DETECTOR(1, 2, 0) rec[-12] rec[-4]
        DETECTOR(2, -1, 0) rec[-16] rec[-5]
        DETECTOR(2, 1, 0) rec[-15] rec[-6]
        DETECTOR(3, -2, 0) rec[-11] rec[-7]
        DETECTOR(3, 0, 0) rec[-10] rec[-8]
        DETECTOR(4, 1, 0) rec[-14] rec[-9]
    """)
