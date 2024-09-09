from typing import Literal

import stim

import gen
from ._color_code import make_color_code, make_chunk_color_code_superdense_cycle
from ._injection_stage import make_chunk_d3_init_unitary, make_chunk_d3_init_degenerate_teleport, \
    make_chunk_d3_init_bell_pair_growth


def make_chunk_d3_double_cat_check() -> gen.Chunk:
    chunk = gen.Chunk.from_circuit_with_mpp_boundaries(stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        QUBIT_COORDS(1, 0) 2
        QUBIT_COORDS(1, 1) 3
        QUBIT_COORDS(2, 0) 4
        QUBIT_COORDS(2, 1) 5
        QUBIT_COORDS(2, 2) 6
        QUBIT_COORDS(2, 3) 7
        QUBIT_COORDS(3, 0) 8
        QUBIT_COORDS(3, 1) 9
        QUBIT_COORDS(3, 2) 10
        QUBIT_COORDS(4, 0) 11
        QUBIT_COORDS(4, 1) 12
        #!pragma POLYGON(0,0,1,0.25) 8 11 10 5
        #!pragma POLYGON(0,1,0,0.25) 7 10 5 3
        #!pragma POLYGON(1,0,0,0.25) 3 5 8 0
        TICK
        MPP Y0*Y8*Y11*Y10*Y7*Y5*Y3
        TICK
        MPP X8*X0*X3*X5
        TICK
        MPP X3*X5*X10*X7
        TICK
        MPP X11*X8*X5*X10
        TICK
        MPP Z8*Z0*Z3*Z5
        TICK
        MPP Z7*Z10*Z5*Z3
        TICK
        MPP Z11*Z8*Z5*Z10
        TICK
        RX 12 9 4 2 6 1
        TICK
        # CZ sweep[7] 0 sweep[7] 3 sweep[7] 5 sweep[7] 8
        # CX sweep[8] 0 sweep[8] 3 sweep[8] 5 sweep[8] 8
        # CZ sweep[9] 3 sweep[9] 5 sweep[9] 7 sweep[9] 10
        # CX sweep[10] 3 sweep[10] 5 sweep[10] 7 sweep[10] 10
        # CZ sweep[11] 5 sweep[11] 8 sweep[11] 10 sweep[11] 11
        # CX sweep[12] 5 sweep[12] 8 sweep[12] 10 sweep[12] 11
        S_DAG 0 3 5 7 8 10 11
        # CZ sweep[0] 0
        # CZ sweep[1] 3
        # CZ sweep[2] 5
        # CZ sweep[3] 7
        # CZ sweep[4] 8
        # CZ sweep[5] 10
        # CZ sweep[6] 11
        TICK
        CX 1 0 9 8 6 7 4 5 2 3 12 11
        TICK
        CX 3 1 5 6 9 12
        TICK
        CX 5 3 9 10
        TICK
        CX 5 9
        TICK
        # CZ sweep[0] 5
        # CZ sweep[1] 5
        # CZ sweep[2] 5
        # CZ sweep[3] 5
        # CZ sweep[4] 5
        # CZ sweep[5] 5
        # CZ sweep[6] 5
        MX 5
        DETECTOR(0, 0, 0) rec[-1] rec[-8]
        TICK
        RX 5
        # CZ sweep[0] 5
        # CZ sweep[1] 5
        # CZ sweep[2] 5
        # CZ sweep[3] 5
        # CZ sweep[4] 5
        # CZ sweep[5] 5
        # CZ sweep[6] 5
        TICK
        CX 5 9
        TICK
        CX 5 3 9 10
        TICK
        CX 3 1 5 6 9 12
        TICK
        CX 1 0 9 8 6 7 12 11 4 5 2 3
        TICK
        # CZ sweep[0] 0
        # CZ sweep[1] 3
        # CZ sweep[2] 5
        # CZ sweep[3] 7
        # CZ sweep[4] 8
        # CZ sweep[5] 10
        # CZ sweep[6] 11
        # CZ sweep[13] 0 sweep[13] 3 sweep[13] 5 sweep[13] 8
        # CX sweep[14] 0 sweep[14] 3 sweep[14] 5 sweep[14] 8
        # CZ sweep[15] 3 sweep[15] 5 sweep[15] 7 sweep[15] 10
        # CX sweep[16] 3 sweep[16] 5 sweep[16] 7 sweep[16] 10
        # CZ sweep[17] 5 sweep[17] 8 sweep[17] 10 sweep[17] 11
        # CX sweep[18] 5 sweep[18] 8 sweep[18] 10 sweep[18] 11
        S 5 10 11 8 7 3 0
        TICK
        MX 12 9 4 2 6 1
        DETECTOR(4, 1, 1) rec[-6]
        DETECTOR(3, 1, 1) rec[-5]
        DETECTOR(2, 1, 1) rec[-4] rec[-7]
        DETECTOR(1, 0, 1) rec[-3]
        DETECTOR(2, 2, 1) rec[-2]
        DETECTOR(0, 1, 1) rec[-1]
        TICK
        MPP X8*X0*X3*X5
        DETECTOR(3, 0, 2) rec[-1] rec[-8] rec[-14]
        TICK
        MPP X7*X10*X5*X3
        DETECTOR(1, 1, 3) rec[-1] rec[-9] rec[-14]
        TICK
        MPP X11*X8*X5*X10
        DETECTOR(4, 0, 4) rec[-1] rec[-10] rec[-14]
        TICK
        MPP Z8*Z0*Z3*Z5
        DETECTOR(3, 0, 5) rec[-1] rec[-14]
        TICK
        MPP Z7*Z10*Z5*Z3
        DETECTOR(2, 3, 6) rec[-1] rec[-14]
        TICK
        MPP Z11*Z8*Z5*Z10
        DETECTOR(4, 0, 7) rec[-1] rec[-14]
        TICK
        MPP Y0*Y8*Y11*Y10*Y7*Y5*Y3
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-8] rec[-9] rec[-12] rec[-13]
    """))
    return chunk.with_flag_added_to_all_flows('stage=cultivation')


def make_chunk_d3_to_d5_color_code() -> gen.Chunk:
    chunk = gen.Chunk.from_circuit_with_mpp_boundaries(stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(2, 1) 2
        QUBIT_COORDS(2, 3) 3
        QUBIT_COORDS(3, 0) 4
        QUBIT_COORDS(3, 2) 5
        QUBIT_COORDS(3, 4) 6
        QUBIT_COORDS(3, 5) 7
        QUBIT_COORDS(4, 0) 8
        QUBIT_COORDS(4, 2) 9
        QUBIT_COORDS(4, 4) 10
        QUBIT_COORDS(4, 5) 11
        QUBIT_COORDS(4, 6) 12
        QUBIT_COORDS(5, 1) 13
        QUBIT_COORDS(5, 2) 14
        QUBIT_COORDS(5, 3) 15
        QUBIT_COORDS(5, 4) 16
        QUBIT_COORDS(5, 5) 17
        QUBIT_COORDS(6, 1) 18
        QUBIT_COORDS(6, 2) 19
        QUBIT_COORDS(6, 3) 20
        QUBIT_COORDS(7, 0) 21
        QUBIT_COORDS(7, 2) 22
        QUBIT_COORDS(8, 0) 23
        #!pragma POLYGON(0,0,1,0.25) 4 8 5 2
        #!pragma POLYGON(0,1,0,0.25) 3 5 2 1
        #!pragma POLYGON(1,0,0,0.25) 1 2 4 0
        TICK
        MPP Y0*Y4*Y8*Y5*Y3*Y2*Y1
        TICK
        MPP X0*X4*X2*X1
        TICK
        MPP X1*X2*X5*X3
        TICK
        MPP X4*X8*X5*X2
        TICK
        MPP Z0*Z4*Z2*Z1
        TICK
        MPP Z1*Z2*Z5*Z3
        TICK
        MPP Z4*Z8*Z5*Z2
        TICK
        #!pragma POLYGON(0,0,1,0.25) 15 20 17 10
        #!pragma POLYGON(0,0,1,0.25) 21 23 22 18
        #!pragma POLYGON(0,0,1,0.25) 4 8 13 9 5 2
        #!pragma POLYGON(0,1,0,0.25) 13 18 22 20 15 9
        #!pragma POLYGON(0,1,0,0.25) 6 10 17 12
        #!pragma POLYGON(0,1,0,0.25) 1 2 5 3
        #!pragma POLYGON(1,0,0,0.25) 0 4 2 1
        #!pragma POLYGON(1,0,1,0.25) 13 9
        #!pragma POLYGON(1,0,1,0.25) 20 15
        #!pragma POLYGON(1,0,1,0.25) 10 17
        #!pragma POLYGON(1,0,1,0.25) 6 12
        #!pragma POLYGON(1,0,1,0.25) 18 22
        #!pragma POLYGON(1,0,1,0.25) 23 21
        TICK
        R 9 13 18 22 23 12 6 11 10 17 20
        RX 14 19 21 7 16 15
        TICK
        CX 14 9 19 22 21 23 15 20 16 10 7 11
        TICK
        CX 14 13 19 18 16 17 11 12 7 6
        TICK
        CX 13 14 18 19 17 16 12 11 6 7
        TICK
        M 14 19 16 11 7
        DETECTOR(5, 2, 0) rec[-5]
        DETECTOR(6, 2, 0) rec[-4]
        DETECTOR(5, 4, 0) rec[-3]
        DETECTOR(4, 5, 0) rec[-2]
        DETECTOR(3, 5, 0) rec[-1]
        TICK
        #!pragma POLYGON(0,0,1,0.25) 4 8 13 9 5 2
        #!pragma POLYGON(0,0,1,0.25) 15 20 17 10
        #!pragma POLYGON(0,0,1,0.25) 21 23 22 18
        #!pragma POLYGON(0,1,0,0.25) 3 5 2 1
        #!pragma POLYGON(0,1,0,0.25) 13 18 22 20 15 9
        #!pragma POLYGON(0,1,0,0.25) 6 10 17 12
        #!pragma POLYGON(1,0,0,0.25) 1 2 4 0
        TICK
        MPP X0*X4*X2*X1 X8*X21*X18*X13 X5*X9*X15*X10*X6*X3
        DETECTOR(0, 0, 1) rec[-3] rec[-14]
        TICK
        MPP X13*X18*X22*X20*X15*X9 X1*X2*X5*X3 X6*X10*X17*X12
        DETECTOR(5, 1, 2) rec[-3]
        DETECTOR(1, 1, 2) rec[-2] rec[-16]
        DETECTOR(3, 4, 2) rec[-1]
        TICK
        MPP X4*X8*X13*X9*X5*X2 X15*X20*X17*X10 X21*X23*X22*X18
        DETECTOR(3, 0, 3) rec[-3] rec[-18]
        DETECTOR(5, 3, 3) rec[-2]
        DETECTOR(7, 0, 3) rec[-1]
        TICK
        MPP Z0*Z4*Z2*Z1 Z8*Z21*Z18*Z13 Z5*Z9*Z15*Z10*Z6*Z3
        DETECTOR(0, 0, 4) rec[-3] rec[-20]
        TICK
        MPP Z13*Z18*Z22*Z20*Z15*Z9 Z1*Z2*Z5*Z3 Z6*Z10*Z17*Z12
        DETECTOR(5, 1, 5) rec[-3]
        DETECTOR(1, 1, 5) rec[-2] rec[-22]
        DETECTOR(3, 4, 5) rec[-1]
        TICK
        MPP Z4*Z8*Z13*Z9*Z5*Z2 Z15*Z20*Z17*Z10 Z21*Z23*Z22*Z18
        DETECTOR(3, 0, 6) rec[-3] rec[-24]
        DETECTOR(5, 3, 6) rec[-2]
        DETECTOR(7, 0, 6) rec[-1]
        TICK
        MPP Y1*Y2*Y0*Y4*Y8*Y13*Y21*Y18*Y23*Y22*Y20*Y15*Y17*Y12*Y10*Y6*Y3*Y5*Y9
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-31]
    """))
    included_out = {
        flow.end
        for flow in chunk.flows
        if flow.end
    }
    discarded_out = {
        tile.to_data_pauli_string()
        for tile in make_color_code(5).tiles
    } - included_out
    return chunk.with_flag_added_to_all_flows('stage=cultivation').with_edits(
        discarded_outputs=discarded_out,
    )


def make_chunk_d5_double_cat_check() -> gen.Chunk:
    chunk = gen.Chunk.from_circuit_with_mpp_boundaries(stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        QUBIT_COORDS(1, 0) 2
        QUBIT_COORDS(1, 1) 3
        QUBIT_COORDS(2, 0) 4
        QUBIT_COORDS(2, 1) 5
        QUBIT_COORDS(2, 2) 6
        QUBIT_COORDS(2, 3) 7
        QUBIT_COORDS(2, 4) 8
        QUBIT_COORDS(3, 0) 9
        QUBIT_COORDS(3, 1) 10
        QUBIT_COORDS(3, 2) 11
        QUBIT_COORDS(3, 3) 12
        QUBIT_COORDS(3, 4) 13
        QUBIT_COORDS(4, 0) 14
        QUBIT_COORDS(4, 1) 15
        QUBIT_COORDS(4, 2) 16
        QUBIT_COORDS(4, 3) 17
        QUBIT_COORDS(4, 4) 18
        QUBIT_COORDS(4, 5) 19
        QUBIT_COORDS(4, 6) 20
        QUBIT_COORDS(5, 0) 21
        QUBIT_COORDS(5, 1) 22
        QUBIT_COORDS(5, 2) 23
        QUBIT_COORDS(5, 3) 24
        QUBIT_COORDS(5, 4) 25
        QUBIT_COORDS(5, 5) 26
        QUBIT_COORDS(5, 6) 27
        QUBIT_COORDS(6, 0) 28
        QUBIT_COORDS(6, 1) 29
        QUBIT_COORDS(6, 2) 30
        QUBIT_COORDS(6, 3) 31
        QUBIT_COORDS(7, 0) 32
        QUBIT_COORDS(7, 1) 33
        QUBIT_COORDS(7, 2) 34
        QUBIT_COORDS(7, 3) 35
        QUBIT_COORDS(8, 0) 36
        QUBIT_COORDS(8, 1) 37
        #!pragma POLYGON(0,0,1,0.25) 9 14 22 16 11 5
        #!pragma POLYGON(0,0,1,0.25) 29 32 36 34
        #!pragma POLYGON(0,0,1,0.25) 24 31 26 18
        #!pragma POLYGON(0,1,0,0.25) 7 11 5 3
        #!pragma POLYGON(0,1,0,0.25) 22 29 34 31 24 16
        #!pragma POLYGON(0,1,0,0.25) 13 18 26 20
        #!pragma POLYGON(1,0,0,0.25) 3 5 9 0
        #!pragma POLYGON(1,0,0,0.25) 14 32 29 22
        #!pragma POLYGON(1,0,0,0.25) 11 16 24 18 13 7
        TICK
        MPP X0*X9*X5*X3 X14*X32*X29*X22 X11*X16*X24*X18*X13*X7
        TICK
        MPP X22*X29*X34*X31*X24*X16 X3*X5*X11*X7 X13*X18*X26*X20
        TICK
        MPP X9*X14*X22*X16*X11*X5 X24*X31*X26*X18 X29*X32*X36*X34
        TICK
        MPP Z0*Z9*Z5*Z3 Z14*Z32*Z29*Z22 Z11*Z16*Z24*Z18*Z13*Z7
        TICK
        MPP Z22*Z29*Z34*Z31*Z24*Z16 Z3*Z5*Z11*Z7 Z13*Z18*Z26*Z20
        TICK
        MPP Z9*Z14*Z22*Z16*Z11*Z5 Z24*Z31*Z26*Z18 Z29*Z32*Z36*Z34
        TICK
        MPP Y3*Y5*Y0*Y9*Y14*Y22*Y32*Y29*Y34*Y31*Y24*Y26*Y20*Y18*Y13*Y7*Y11*Y16*Y36
        TICK
        RX 21 28 2 4 6 8 17 19 35 1 10 15 33 30 12 23 25 27 37
        TICK
        S_DAG 9 11 13 14 16 18 20 22 24 26 29 31 34 36 32 3 5 7 0
        TICK
        CX 1 0 2 3 4 5 10 9 15 14 21 22 28 29 33 32 30 31 35 34 23 16 17 24 25 18 19 26 27 20 6 11 12 7 8 13 37 36
        TICK
        CX 12 13 3 1 26 27 33 37 30 34
        TICK
        CX 25 26 11 12 5 3 29 33 23 30
        TICK
        CX 24 25 10 11 22 29
        TICK
        CX 10 5 23 24
        TICK
        CX 15 10 22 23
        TICK
        CX 22 15
        TICK
        MX 22
        DETECTOR(1, 1, 0) rec[-1] rec[-2]
        TICK
        RX 22
        TICK
        CX 22 15
        TICK
        CX 15 10 22 23
        TICK
        CX 10 5 23 24
        TICK
        CX 24 25 10 11 22 29
        TICK
        CX 25 26 11 12 5 3 29 33 23 30
        TICK
        CX 12 13 3 1 26 27 33 37 30 34
        TICK
        CX 1 0 2 3 4 5 10 9 15 14 21 22 28 29 33 32 30 31 35 34 23 16 17 24 25 18 19 26 27 20 12 7 8 13 6 11 37 36
        TICK
        S 9 11 13 14 16 18 20 22 24 26 29 31 34 32 3 5 7 0 36
        TICK
        MX 17 19 6 8 2 4 21 28 35 1 10 15 33 30 12 23 25 27 37
        DETECTOR(4, 3, 1) rec[-19]
        DETECTOR(4, 5, 1) rec[-18]
        DETECTOR(2, 2, 1) rec[-17]
        DETECTOR(2, 4, 1) rec[-16]
        DETECTOR(1, 0, 1) rec[-15]
        DETECTOR(2, 0, 1) rec[-14]
        DETECTOR(5, 1, 1) rec[-13] rec[-20]
        DETECTOR(6, 0, 1) rec[-12]
        DETECTOR(7, 3, 1) rec[-11]
        DETECTOR(0, 1, 1) rec[-10]
        DETECTOR(3, 1, 1) rec[-9]
        DETECTOR(4, 1, 1) rec[-8]
        DETECTOR(7, 1, 1) rec[-7]
        DETECTOR(6, 2, 1) rec[-6]
        DETECTOR(3, 3, 1) rec[-5]
        DETECTOR(5, 2, 1) rec[-4]
        DETECTOR(5, 4, 1) rec[-3]
        DETECTOR(5, 6, 1) rec[-2]
        DETECTOR(8, 1, 1) rec[-1]
        TICK
        MPP X0*X9*X5*X3 X14*X32*X29*X22 X11*X16*X24*X18*X13*X7
        DETECTOR(0, 0, 2) rec[-3] rec[-42]
        DETECTOR(4, 0, 2) rec[-2] rec[-23] rec[-41]
        DETECTOR(3, 2, 2) rec[-1] rec[-40]
        TICK
        MPP X22*X29*X34*X31*X24*X16 X3*X5*X11*X7 X13*X18*X26*X20
        DETECTOR(5, 1, 3) rec[-3] rec[-26] rec[-42]
        DETECTOR(1, 1, 3) rec[-2] rec[-41]
        DETECTOR(3, 4, 3) rec[-1] rec[-40]
        TICK
        MPP X9*X14*X22*X16*X11*X5 X24*X31*X26*X18 X32*X36*X34*X29
        DETECTOR(3, 0, 4) rec[-3] rec[-29] rec[-42]
        DETECTOR(5, 3, 4) rec[-2] rec[-41]
        DETECTOR(6, 1, 4) rec[-1] rec[-40]
        TICK
        MPP Z0*Z9*Z5*Z3 Z14*Z32*Z29*Z22 Z11*Z16*Z24*Z18*Z13*Z7
        DETECTOR(0, 0, 5) rec[-3] rec[-42]
        DETECTOR(4, 0, 5) rec[-2] rec[-41]
        DETECTOR(3, 2, 5) rec[-1] rec[-40]
        TICK
        MPP Z22*Z29*Z34*Z31*Z24*Z16 Z3*Z5*Z11*Z7 Z13*Z18*Z26*Z20
        DETECTOR(5, 1, 6) rec[-3] rec[-42]
        DETECTOR(1, 1, 6) rec[-2] rec[-41]
        DETECTOR(3, 4, 6) rec[-1] rec[-40]
        TICK
        MPP Z9*Z14*Z22*Z16*Z11*Z5 Z24*Z31*Z26*Z18 Z32*Z36*Z34*Z29
        DETECTOR(3, 0, 7) rec[-3] rec[-42]
        DETECTOR(5, 3, 7) rec[-2] rec[-41]
        DETECTOR(6, 1, 7) rec[-1] rec[-40]
        TICK
        MPP Y3*Y5*Y0*Y9*Y14*Y22*Y32*Y29*Y36*Y34*Y31*Y24*Y26*Y20*Y18*Y13*Y7*Y11*Y16
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-20] rec[-21] rec[-22] rec[-23] rec[-24] rec[-25] rec[-26] rec[-27] rec[-28] rec[-29]
    """))
    return chunk.with_flag_added_to_all_flows('stage=cultivation')


def make_inject_and_cultivate_chunks_d3(*, style: Literal['degenerate', 'bell', 'unitary']) -> list[gen.Chunk | gen.ChunkReflow]:
    if style == 'degenerate':
        return [
            make_chunk_d3_init_degenerate_teleport().with_obs_flows_as_det_flows(),
            make_chunk_color_code_superdense_cycle(make_color_code(3, obs_location='all'), obs_basis='Y').with_obs_flows_as_det_flows().with_flag_added_to_all_flows('stage=cultivation'),
            make_chunk_d3_double_cat_check(),
        ]
    elif style == 'bell':
        return [
            make_chunk_d3_init_bell_pair_growth().with_obs_flows_as_det_flows(),
            make_chunk_color_code_superdense_cycle(make_color_code(3, obs_location='all'), obs_basis='Y').with_obs_flows_as_det_flows().with_flag_added_to_all_flows('stage=cultivation'),
            make_chunk_d3_double_cat_check(),
        ]
    elif style == 'unitary':
        return [
            make_chunk_d3_init_unitary().with_obs_flows_as_det_flows(),
            # Note: A modified superdense cycle is included in the init circuit.
            make_chunk_d3_double_cat_check(),
        ]
    else:
        raise NotImplementedError(f'style=')


def make_inject_and_cultivate_chunks_d5(*, style: Literal['degenerate', 'bell', 'unitary']) -> list[gen.Chunk | gen.ChunkReflow]:
    return [
        *[chunk.with_obs_flows_as_det_flows() for chunk in make_inject_and_cultivate_chunks_d3(style=style)],
        make_chunk_d3_to_d5_color_code().with_obs_flows_as_det_flows(),
        make_chunk_color_code_superdense_cycle(make_color_code(5, obs_location='all'), obs_basis='Y').time_reversed().with_obs_flows_as_det_flows().with_flag_added_to_all_flows('stage=cultivation') * 3,
        make_chunk_d5_double_cat_check(),
    ]
