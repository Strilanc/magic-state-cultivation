import stim

import cultiv
from cultiv.make_lifetime_plot import circuit_to_layers


def test_circuit_to_layers():
    circuit = cultiv.make_end2end_cultivation_circuit(dcolor=3, dsurface=6, basis='Y', r_growing=3, r_end=3, inject_style='unitary')
    layers = circuit_to_layers(circuit)
    assert layers[0] == stim.Circuit("""
        RX 12 18 20 5 4 6 30 1 7 29
        R 19 13 11
        TICK
        CX 12 19 18 11 20 13
        TICK
        CX 12 13 18 19
        TICK
        CX 12 11 20 19
        TICK
        CX 4 11 6 13 30 19
        TICK
        CX 11 4 13 6 19 30
        TICK
        CX 5 6 29 30
        TICK
        CX 5 4 30 29
        TICK
        CX 6 5
        TICK
        CX 4 5
        TICK
        S_DAG 5
        TICK
        CX 4 5
        TICK
        CX 6 5
        TICK
        CX 1 4 7 6
        TICK
        CX 4 1 6 7
        TICK
    """)
    assert layers[1] == stim.Circuit("""
        RX 4 19 6
        R 11 30 13 14
        TICK
        CX 4 11 6 13 7 14 19 30
        TICK
        CX 4 1 11 18 13 20 14 7 19 12
        TICK
        CX 4 5 11 12 13 14 19 20
        TICK
        CX 6 5 13 12 19 18 30 29
        TICK
        CX 5 6 12 13 18 19 29 30
        TICK
        CX 1 4 12 19 18 11 20 13
        TICK
        CX 5 4 12 11 14 13 20 19
        TICK
        CX 4 11 6 13 19 30
        TICK
        M 11 30 13 7
        MX 4 19 6
        DETECTOR(2, 0, 0) rec[-7]
        DETECTOR(4, 1, 0) rec[-6]
        DETECTOR(2, 2, 0) rec[-5]
        DETECTOR(1, 3, 0) rec[-4]
        DETECTOR(1, 0, 0) rec[-3]
        DETECTOR(3, 1, 0) rec[-2]
        DETECTOR(1, 2, 0) rec[-1]
        TICK
    """)
    assert layers[2] == stim.Circuit("""
        RX 30 19 11 4 13 2
        TICK
        S_DAG 1 5 12 14 18 20 29
        TICK
        CX 2 1 4 5 11 12 13 14 19 18 30 29
        TICK
        CX 5 2 12 13 19 30
        TICK
        CX 12 5 19 20
        TICK
        CX 12 19
        TICK
        MX 12
        TICK
    """)
    assert layers[3].approx_equals(stim.Circuit("""
        RX 12
        TICK
        CX 12 19
        TICK
        CX 12 5 19 20
        TICK
        CX 5 2 12 13 19 30
        TICK
        CX 2 1 4 5 11 12 13 14 19 18 30 29
        TICK
        MX 30 19 11 4 13 2
        TICK
        S 1 5 12 14 18 20 29
        OBSERVABLE_INCLUDE(0) rec[-7]
        OBSERVABLE_INCLUDE(0) rec[-6] rec[-5] rec[-2] rec[-1]
        DETECTOR(2.14286, 1, 1, -1, -9) rec[-9] rec[-8] rec[-7]
        DETECTOR(4, 1, 2) rec[-6]
        DETECTOR(3, 1, 2) rec[-5]
        DETECTOR(2, 1, 2) rec[-4] rec[-7]
        DETECTOR(1, 0, 2) rec[-3]
        DETECTOR(2, 2, 2) rec[-2]
        DETECTOR(0, 1, 2) rec[-1]
        TICK
    """), atol=1e03)
    assert layers[4] == stim.Circuit("""
        RX 22 31 33 41 43 45 50 52 54 61 63 66 68 73 74 0 4 6 8 10 19 21 23 24 26 28 47 49 51 53 55 65 67 69 75
        R 3 9 15 17 25 27 35 37 39 46 48 57 59 64 71 11 13 16 30 32 34 36 38 40 42 44 56 58 60 62 70 72
        TICK
        CX 0 1 4 11 6 13 8 9 10 17 19 30 21 32 23 34 24 25 28 39 55 54 57 56 69 68 71 70 75 74
        TICK
        CX 0 3 4 1 8 15 11 18 13 20 19 12 24 35 30 41 32 43 46 56 55 45 64 70 69 63 75 73
        TICK
        CX 4 5 10 9 11 12 13 14 17 18 19 20 21 22 28 29 30 31 32 33
        TICK
        CX 6 5 10 3 13 12 15 16 19 18 26 25 28 27 30 29 32 31 35 36 37 38 39 40 41 42 43 44 47 46 49 48 51 50 53 52 57 58 59 60 61 62 65 64 67 66 71 72
        TICK
        CX 1 4 12 19 14 21 18 11 20 13 26 15 27 16 28 17 46 36 47 37 48 38 49 39 50 40 51 41 52 42 53 43 54 44 64 58 65 59 66 60 67 61 68 62 74 72
        TICK
        CX 5 4 9 16 12 11 14 13 17 18 20 19 22 21 25 36 26 37 27 38 29 40 31 42 33 44 47 57 48 58 49 59 50 60 51 61 52 62 53 63 65 71 66 72 67 73
        TICK
        CX 3 4 5 6 12 13 18 19 20 21 22 23 29 30 31 32 33 34
        TICK
        CX 4 11 6 13 17 16 19 30 21 32 23 34 26 27 37 36 39 38 41 40 43 42 45 44 47 48 49 50 51 52 53 54 59 58 61 60 63 62 65 66 67 68 73 72
        TICK
        MX 0 4 6 8 10 19 21 23 24 26 28 47 49 51 53 55 65 67 69 75
        M 11 13 16 30 32 34 36 38 40 42 44 56 58 60 62 70 72
        OBSERVABLE_INCLUDE(0) rec[-37] rec[-34] rec[-33] rec[-29] rec[-28] rec[-27] rec[-26] rec[-25] rec[-21] rec[-13] rec[-12] rec[-9] rec[-8] rec[-7] rec[-4] rec[-3] rec[-1]
        DETECTOR(3, 5, 4, 0, 6) rec[-30]
        DETECTOR(1.125, 0.125, 4, -1, -9) rec[-44] rec[-36]
        DETECTOR(1.25, 1.9375, 4, -1, -9) rec[-44] rec[-35]
        DETECTOR(1.875, 0.125, 4, -1, -9) rec[-17]
        DETECTOR(2, 1.9375, 4, -1, -9) rec[-16]
        DETECTOR(3, -2, 4, 3, 7) rec[-15]
        DETECTOR(3, 0.9375, 4, -1, -9) rec[-44] rec[-32]
        DETECTOR(3, 3, 4, 0, 6) rec[-31]
        DETECTOR(3.75, 0.9375, 4, -1, -9) rec[-14]
        DETECTOR(5, -4, 4, 3, 7) rec[-11]
        DETECTOR(5, -2, 4, 3, 7) rec[-10]
        DETECTOR(6, 1, 4, 0, 6) rec[-24]
        DETECTOR(6, 3, 4, 0, 6) rec[-23]
        DETECTOR(6, 5, 4, 0, 6) rec[-22]
        DETECTOR(7, -4, 4, 3, 7) rec[-6]
        DETECTOR(7, -2, 4, 3, 7) rec[-5]
        DETECTOR(8, 1, 4, 0, 6) rec[-20]
        DETECTOR(8, 3, 4, 0, 6) rec[-19]
        DETECTOR(9, -2, 4, 3, 7) rec[-2]
        DETECTOR(10, 1, 4, 0, 6) rec[-18]
        TICK
    """)
    assert layers[6] == stim.Circuit("""
        RX 0 4 6 8 10 19 21 23 24 26 28 47 49 51 53 55 65 67 69 75
        R 11 13 16 30 32 34 36 38 40 42 44 56 58 60 62 70 72
        TICK
        CX 0 1 4 11 6 13 8 9 10 17 19 30 21 32 23 34 24 25 28 39 55 54 57 56 69 68 71 70 75 74
        TICK
        CX 0 3 4 1 8 15 11 18 13 20 19 12 24 35 30 41 32 43 46 56 55 45 64 70 69 63 75 73
        TICK
        CX 4 5 10 9 11 12 13 14 17 18 19 20 21 22 28 29 30 31 32 33
        TICK
        CX 6 5 10 3 13 12 15 16 19 18 26 25 28 27 30 29 32 31 35 36 37 38 39 40 41 42 43 44 47 46 49 48 51 50 53 52 57 58 59 60 61 62 65 64 67 66 71 72
        TICK
        CX 1 4 12 19 14 21 18 11 20 13 26 15 27 16 28 17 46 36 47 37 48 38 49 39 50 40 51 41 52 42 53 43 54 44 64 58 65 59 66 60 67 61 68 62 74 72
        TICK
        CX 5 4 9 16 12 11 14 13 17 18 20 19 22 21 25 36 26 37 27 38 29 40 31 42 33 44 47 57 48 58 49 59 50 60 51 61 52 62 53 63 65 71 66 72 67 73
        TICK
        CX 3 4 5 6 12 13 18 19 20 21 22 23 29 30 31 32 33 34
        TICK
        CX 4 11 6 13 17 16 19 30 21 32 23 34 26 27 37 36 39 38 41 40 43 42 45 44 47 48 49 50 51 52 53 54 59 58 61 60 63 62 65 66 67 68 73 72
        TICK
        MX 0 4 6 8 10 19 21 23 24 26 28 47 49 51 53 55 65 67 69 75
        M 11 13 16 30 32 34 36 38 40 42 44 56 58 60 62 70 72
        DETECTOR(3, 5, 6, 0, 6) rec[-30]
        DETECTOR(0, -1, 6, 0, 6) rec[-74] rec[-37]
        DETECTOR(1, 0, 6, 0, 0) rec[-73] rec[-72] rec[-36]
        DETECTOR(1, 2, 6, 1, 1) rec[-73] rec[-35]
        DETECTOR(2, -3, 6, 0, 6) rec[-71] rec[-34]
        DETECTOR(2, -1, 6, 0, 6) rec[-73] rec[-70] rec[-33]
        DETECTOR(2, 0, 6, 3, 3) rec[-54] rec[-17]
        DETECTOR(2, 2, 6, 4, 4) rec[-53] rec[-16]
        DETECTOR(3, -2, 6, 3, 7) rec[-52] rec[-15]
        DETECTOR(3, 1, 6, 2, 2) rec[-68] rec[-32]
        DETECTOR(3, 3, 6, 0, 6) rec[-31]
        DETECTOR(4, -5, 6, 0, 6) rec[-66] rec[-29]
        DETECTOR(4, -3, 6, 0, 6) rec[-65] rec[-28]
        DETECTOR(4, -1, 6, 1, 1) rec[-69] rec[-64] rec[-27]
        DETECTOR(4, 1, 6, 5, 5) rec[-51] rec[-14]
        DETECTOR(4, 3, 6, 3, 3) rec[-50] rec[-13]
        DETECTOR(4, 5, 6, 3, 7) rec[-49] rec[-12]
        DETECTOR(5, -4, 6, 3, 7) rec[-48] rec[-11]
        DETECTOR(5, -2, 6, 3, 7) rec[-47] rec[-10]
        DETECTOR(5, 0, 6, 3, 7) rec[-46] rec[-9]
        DETECTOR(5, 2, 6, 3, 7) rec[-45] rec[-8]
        DETECTOR(5, 4, 6, 3, 7) rec[-44] rec[-7]
        DETECTOR(6, -3, 6, 0, 6) rec[-63] rec[-26]
        DETECTOR(6, -1, 6, 0, 6) rec[-62] rec[-25]
        DETECTOR(6, 1, 6, 0, 6) rec[-61] rec[-24]
        DETECTOR(6, 3, 6, 0, 6) rec[-60] rec[-23]
        DETECTOR(6, 5, 6, 0, 6) rec[-59] rec[-22]
        DETECTOR(7, -4, 6, 3, 7) rec[-43] rec[-6]
        DETECTOR(7, -2, 6, 3, 7) rec[-42] rec[-5]
        DETECTOR(7, 0, 6, 3, 7) rec[-41] rec[-4]
        DETECTOR(7, 2, 6, 3, 7) rec[-40] rec[-3]
        DETECTOR(8, -1, 6, 0, 6) rec[-58] rec[-21]
        DETECTOR(8, 1, 6, 0, 6) rec[-57] rec[-20]
        DETECTOR(8, 3, 6, 0, 6) rec[-56] rec[-19]
        DETECTOR(9, -2, 6, 3, 7) rec[-39] rec[-2]
        DETECTOR(9, 0, 6, 3, 7) rec[-38] rec[-1]
        DETECTOR(10, 1, 6, 0, 6) rec[-55] rec[-18]
        TICK
    """)
    assert layers[7] == stim.Circuit("""
        RX 0 4 6 8 10 19 21 23 24 26 28 47 49 51 53 55 65 67 69 75
        R 7 11 13 16 30 32 34 36 38 40 42 44 56 58 60 62 70 72
        TICK
        CX 0 1 4 11 6 13 8 9 10 17 19 30 21 32 23 34 24 25 28 39 55 54 57 56 69 68 71 70 75 74
        TICK
        CX 0 3 8 15 13 20 19 12 24 35 30 41 32 43 46 56 55 45 64 70 69 63 75 73
        TICK
        CX 6 7 10 9 13 14 17 18 19 20 21 22 28 29 30 31 32 33
        TICK
        CX 6 5 10 3 13 12 14 7 15 16 19 18 26 25 28 27 30 29 32 31 35 36 37 38 39 40 41 42 43 44 47 46 49 48 51 50 53 52 57 58 59 60 61 62 65 64 67 66 71 72
        TICK
        CX 1 4 5 6 14 21 18 11 20 13 26 15 27 16 28 17 46 36 47 37 48 38 49 39 50 40 51 41 52 42 53 43 54 44 64 58 65 59 66 60 67 61 68 62 74 72
        TICK
        CX 5 4 6 7 9 16 12 11 17 18 22 21 25 36 26 37 27 38 29 40 31 42 33 44 47 57 48 58 49 59 50 60 51 61 52 62 53 63 65 71 66 72 67 73
        TICK
        CX 3 4 5 6 12 13 18 19 20 21 22 23 29 30 31 32 33 34
        TICK
        CX 4 11 6 13 17 16 19 30 21 32 23 34 26 27 37 36 39 38 41 40 43 42 45 44 47 48 49 50 51 52 53 54 59 58 61 60 63 62 65 66 67 68 73 72
        TICK
        MX 0 4 6 8 10 19 21 23 24 26 28 47 49 51 53 55 65 67 69 75
        M 7 11 13 16 30 32 34 36 38 40 42 44 56 58 60 62 70 72
        DETECTOR(1, 0, 7, 0, 6) rec[-37]
        DETECTOR(3, 5, 7, 0, 6) rec[-31]
        DETECTOR(0, -1, 7, 0, 6) rec[-75] rec[-38]
        DETECTOR(1, 2, 7, 1, 1) rec[-74] rec[-36]
        DETECTOR(2, -3, 7, 0, 6) rec[-72] rec[-35]
        DETECTOR(2, -1, 7, 0, 6) rec[-74] rec[-71] rec[-34]
        DETECTOR(2, 0, 7, 3, 3) rec[-55] rec[-17]
        DETECTOR(2, 2, 7, 4, 4) rec[-54] rec[-18] rec[-16]
        DETECTOR(3, -2, 7, 3, 7) rec[-53] rec[-15]
        DETECTOR(3, 1, 7, 2, 2) rec[-69] rec[-33]
        DETECTOR(3, 3, 7, 0, 6) rec[-32]
        DETECTOR(4, -5, 7, 0, 6) rec[-67] rec[-30]
        DETECTOR(4, -3, 7, 0, 6) rec[-66] rec[-29]
        DETECTOR(4, -1, 7, 1, 1) rec[-70] rec[-65] rec[-28]
        DETECTOR(4, 1, 7, 5, 5) rec[-52] rec[-16] rec[-14]
        DETECTOR(4, 3, 7, 3, 3) rec[-51] rec[-13]
        DETECTOR(4, 5, 7, 3, 7) rec[-50] rec[-12]
        DETECTOR(5, -4, 7, 3, 7) rec[-49] rec[-11]
        DETECTOR(5, -2, 7, 3, 7) rec[-48] rec[-10]
        DETECTOR(5, 0, 7, 3, 7) rec[-47] rec[-9]
        DETECTOR(5, 2, 7, 3, 7) rec[-46] rec[-8]
        DETECTOR(5, 4, 7, 3, 7) rec[-45] rec[-7]
        DETECTOR(6, -3, 7, 0, 6) rec[-64] rec[-27]
        DETECTOR(6, -1, 7, 0, 6) rec[-63] rec[-26]
        DETECTOR(6, 1, 7, 0, 6) rec[-62] rec[-25]
        DETECTOR(6, 3, 7, 0, 6) rec[-61] rec[-24]
        DETECTOR(6, 5, 7, 0, 6) rec[-60] rec[-23]
        DETECTOR(7, -4, 7, 3, 7) rec[-44] rec[-6]
        DETECTOR(7, -2, 7, 3, 7) rec[-43] rec[-5]
        DETECTOR(7, 0, 7, 3, 7) rec[-42] rec[-4]
        DETECTOR(7, 2, 7, 3, 7) rec[-41] rec[-3]
        DETECTOR(8, -1, 7, 0, 6) rec[-59] rec[-22]
        DETECTOR(8, 1, 7, 0, 6) rec[-58] rec[-21]
        DETECTOR(8, 3, 7, 0, 6) rec[-57] rec[-20]
        DETECTOR(9, -2, 7, 3, 7) rec[-40] rec[-2]
        DETECTOR(9, 0, 7, 3, 7) rec[-39] rec[-1]
        DETECTOR(10, 1, 7, 0, 6) rec[-56] rec[-19]
        TICK
    """)
