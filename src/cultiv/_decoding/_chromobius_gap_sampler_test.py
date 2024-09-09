import sinter
import stim

from ._chromobius_gap_sampler import CompiledChromobiusGapSampler


def test_simple_case():
    circuit = stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 1) 1
        QUBIT_COORDS(2, 1) 2
        QUBIT_COORDS(2, 3) 3
        QUBIT_COORDS(3, 0) 4
        QUBIT_COORDS(3, 2) 5
        QUBIT_COORDS(3, 4) 6
        QUBIT_COORDS(4, 0) 7
        QUBIT_COORDS(4, 2) 8
        QUBIT_COORDS(4, 4) 9
        QUBIT_COORDS(4, 6) 10
        QUBIT_COORDS(5, 1) 11
        QUBIT_COORDS(5, 3) 12
        QUBIT_COORDS(5, 5) 13
        QUBIT_COORDS(6, 1) 14
        QUBIT_COORDS(6, 3) 15
        QUBIT_COORDS(7, 0) 16
        QUBIT_COORDS(7, 2) 17
        QUBIT_COORDS(8, 0) 18
        MPP X0*X4*X7*X16*X18
        OBSERVABLE_INCLUDE(0) rec[-1]
        TICK
        MPP X0*X1*X2*X4 X1*X2*X3*X5 Z0*Z1*Z2*Z4 Z1*Z2*Z3*Z5 X2*X4*X5*X7*X8*X11 X3*X5*X6*X8*X9*X12 X6*X9*X10*X13 Z2*Z4*Z5*Z7*Z8*Z11 Z3*Z5*Z6*Z8*Z9*Z12 Z6*Z9*Z10*Z13 X7*X11*X14*X16 X8*X11*X12*X14*X15*X17 X9*X12*X13*X15 Z7*Z11*Z14*Z16 Z8*Z11*Z12*Z14*Z15*Z17 Z9*Z12*Z13*Z15 X14*X16*X17*X18 Z14*Z16*Z17*Z18
        TICK
        DEPOLARIZE1(0.04) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18
        TICK
        MPP X0*X1*X2*X4
        DETECTOR(1, 0, 0, 0) rec[-19] rec[-1]
        MPP X1*X2*X3*X5
        DETECTOR(1, 2, 0, 1) rec[-19] rec[-1]
        MPP Z0*Z1*Z2*Z4
        DETECTOR(2, 0, 0, 3) rec[-19] rec[-1]
        MPP Z1*Z2*Z3*Z5
        DETECTOR(2, 2, 0, 4) rec[-19] rec[-1]
        MPP X2*X4*X5*X7*X8*X11
        DETECTOR(3, 1, 0, 2) rec[-19] rec[-1]
        MPP X3*X5*X6*X8*X9*X12
        DETECTOR(3, 3, 0, 0) rec[-19] rec[-1]
        MPP X6*X9*X10*X13
        DETECTOR(3, 5, 0, 1) rec[-19] rec[-1]
        MPP Z2*Z4*Z5*Z7*Z8*Z11
        DETECTOR(4, 1, 0, 5) rec[-19] rec[-1]
        MPP Z3*Z5*Z6*Z8*Z9*Z12
        DETECTOR(4, 3, 0, 3) rec[-19] rec[-1]
        MPP Z6*Z9*Z10*Z13
        DETECTOR(4, 5, 0, 4) rec[-19] rec[-1]
        MPP X7*X11*X14*X16
        DETECTOR(5, 0, 0, 0) rec[-19] rec[-1]
        MPP X8*X11*X12*X14*X15*X17
        DETECTOR(5, 2, 0, 1) rec[-19] rec[-1]
        MPP X9*X12*X13*X15
        DETECTOR(5, 4, 0, 2) rec[-19] rec[-1]
        MPP Z7*Z11*Z14*Z16
        DETECTOR(6, 0, 0, 3) rec[-19] rec[-1]
        MPP Z8*Z11*Z12*Z14*Z15*Z17
        DETECTOR(6, 2, 0, 4) rec[-19] rec[-1]
        MPP Z9*Z12*Z13*Z15
        DETECTOR(6, 4, 0, 5) rec[-19] rec[-1]
        MPP X14*X16*X17*X18
        DETECTOR(7, 1, 0, 2) rec[-19] rec[-1]
        MPP Z14*Z16*Z17*Z18
        DETECTOR(8, 1, 0, 5) rec[-19] rec[-1]
        TICK
        MPP X0*X4*X7*X16*X18
        OBSERVABLE_INCLUDE(0) rec[-1]
    """)
    decoder = CompiledChromobiusGapSampler(sinter.Task(
        circuit=circuit,
        detector_error_model=circuit.detector_error_model(),
        json_metadata={},
    ))
    assert decoder.gap_dem_base.approx_equals(stim.DetectorErrorModel("""
        error(0.013516) D0 D1
        error(0.013516) D0 D1 D2 D3
        error(0.013516) D0 D1 D2 D3 D4 D7
        error(0.013516) D0 D1 D4
        error(0.013516) D0 D2 D4 D7 D18 L0
        error(0.013516) D0 D2 D18 L0
        error(0.013516) D0 D4 D18 L0
        error(0.013516) D0 D18 L0
        error(0.013516) D1 D3 D4 D5 D7 D8
        error(0.013516) D1 D3 D5 D8
        error(0.013516) D1 D4 D5
        error(0.013516) D1 D5
        error(0.013516) D2
        error(0.013516) D2 D3
        error(0.013516) D2 D3 D7
        error(0.013516) D2 D7
        error(0.013516) D3 D7 D8
        error(0.013516) D3 D8
        error(0.013516) D4 D5 D7 D8 D11 D14
        error(0.013516) D4 D5 D11
        error(0.013516) D4 D7 D10 D11 D13 D14
        error(0.013516) D4 D7 D10 D13 D18 L0
        error(0.013516) D4 D10 D11
        error(0.013516) D4 D10 D18 L0
        error(0.013516) D5 D6
        error(0.013516) D5 D6 D8 D9
        error(0.013516) D5 D6 D8 D9 D12 D15
        error(0.013516) D5 D6 D12
        error(0.013516) D5 D8 D11 D12 D14 D15
        error(0.013516) D5 D11 D12
        error(0.013516) D6
        error(0.013516) D6 D9
        error(0.013516) D6 D9 D12 D15
        error(0.013516) D6 D12
        error(0.013516) D7 D8 D14
        error(0.013516) D7 D13
        error(0.013516) D7 D13 D14
        error(0.013516) D8 D9
        error(0.013516) D8 D9 D15
        error(0.013516) D8 D14 D15
        error(0.013516) D9
        error(0.013516) D9 D15
        error(0.013516) D10 D11 D13 D14 D16 D17
        error(0.013516) D10 D11 D16
        error(0.013516) D10 D13 D16 D17 D18 L0
        error(0.013516) D10 D16 D18 L0
        error(0.013516) D11 D12
        error(0.013516) D11 D12 D14 D15
        error(0.013516) D11 D14 D16 D17
        error(0.013516) D11 D16
        error(0.013516) D13 D14 D17
        error(0.013516) D13 D17
        error(0.013516) D14 D15
        error(0.013516) D14 D17
        error(0.013516) D16 D17 D18 L0
        error(0.013516) D16 D18 L0
        error(0.013516) D17
        detector(1, 0, 0, 0) D0
        detector(1, 2, 0, 1) D1
        detector(2, 0, 0, 3) D2
        detector(2, 2, 0, 4) D3
        detector(3, 1, 0, 2) D4
        detector(3, 3, 0, 0) D5
        detector(3, 5, 0, 1) D6
        detector(4, 1, 0, 5) D7
        detector(4, 3, 0, 3) D8
        detector(4, 5, 0, 4) D9
        detector(5, 0, 0, 0) D10
        detector(5, 2, 0, 1) D11
        detector(5, 4, 0, 2) D12
        detector(6, 0, 0, 3) D13
        detector(6, 2, 0, 4) D14
        detector(6, 4, 0, 5) D15
        detector(7, 1, 0, 2) D16
        detector(8, 1, 0, 5) D17
        detector(-9, -9, -9, 1) D18
    """), atol=1e-4)

    assert decoder.decode_dets(frozenset([])) == (False, 17)
    assert decoder.decode_dets(frozenset([0])) == (True, 9)
    assert decoder.decode_dets(frozenset([16])) == (True, 9)
    assert decoder.decode_dets(frozenset([17])) == (False, 17)
    assert decoder.decode_dets(frozenset([6])) == (False, 15)
    assert decoder.decode_dets(frozenset([10, 11, 16])) == (False, 11)
    assert decoder.decode_dets(frozenset([8, 14, 15])) == (False, 17)
