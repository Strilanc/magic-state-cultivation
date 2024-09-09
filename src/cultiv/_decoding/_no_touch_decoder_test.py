import stim

from cultiv._error_set import DemError
from ._no_touch_decoder import CompiledNoTouchDecoder


def test_simple_case():
    dem = stim.DetectorErrorModel("""
        error(0.1) D0 L0
        error(0.1) D0 D1 L1
        error(0.1) D1 D2 L2
        error(0.1) D2 D3 L3
        error(0.1) D3 D5 L4
        error(0.1) D3 D4 L5
        error(0.1) D4 D5 L6
        error(0.1) D5 D6 L0 L1 L2 L3 L4 L5 L6 L7
        error(0.1) D6 L7
    """)
    decoder = CompiledNoTouchDecoder(dem, discard_on_fail=True)
    assert decoder.lookup.det_to_local_lookup[0].neighborhood == {0, 1}
    assert decoder.lookup.det_to_local_lookup[1].neighborhood == {0, 1, 2}
    assert decoder.lookup.det_to_local_lookup[2].neighborhood == {1, 2, 3}
    assert decoder.lookup.det_to_local_lookup[3].neighborhood == {2, 3, 4, 5}
    assert decoder.lookup.det_to_local_lookup[4].neighborhood == {3, 4, 5}
    assert decoder.lookup.det_to_local_lookup[5].neighborhood == {3, 4, 5, 6}
    assert decoder.lookup.det_to_local_lookup[6].neighborhood == {5, 6}

    assert decoder.lookup.det_to_local_lookup[3].local_symptoms_to_error_index == {
        frozenset([2, 3]): 3,
        frozenset([3, 5]): 4,
        frozenset([3, 4]): 5,
    }

    assert decoder.decode_det_set(frozenset([])) == 0b00000000
    assert decoder.decode_det_set(frozenset([0])) == 0b00000001
    assert decoder.decode_det_set(frozenset([0, 1])) == 0b00000010
    assert decoder.decode_det_set(frozenset([1, 2])) == 0b00000100
    assert decoder.decode_det_set(frozenset([2, 3])) == 0b00001000
    assert decoder.decode_det_set(frozenset([3, 5])) == 0b00010000
    assert decoder.decode_det_set(frozenset([3, 4])) == 0b00100000
    assert decoder.decode_det_set(frozenset([4, 5])) == 0b01000000
    assert decoder.decode_det_set(frozenset([5, 6])) == 0b11111111
    assert decoder.decode_det_set(frozenset([6])) == 0b10000000
    assert decoder.decode_det_set(frozenset([1])) is None
    assert decoder.decode_det_set(frozenset([2])) is None
    assert decoder.decode_det_set(frozenset([3])) is None
    assert decoder.decode_det_set(frozenset([4])) is None
    assert decoder.decode_det_set(frozenset([5])) is None
    assert decoder.decode_det_set(frozenset([0, 6])) == 0b10000001
    assert decoder.decode_det_set(frozenset([0, 2, 3, 6])) == 0b10001001
    assert decoder.decode_det_set(frozenset([0, 3, 4, 6])) == 0b10100001
    assert decoder.decode_det_set(frozenset([0, 3, 5, 6])) is None
    assert decoder.decode_det_set(frozenset([0, 2, 3, 5, 6])) is None


def test_surface_code():
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_x",
        distance=5,
        rounds=5,
        after_clifford_depolarization=1e-3,
    )
    dem = circuit.detector_error_model()
    decoder = CompiledNoTouchDecoder(dem, discard_on_fail=True)
    assert decoder.decode_det_set(frozenset()) == 0
    err_list = []
    for inst in dem.flattened():
        if inst.type == 'error':
            err_list.append(DemError.from_error_instruction(inst))

    for err in err_list:
        assert decoder.decode_det_set(frozenset(err.det_list())) == err.obs
