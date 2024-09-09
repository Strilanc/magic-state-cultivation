import numpy as np
import stim

from ._error_set import DemError, \
    int_to_flipped_bits, iter_pair_chunks, iter_triplet_chunks, DemErrorSet, chance_of_exactly_1, chance_of_exactly_0


def test_int_to_flipped_bits():
    assert int_to_flipped_bits(-1) == [-1]
    assert int_to_flipped_bits(-1 ^ (1 << 1000)) == [-1, 1000]
    assert int_to_flipped_bits(0) == []
    assert int_to_flipped_bits(1) == [0]
    assert int_to_flipped_bits(2) == [1]
    assert int_to_flipped_bits(3) == [1, 0]
    assert int_to_flipped_bits(4) == [2]
    assert int_to_flipped_bits(0b100000101) == [8, 2, 0]
    assert int_to_flipped_bits((1 << 1000) | (1 << 250)) == [1000, 250]


def test_from_error_instruction():
    assert DemError.from_error_instruction(
        stim.DemInstruction('error', [0.125], [stim.target_relative_detector_id(3)]),
    ) == DemError(
        p=0.125,
        det=0b1000,
        obs=0,
    )

    assert DemError.from_error_instruction(
        stim.DemInstruction(
            'error',
            [0.25],
            [
                stim.target_relative_detector_id(3),
                stim.target_relative_detector_id(4),
                stim.target_logical_observable_id(1),
                stim.target_separator(),
                stim.target_relative_detector_id(3),
                stim.target_relative_detector_id(5),
                stim.target_logical_observable_id(1),
                stim.target_logical_observable_id(2),
            ]),
    ) == DemError(
        p=0.25,
        det=0b110000,
        obs=0b100,
    )


def test_iter_pair_chunks():
    actual = []
    for chunk in iter_pair_chunks(np.array([
        0b000001,
        0b000110,
        0b111000,
    ])):
        actual.extend(chunk)
    assert len(actual) == len(set(actual))
    assert set(actual) == {
        0b000001 ^ 0b000110,
        0b000001 ^ 0b111000,
        0b000110 ^ 0b111000,
    }

    actual = []
    for chunk in iter_pair_chunks(np.array([
        0b0000000001,
        0b0000000110,
        0b0000111000,
        0b11110000000,
    ])):
        actual.extend(chunk)
    assert len(actual) == len(set(actual))
    assert set(actual) == {
        0b000001 ^ 0b000110,
        0b000001 ^ 0b111000,
        0b000110 ^ 0b111000,
        0b11110000000 ^ 0b000001,
        0b11110000000 ^ 0b110,
        0b11110000000 ^ 0b111000,
    }


def test_iter_triplet_chunks():
    actual = []
    for chunk in iter_triplet_chunks(np.array([
        0b000001,
        0b000110,
        0b111000,
    ])):
        actual.extend(chunk)
    assert len(actual) == len(set(actual))
    assert set(actual) == {
        0b000001 ^ 0b000110 ^ 0b111000,
    }

    actual = []
    for chunk in iter_triplet_chunks(np.array([
        0b00000000001,
        0b00000000110,
        0b00000111000,
        0b11110000000,
    ])):
        actual.extend(chunk)
    assert len(actual) == len(set(actual))
    assert set(actual) == {
        0b00000000110 ^ 0b00000111000 ^ 0b11110000000,
        0b00000000001 ^ 0b00000111000 ^ 0b11110000000,
        0b00000000001 ^ 0b00000000110 ^ 0b11110000000,
        0b00000000001 ^ 0b00000000110 ^ 0b00000111000,
    }

    t = 0
    for chunk in iter_triplet_chunks(np.array(range(10))):
        t += len(chunk)
    assert t == 10*9*8 // 6


def test_find_errors():
    dem = stim.DetectorErrorModel("""
        error(0.125) L0 D0
        error(0.375) D0 D1
        error(0.25) D0 D11
        error(0.125) D1 D2
        error(0.25) D11 D12
        error(0.125) D2 D3
        error(0.25) D12 D3
        error(0.125) D3
    """)
    ds = DemErrorSet.from_dem(dem)
    assert np.array_equal(ds.masks, [
        0b00000000000011,
        0b00000000000110,
        0b00000000001100,
        0b00000000010000,
        0b00000000011000,
        0b01000000000010,
        0b10000000010000,
        0b11000000000000,
    ])
    assert np.array_equal(ds.probs, [0.125, 0.375, 0.125, 0.125, 0.125, 0.25 , 0.25 , 0.25])
    assert ds.find_masks_reached_by_errors_up_to(max_distance=0) == {0}
    assert ds.find_masks_reached_by_errors_up_to(max_distance=1) == {
        0,
        *ds.masks,
    }
    stored2 = ds.find_masks_reached_by_errors_up_to(max_distance=2)
    assert stored2 == {
        0,
        *ds.masks,
        *[a ^ b for a in ds.masks for b in ds.masks],
    }
    stored3 = ds.find_masks_reached_by_errors_up_to(max_distance=3)
    assert stored3 == {
        0,
        *ds.masks,
        *[a ^ b for a in ds.masks for b in ds.masks],
        *[a ^ b ^ c for a in ds.masks for b in ds.masks for c in ds.masks],
    }

    assert ds.find_masks_reached_by_errors_up_to(max_distance=2, opposing_masks=stored2) == set()
    mids = ds.find_masks_reached_by_errors_up_to(max_distance=3, opposing_masks=stored2)
    assert mids == {
        (0b00000000000110 ^ 0b00000000000010) | 1,
        (0b00000000001100 ^ 0b00000000000010) | 1,
        (0b00000000011000 ^ 0b00000000000010) | 1,
        (0b00000000010000 ^ 0b00000000000010) | 1,
        (0b00000000001100 ^ 0b00000000000110) | 1,
        (0b00000000011000 ^ 0b00000000000110) | 1,
        (0b00000000010000 ^ 0b00000000000110) | 1,
        (0b00000000011000 ^ 0b00000000001100) | 1,
        (0b00000000010000 ^ 0b00000000001100) | 1,
        (0b00000000010000 ^ 0b00000000011000) | 1,
        (0b01000000000010 ^ 0b00000000000010) | 1,
        (0b11000000000000 ^ 0b00000000000010) | 1,
        (0b10000000010000 ^ 0b00000000000010) | 1,
        (0b11000000000000 ^ 0b01000000000010) | 1,
        (0b10000000010000 ^ 0b01000000000010) | 1,
        (0b00000000010000 ^ 0b01000000000010) | 1,
        (0b10000000010000 ^ 0b11000000000000) | 1,
        (0b00000000010000 ^ 0b11000000000000) | 1,
        (0b00000000010000 ^ 0b10000000010000) | 1,
    }

    errs = ds.find_errors_for_midpoint_masks(mids, max_distance=3)
    for k, combos in errs.items():
        for combo in combos:
            acc = 0
            for v in combo:
                acc ^= ds.masks[v]
            assert acc == k


def test_find_errors_multi_word_det_masks():
    dem = stim.DetectorErrorModel("""
        error(0.125) L0 D0
        error(0.375) D0 D1
        error(0.25) D0 D11
        error(0.125) D1 D2
        error(0.25) D11 D12
        error(0.125) D2 D3
        error(0.25) D12 D3
        error(0.125) D3
    """)
    r = DemErrorSet.from_dem(dem).find_logical_errors(max_distance=6)
    assert len(r) == 2

    dem = stim.DetectorErrorModel("""
        error(0.125) L0 D0
        error(0.375) D0 D1
        error(0.25) D0 D81
        error(0.125) D1 D2
        error(0.25) D81 D82
        error(0.125) D2 D3
        error(0.25) D82 D3
        error(0.125) D3
    """)
    r = DemErrorSet.from_dem(dem).find_logical_errors(max_distance=6)
    assert len(r) == 2


def test_chance_of_exactly_0():
    assert chance_of_exactly_0([]) == 1
    assert chance_of_exactly_0([0.25]) == 0.75
    assert chance_of_exactly_0([0.5]) == 0.5
    assert chance_of_exactly_0([1]) == 0
    assert chance_of_exactly_0([1, 1]) == 0
    assert chance_of_exactly_0([0.5, 0]) == 0.5
    assert chance_of_exactly_0([0, 0, 0, 0.25, 0]) == 0.75
    assert chance_of_exactly_0([0.5, 0, 0, 0.5, 0]) == 1 / 4
    assert chance_of_exactly_0([0.25, 0, 0, 0.25, 0]) == 9 / 16
    assert chance_of_exactly_0([0.25, 0.25]) == 9 / 16
    assert chance_of_exactly_0([0.25, 0.5]) == 3 / 8
    assert chance_of_exactly_0([0.5, 0.25]) == 3 / 8
    assert chance_of_exactly_0([0.25, 1]) == 0
    assert chance_of_exactly_0([0.5, 0.5, 0.5]) == 1 / 8
    assert chance_of_exactly_0([0.5, 0.5, 0.5, 0.5]) == 1 / 16
    assert chance_of_exactly_0([0.5, 0.5, 0.5, 0.5, 0.5]) == 1 / 32
    assert chance_of_exactly_0([0.25, 0.5, 0.5, 0.5, 0.5]) == 3 / 64


def test_chance_of_exactly_1():
    assert chance_of_exactly_1([]) == 0
    assert chance_of_exactly_1([0.25]) == 0.25
    assert chance_of_exactly_1([0.5]) == 0.5
    assert chance_of_exactly_1([1]) == 1
    assert chance_of_exactly_1([1, 1]) == 0
    assert chance_of_exactly_1([0.5, 0]) == 0.5
    assert chance_of_exactly_1([0, 0, 0, 0.25, 0]) == 0.25
    assert chance_of_exactly_1([0.5, 0, 0, 0.5, 0]) == 1 / 2
    assert chance_of_exactly_1([0.25, 0, 0, 0.25, 0]) == 6 / 16
    assert chance_of_exactly_1([0.25, 0.25]) == 6 / 16
    assert chance_of_exactly_1([0.25, 0.5]) == 4 / 8
    assert chance_of_exactly_1([0.5, 0.25]) == 4 / 8
    assert chance_of_exactly_1([0.25, 1]) == 3 / 4
    assert chance_of_exactly_1([0.5, 0.5, 0.5]) == 3 / 8
    assert chance_of_exactly_1([0.5, 0.5, 0.5, 0.5]) == 4 / 16
    assert chance_of_exactly_1([0.5, 0.5, 0.5, 0.5, 0.5]) == 5 / 32
    assert chance_of_exactly_1([0.25, 0.5, 0.5, 0.5, 0.5]) == 13 / 64
