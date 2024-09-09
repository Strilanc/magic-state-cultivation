from typing import Any

import stim

from latte.dem_util import dem_with_replaced_targets, \
    dem_with_compressed_detectors, bernoulli_combo, Symptom


def test_dem_with_replaced_targets():
    dem = stim.DetectorErrorModel("""
        error(0.25) D2 D3 L0
        logical_observable L0
        detector(1, 2) D2
    """)
    dem2 = dem_with_replaced_targets(
        dem,
        {stim.target_logical_observable_id(0): stim.target_relative_detector_id(5)},
    )
    assert dem2 == stim.DetectorErrorModel("""
        error(0.25) D2 D3 D5
        detector D5
        detector(1, 2) D2
    """)


def approx_equal(v1: Any, v2: Any, *, atol: float):
    if isinstance(v1, dict) and isinstance(v2, dict):
        if v1.keys() != v2.keys():
            return False
        for k, s1 in v1.items():
            s2 = v2[k]
            if not approx_equal(s1, s2, atol=atol):
                return False
        return True
    elif (isinstance(v1, tuple) and isinstance(v2, tuple)) or (isinstance(v1, list) and isinstance(v2, list)):
        if len(v1) != len(v2):
            return False
        for s1, s2 in zip(v1, v2):
            if not approx_equal(s1, s2, atol=atol):
                return False
        return True
    elif isinstance(v1, float) and isinstance(v2, float):
        return abs(v2 - v1) <= atol
    else:
        return v1 == v2


def test_bernoulli_combo():
    # assert bernoulli_combo(
    #     errors={
    #         Symptom(obs_mask=1 << 0, dets=frozenset([0])): 0.01,
    #         Symptom(obs_mask=1 << 1, dets=frozenset([0, 1])): 0.01,
    #         Symptom(obs_mask=1 << 2, dets=frozenset([1, 2])): 0.01,
    #         Symptom(obs_mask=1 << 3, dets=frozenset([2, 3])): 0.01,
    #         Symptom(obs_mask=1 << 4, dets=frozenset([3])): 0.01,
    #     },
    #     compressed_dets=frozenset([]),
    #     max_errors=20,
    #     error_size_cutoff=3,
    #     detection_event_cutoff=2,
    # ) == {}
    #
    # assert bernoulli_combo(
    #     errors={
    #         Symptom(obs_mask=1 << 0, dets=frozenset([0])): 0.01,
    #         Symptom(obs_mask=1 << 1, dets=frozenset([0, 1])): 0.01,
    #         Symptom(obs_mask=1 << 2, dets=frozenset([1, 2])): 0.01,
    #         Symptom(obs_mask=1 << 3, dets=frozenset([2, 3])): 0.01,
    #         Symptom(obs_mask=1 << 4, dets=frozenset([3])): 0.01,
    #     },
    #     compressed_dets=frozenset([0]),
    #     max_errors=1,
    #     error_size_cutoff=3,
    #     detection_event_cutoff=2,
    # ) == {}

    v2 = bernoulli_combo(
        errors={
            Symptom(obs_mask=1 << 0, dets=frozenset([0])): 0.01,
            Symptom(obs_mask=1 << 1, dets=frozenset([0, 1])): 0.01,
            Symptom(obs_mask=1 << 2, dets=frozenset([1, 2])): 0.01,
            Symptom(obs_mask=1 << 3, dets=frozenset([2, 3])): 0.01,
            Symptom(obs_mask=1 << 4, dets=frozenset([3])): 0.01,
        },
        compressed_dets=frozenset([0]),
        max_errors=2,
        error_size_cutoff=3,
        detection_event_cutoff=2,
    )
    assert approx_equal(v2, {1: (3, 1e-4)}, atol=1e-5)

    v3 = bernoulli_combo(
        errors={
            Symptom(obs_mask=1 << 0, dets=frozenset([0])): 0.01,
            Symptom(obs_mask=1 << 1, dets=frozenset([0, 1])): 0.01,
            Symptom(obs_mask=1 << 2, dets=frozenset([1, 2])): 0.01,
            Symptom(obs_mask=1 << 3, dets=frozenset([2, 3])): 0.01,
            Symptom(obs_mask=1 << 4, dets=frozenset([3])): 0.01,
        },
        compressed_dets=frozenset([0, 1]),
        max_errors=3,
        error_size_cutoff=3,
        detection_event_cutoff=2,
    )
    assert approx_equal(v3, {2: (7, 1e-6)}, atol=5e-7)


def test_dem_with_compressed_detectors():
    dem = stim.DetectorErrorModel("""
        error(0.01) D0 L0
        error(0.01) D0 D1
        error(0.01) D1 D2
        error(0.01) D2 D3
        error(0.01) D3 D4
        error(0.01) D4
        detector(0) D0
        detector(1) D1
        detector(2) D2
        detector(3) D3
        detector(4) D4
    """)

    dem2 = dem_with_compressed_detectors(
        dem,
        lambda coords: len(coords) > 0 and coords[0] < 1,
        error_size_cutoff=3,
        detection_event_cutoff=4,
        max_compressed_errors=2,
    )
    assert dem2.approx_equals(stim.DetectorErrorModel("""
        error(0.01) D1 D2
        error(0.01) D2 D3
        error(0.01) D3 D4
        error(0.01) D4
        detector(0) D0
        detector(1) D1
        detector(2) D2
        detector(3) D3
        detector(4) D4
        error(0.0001) D1 L0
        error(0.5) D0
    """), atol=0.00001)
