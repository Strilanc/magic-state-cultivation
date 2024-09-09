import sinter

import gen
import cultiv
from ._desaturation_sampler import DesaturationSampler, _DemError


def test_from_dem_end_to_end_d3():
    c = cultiv.make_end2end_cultivation_circuit(dcolor=3, dsurface=7, basis='Y', r_growing=2, r_end=1, inject_style='unitary')
    c = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(c)
    dem = c.detector_error_model()
    dec = DesaturationSampler().compiled_sampler_for_task(sinter.Task(circuit=c, detector_error_model=dem))

    assert dec.decode_det_set(set()) == (False, 67)
    assert dec.decode_det_set({0}) == (False, 0)
    det_coords = dem.get_detector_coordinates()
    assert len(set(tuple(v) for v in det_coords.values())) == len(det_coords)

    errors = []
    for inst in dem.flattened():
        if inst.type == 'error':
            errors.append(_DemError.from_error_instruction(inst))
    pt = 0
    for err in sorted(errors, key=lambda e: len(e.det_set)):
        pred, gap = dec.decode_det_set(set(err.det_set))
        if gap == 0:
            pt += err.p
            continue
        assert gap > 0, (err, pred, gap)
        assert pred == err.obs_mask, (err, pred, gap)

    stats = dec.sample(shots=1024)
    assert stats.errors / (stats.shots - stats.discards) < 0.1


def test_from_dem_end_to_end_d5():
    c = cultiv.make_end2end_cultivation_circuit(dcolor=5, dsurface=11, basis='Y', r_growing=4, r_end=1, inject_style='unitary')
    c = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(c)
    dem = c.detector_error_model()
    dec = DesaturationSampler().compiled_sampler_for_task(sinter.Task(circuit=c, detector_error_model=dem))

    assert dec.decode_det_set(set()) == (False, 110)
    det_coords = dem.get_detector_coordinates()
    assert len(set(tuple(v) for v in det_coords.values())) == len(det_coords)

    for inst in dem.flattened():
        if inst.type == 'error':
            err = _DemError.from_error_instruction(inst)
            pred, gap = dec.decode_det_set(set(err.det_set))
            if gap != 0:
                assert pred == err.obs_mask, (err, pred, gap)
