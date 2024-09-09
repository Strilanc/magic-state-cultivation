import math
import time

import numpy as np
import sinter
import stim

from cultiv._error_set import DemErrorSet


class HighlanderSampler(sinter.Sampler):
    """Lookup table decoder that allows at most one error, else discards."""
    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledHighlanderSampler(task)


def dem_to_single_error_lookup_table(dem: stim.DetectorErrorModel) -> dict[tuple[int, ...], np.ndarray]:
    num_obs = dem.num_observables
    result = {
        (): np.zeros(dtype=np.uint8, shape=math.ceil(num_obs / 8)),
    }
    for err in DemErrorSet.from_dem(dem).errors:
        prediction = np.zeros(dtype=np.uint8, shape=math.ceil(num_obs / 8))
        for obs in err.obs_list():
            prediction[obs // 8] ^= 1 << (obs % 8)
        result[tuple(sorted(err.det_list()))] = prediction
    return result


class CompiledHighlanderSampler(sinter.CompiledSampler):
    def __init__(self, task: sinter.Task):
        self.stim_sampler = task.circuit.compile_detector_sampler()
        self.lookup_table = dem_to_single_error_lookup_table(task.detector_error_model)

    def sample(self, max_shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets, obs = self.stim_sampler.sample(
            shots=max_shots,
            bit_packed=True,
            separate_observables=True,
        )
        num_shots = dets.shape[0]

        num_discards = 0
        num_errors = 0
        for shot in range(num_shots):
            key = tuple(np.flatnonzero(np.unpackbits(dets[shot], bitorder='little')))
            prediction = self.lookup_table.get(key)
            if prediction is None:
                num_discards += 1
            elif not np.array_equal(prediction, obs[shot]):
                num_errors += 1
        t1 = time.monotonic()

        return sinter.AnonTaskStats(
            shots=num_shots,
            errors=num_errors,
            discards=num_discards,
            seconds=t1 - t0,
        )
