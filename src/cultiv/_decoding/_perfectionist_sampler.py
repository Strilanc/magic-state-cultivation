import time

import numpy as np
import sinter


class PerfectionistSampler(sinter.Sampler):
    """Predicts obs aren't flipped. Discards shots with any detection events."""
    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledPerfectionistSampler(task)


class CompiledPerfectionistSampler(sinter.CompiledSampler):
    def __init__(self, task: sinter.Task):
        self.stim_sampler = task.circuit.compile_detector_sampler()

    def sample(self, max_shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets, obs = self.stim_sampler.sample(
            shots=max_shots,
            bit_packed=True,
            separate_observables=True,
        )
        num_shots = dets.shape[0]
        discards = np.any(dets, axis=1)
        errors = np.any(obs, axis=1)
        num_discards = np.count_nonzero(discards)
        num_errors = np.count_nonzero(errors & ~discards)
        t1 = time.monotonic()

        return sinter.AnonTaskStats(
            shots=num_shots,
            errors=num_errors,
            discards=num_discards,
            seconds=t1 - t0,
        )
