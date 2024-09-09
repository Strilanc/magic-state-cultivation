import collections
import dataclasses
import time
from typing import Iterable

import numpy as np
import sinter
import stim

import gen


class ChromobiusGapSampler(sinter.Sampler):
    """Samples from chromobius while collecting gaps based on the mobius matching weights.

    Attempts to attach a detector to represent the logical observable, with its color and
    basis picked on the most common adjacent colors to the observable. Then compares the
    weight from exciting and not exciting that detector.
    """
    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledChromobiusGapSampler(task)


@dataclasses.dataclass(frozen=True)
class _DemError:
    p: float
    det_set: frozenset[int]
    has_obs: bool

    def __mul__(self, other):
        return _DemError(
            p=self.p * other.p,
            det_set=self.det_set ^ other.det_set,
            has_obs=self.has_obs ^ other.has_obs,
        )

    @staticmethod
    def from_error_instruction(instruction: stim.DemInstruction) -> '_DemError':
        assert instruction.type == 'error'
        p = instruction.args_copy()[0]
        det_list = []
        has_obs = False
        for target in instruction.targets_copy():
            if target.is_logical_observable_id():
                assert target.val == 0
                has_obs ^= True
            elif target.is_relative_detector_id():
                det_list.append(target.val)
            elif target.is_separator():
                pass
            else:
                raise NotImplementedError(f'{instruction}')
        return _DemError(p=p, det_set=frozenset(gen.xor_sorted(det_list)), has_obs=has_obs)

    def to_instruction(self, obs_det: int) -> stim.DemInstruction:
        targets = []
        for d in sorted(self.det_set):
            targets.append(stim.target_relative_detector_id(d))
        if self.has_obs:
            if obs_det != -1:
                targets.append(stim.target_relative_detector_id(obs_det))
            targets.append(stim.target_logical_observable_id(0))
        return stim.DemInstruction('error', [self.p], targets)


class CompiledChromobiusGapSampler(sinter.CompiledSampler):
    def __init__(self, task: sinter.Task):
        if task.detector_error_model.num_observables != 1:
            raise NotImplementedError(f'{task.detector_error_model.num_observables=} != 1')
        self.main_dem = task.detector_error_model.flattened()
        self.obs_det = self.main_dem.num_detectors
        coords = self.main_dem.get_detector_coordinates()
        adj_pairs = collections.Counter()

        self.gap_dem_base = stim.DetectorErrorModel()
        for inst in self.main_dem:
            if inst.type == 'error':
                err = _DemError.from_error_instruction(inst)
                if err.has_obs and len(err.det_set) == 2:
                    d1, d2 = err.det_set
                    c1 = coords[d1][3]
                    c2 = coords[d2][3]
                    if (c1 // 3) == (c2 // 3) and (c1 % 3 != c2 % 3):
                        adj_pairs[frozenset([c1, c2])] += 1
                self.gap_dem_base.append(err.to_instruction(self.obs_det))
            else:
                self.gap_dem_base.append(inst)
        max_key = max(adj_pairs.keys(), key=lambda key: adj_pairs[key])
        c1, c2 = max_key
        if (c1 // 3) != (c2 // 3):
            raise NotImplementedError(f'{c1=}, {c2=}')
        obs_color_basis = (c1 // 3) * 3 + (3 - (c1 % 3) - (c2 % 3))
        self.gap_dem_base.append('detector', [-9, -9, -9, obs_color_basis], [stim.target_relative_detector_id(self.obs_det)])
        self.main_dem.append('detector', [-9, -9, -9, obs_color_basis], [stim.target_relative_detector_id(self.obs_det)])

        import chromobius
        self.decoder = chromobius.compile_decoder_for_dem(self.main_dem)
        self.gap_decoder = chromobius.compile_decoder_for_dem(self.gap_dem_base)
        self.gap_decoders: list[tuple[float, chromobius.CompiledDecoder]] = []

        circuit = task.circuit.copy()
        circuit.append("DETECTOR")
        self.stim_sampler = circuit.compile_detector_sampler()

    def decode_dets(self, detection_events: Iterable[int]) -> tuple[bool, int]:
        det_data = np.zeros(shape=self.obs_det + 1, dtype=np.bool_)
        for k in detection_events:
            det_data[k] ^= 1
        return self.decode_shot(np.packbits(det_data, bitorder='little'))

    def decode_shot(self, shot: np.ndarray) -> tuple[bool, int]:
        assert len(shot.shape) == 1
        assert shot.shape[0] == (self.obs_det + 8) // 8
        _, weight0 = self.gap_decoder.predict_weighted_obs_flips_from_dets_bit_packed(shot)
        shot[-1] ^= np.uint8(1 << (self.obs_det % 8))
        _, weight1 = self.gap_decoder.predict_weighted_obs_flips_from_dets_bit_packed(shot)
        shot[-1] ^= np.uint8(1 << (self.obs_det % 8))
        prediction = self.decoder.predict_obs_flips_from_dets_bit_packed(shot)
        return bool(prediction), round(abs(weight1 - weight0))

    def sample(self, shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets, actual_obs = self.stim_sampler.sample(shots, separate_observables=True, bit_packed=True)
        num_errors = 0
        gap_counts = collections.Counter()
        for k in range(shots):
            try:
                pred, gap = self.decode_shot(dets[k])
            except ValueError:
                pred, gap = False, 0
            if pred != np.any(actual_obs[k]):
                num_errors += 1
                gap_counts[f'E{gap}'] += 1
            else:
                gap_counts[f'C{gap}'] += 1
        t1 = time.monotonic()

        return sinter.AnonTaskStats(
            shots=shots,
            errors=num_errors,
            seconds=t1 - t0,
            custom_counts=gap_counts,
        )
