import collections
import math
import time

import numpy as np
import pymatching
import sinter
import stim

from latte.dem_util import dem_with_compressed_detectors, \
    dem_with_replaced_targets


class PymatchingGapSampler(sinter.Sampler):
    """Computes gaps using pymatching, in addition to decoding the shots.

    Requires the observable to exist purely on boundary edges.
    """
    def __init__(self, decoder: sinter.Decoder | None = None):
        self.decoder = decoder

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledPymatchingGapSampler(task, self.decoder)


class CompiledPymatchingGapSampler(sinter.CompiledSampler):
    def __init__(self, task: sinter.Task, decoder: sinter.Decoder | None):
        circuit = task.circuit
        def is_postselected(coords: list[float]) -> bool:
            if len(coords) > 4 and (coords[4] == -99 or coords[4] == -9):
                return True
            if coords[-1] == 999:
                return True
            return False

        self.num_obs = circuit.num_observables
        num_dets = circuit.num_detectors
        if self.num_obs > 8:
            raise NotImplementedError(f"{self.num_obs} > 8")

        dem = circuit.detector_error_model(
            decompose_errors=True,
            approximate_disjoint_errors=True,
            ignore_decomposition_failures=True,
        )
        dem = dem_with_compressed_detectors(
            dem=dem,
            compressed_detector_predicate=is_postselected,
            max_compressed_errors=2,
            error_size_cutoff=3,
            detection_event_cutoff=4,
        )

        # Byte-align the additional detectors, for convenience.
        aligned_circuit = circuit.copy()
        while num_dets & 7:
            num_dets += 1
            aligned_circuit.append("DETECTOR")
        for k in range(self.num_obs):
            aligned_circuit.append("DETECTOR")

        dem_obs2det = dem_with_replaced_targets(dem, {
            stim.target_logical_observable_id(k): stim.target_relative_detector_id(num_dets + k)
            for k in range(self.num_obs)
        })
        dem.append('detector', (), [stim.target_relative_detector_id(aligned_circuit.num_detectors - 1)])

        self.postselection_mask = np.zeros(shape=num_dets // 8 + 1, dtype=np.uint8)
        vv1 = []
        vv2 = []
        self.d2c = circuit.get_detector_coordinates()
        for det, coords in self.d2c.items():
            if is_postselected(coords):
                vv1.append(det)
                vv2.append(coords)
                self.postselection_mask[det >> 3] |= 1 << (det & 7)

        self.controlled_det_byte = num_dets >> 3
        if decoder is not None:
            self.compiled_decoder = decoder.compile_decoder_for_dem(dem=dem)
        else:
            self.compiled_decoder = None
        self.gap_matcher = pymatching.Matching.from_detector_error_model(dem_obs2det)
        self.stim_sampler = aligned_circuit.compile_detector_sampler()

        edge = next(iter(self.gap_matcher.to_networkx().edges.values()))
        edge_w = edge['weight']
        edge_p = edge['error_probability']
        self.decibels_per_w = -math.log10(edge_p / (1 - edge_p)) * 10 / edge_w

    def sample(self, max_shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets, actual_obs = self.stim_sampler.sample(
            shots=max_shots,
            bit_packed=True,
            separate_observables=True,
        )
        num_shots = dets.shape[0]
        discard_mask = np.any(dets & self.postselection_mask, axis=1)
        num_discards = np.count_nonzero(discard_mask)
        dets = dets[~discard_mask]
        actual_obs = actual_obs[~discard_mask]
        num_kept_shots = dets.shape[0]

        predictions: np.ndarray | None = None
        if self.compiled_decoder is not None:
            predictions = self.compiled_decoder.decode_shots_bit_packed(
                bit_packed_detection_event_data=dets
            )[:, 0]

        weights = np.zeros(shape=(num_kept_shots, 1 << self.num_obs), dtype=np.float64)
        for mask in range(1 << self.num_obs):
            dets[:, self.controlled_det_byte] = mask
            weight = _decode_weight_with_pymatching_with_better_error_message(
                self.gap_matcher,
                dets,
                self.d2c,
            )
            weights[:, mask] = weight

        if self.compiled_decoder is None:
            predictions = np.array(np.argmin(weights, axis=1), dtype=np.uint8)
        assert predictions is not None
        errors = predictions != actual_obs[:, 0]
        sorted_weights = np.sort(weights, axis=1)
        gaps = (sorted_weights[:, 1] - sorted_weights[:, 0])
        num_errors = np.count_nonzero(errors)

        # Classify all shots by their error + gap.
        custom_counts = collections.Counter()
        gaps_db = np.round(gaps * self.decibels_per_w).astype(dtype=np.int64)
        for k in range(num_kept_shots):
            g = gaps_db[k]
            e = 'CE'[errors[k]]
            key = f'{e}{g}'
            custom_counts[key] += 1
        t1 = time.monotonic()

        return sinter.AnonTaskStats(
            shots=num_shots,
            errors=num_errors,
            discards=num_discards,
            seconds=t1 - t0,
            custom_counts=custom_counts,
        )


def _decode_weight_with_pymatching_with_better_error_message(
        matcher: pymatching.Matching,
        dets: np.ndarray,
        d2c: dict[int, list[float]],
) -> np.ndarray:
    try:
        _, weight = matcher.decode_batch(
            dets,
            return_weights=True,
            bit_packed_shots=True,
            bit_packed_predictions=True,
        )
        return weight
    except ValueError:
        pass

    for k in range(dets.shape[0]):
        try:
            matcher.decode_batch(
                dets[k:k + 1],
                return_weights=True,
                bit_packed_shots=True,
                bit_packed_predictions=True,
            )
        except ValueError as ex:
            bad_dets = np.flatnonzero(np.unpackbits(dets[k], bitorder='little'))
            lines = [
                "pymatching failed to decode a shot:",
                "    shot" + "".join(f' D{d}' for d in bad_dets),
                "Failing detector coords:",
            ]
            for d in bad_dets:
                lines.append(f'D{d} {tuple(d2c.get(d, ()))!r}')
            raise ValueError('\n'.join(lines)) from ex

    raise ValueError("pymatching failed to decode a batch of shots, but then succeeded on each shot individually?")
