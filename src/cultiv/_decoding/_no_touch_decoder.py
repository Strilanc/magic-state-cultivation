import collections
import dataclasses
import pathlib

import numpy as np
import sinter
import stim

from cultiv._error_set import DemError


class NoTouchDecoder(sinter.Decoder):
    """Discards any shots where an error can't be uniquely decoded from nearby detection events.

    Uses local lookup tables to match detection events to the unique local error, to make the
    prediction.
    """

    def __init__(self, discard_on_fail: bool):
        self.discard_on_fail = discard_on_fail

    def decode_via_files(self, *, num_shots: int, num_dets: int, num_obs: int,
                         dem_path: pathlib.Path, dets_b8_in_path: pathlib.Path,
                         obs_predictions_b8_out_path: pathlib.Path,
                         tmp_dir: pathlib.Path) -> None:
        raise NotImplementedError()

    def compile_decoder_for_dem(
        self,
        *,
        dem: stim.DetectorErrorModel,
    ) -> sinter.CompiledDecoder:
        return CompiledNoTouchDecoder(dem, discard_on_fail=self.discard_on_fail)


@dataclasses.dataclass(frozen=True)
class _LocalLookup:
    neighborhood: frozenset[int]
    local_symptoms_to_error_index: dict[frozenset[int], int]


@dataclasses.dataclass(frozen=True)
class _GlobalLookup:
    obs_masks: np.ndarray
    det_to_local_lookup: dict[int, _LocalLookup]

    @staticmethod
    def from_dem(dem: stim.DetectorErrorModel) -> '_GlobalLookup':
        if dem.num_observables > 8:
            raise NotImplementedError(f'{dem.num_observables=} > 8')

        error_list = []
        for instruction in dem.flattened():
            if instruction.type == 'error':
                error_list.append(DemError.from_error_instruction(instruction))

        det_to_local_err_indices = collections.defaultdict(list)
        for err_index in range(len(error_list)):
            for d in error_list[err_index].det_list():
                det_to_local_err_indices[d].append(err_index)

        det_to_local_lookup = {}
        for det, local_err_indices in det_to_local_err_indices.items():
            det_to_local_lookup[det] = _LocalLookup(
                neighborhood=frozenset(
                    other_det
                    for local_error_index in local_err_indices
                    for other_det in error_list[local_error_index].det_list()
                ),
                local_symptoms_to_error_index={
                    frozenset(error_list[local_err_index].det_list()): local_err_index
                    for local_err_index in local_err_indices
                },
            )

        obs_masks = np.array([err.obs for err in error_list], dtype=np.uint8)

        return _GlobalLookup(det_to_local_lookup=det_to_local_lookup, obs_masks=obs_masks)


class CompiledNoTouchDecoder(sinter.CompiledDecoder):
    def __init__(self, dem: stim.DetectorErrorModel, discard_on_fail: bool):
        self.lookup = _GlobalLookup.from_dem(dem)
        self.discard_on_fail = discard_on_fail

    def decode_det_set(self, detection_events: frozenset[int]) -> int | None:
        forced = {}
        for det in detection_events:
            lookup = self.lookup.det_to_local_lookup[det]
            local_key = lookup.neighborhood & detection_events
            local_err = lookup.local_symptoms_to_error_index.get(local_key)
            if local_err is None:
                if not self.discard_on_fail:
                    continue
                return None
            for n in local_key:
                if forced.setdefault(n, local_err) != local_err:
                    if not self.discard_on_fail:
                        continue
                    return None
        obs = 0
        for err_index in set(forced.values()):
            obs ^= self.lookup.obs_masks[err_index]
        return obs

    def decode_shots_bit_packed(
            self,
            *,
            bit_packed_detection_event_data: np.ndarray,
    ) -> np.ndarray:
        dets = bit_packed_detection_event_data
        result = np.zeros(shape=(dets.shape[0], 2), dtype=np.uint8)
        for k in range(len(dets)):
            prediction = self.decode_det_set(frozenset(np.flatnonzero(np.unpackbits(dets[k], bitorder='little'))))
            if prediction is None:
                result[k, 1] = 1
            else:
                result[k, 0] = prediction
        return result
