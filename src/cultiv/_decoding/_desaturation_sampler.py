import collections
import dataclasses
import heapq
import math
import time
from typing import Literal, cast, Any, AbstractSet

import numpy as np
import pymatching
import sinter
import stim

import gen
from cultiv._error_set import int_to_flipped_bits


class DesaturationSampler(sinter.Sampler):
    def compiled_sampler_for_task(self, task: sinter.Task) -> 'CompiledDesaturationSampler':
        return CompiledDesaturationSampler.from_task(task)


@dataclasses.dataclass(frozen=True)
class _DemError:
    p: float
    det_set: frozenset[int]
    obs_mask: int

    @staticmethod
    def from_error_instruction(instruction: stim.DemInstruction) -> '_DemError':
        p = instruction.args_copy()[0]
        det_list = []
        obs_mask = 0
        for target in instruction.targets_copy():
            if target.is_logical_observable_id():
                obs_mask ^= 1 << target.val
            elif target.is_relative_detector_id():
                det_list.append(target.val)
            elif target.is_separator():
                pass
            else:
                raise NotImplementedError(f'{instruction}')
        return _DemError(p=p, det_set=frozenset(gen.xor_sorted(det_list)), obs_mask=obs_mask)

    def to_instruction(self) -> stim.DemInstruction:
        targets = []
        for d in self.det_set:
            targets.append(stim.target_relative_detector_id(d))
        for d in int_to_flipped_bits(self.obs_mask)[::-1]:
            targets.append(stim.target_logical_observable_id(d))
        return stim.DemInstruction('error', [self.p], targets)

    @staticmethod
    def to_separated_instruction(parts: list['_DemError']) -> stim.DemInstruction:
        assert len(parts) >= 1
        assert len(set(p.p for p in parts)) == 1
        targets = []
        for k in range(len(parts)):
            if k:
                targets.append(stim.target_separator())
            for d in parts[k].det_set:
                targets.append(stim.target_relative_detector_id(d))
            for d in int_to_flipped_bits(parts[k].obs_mask)[::-1]:
                targets.append(stim.target_logical_observable_id(d))
        return stim.DemInstruction('error', [parts[0].p], targets)


@dataclasses.dataclass
class CompiledDesaturationSampler(sinter.CompiledSampler):
    def __init__(
        self,
        task: sinter.Task,
        gap_dem: stim.DetectorErrorModel,
        postselected_detectors: frozenset[int],
        gap_circuit: stim.Circuit,
    ):
        self.task = task
        self.gap_dem = gap_dem
        self.postselected_detectors = postselected_detectors
        self.gap_circuit = gap_circuit

        self.num_dets = self.gap_circuit.num_detectors
        self.num_det_bytes = -(-self.num_dets // 8)
        self._discard_mask = np.packbits(np.array([k in self.postselected_detectors for k in range(self.num_dets)], dtype=np.bool_), bitorder='little')
        self.gap_circuit_sampler = self.gap_circuit.compile_detector_sampler()
        self.gap_decoder = pymatching.Matching.from_detector_error_model(self.gap_dem)
        self._obs_det_byte = 1 << ((self.num_dets - 1) % 8)

        edge = next(iter(self.gap_decoder.to_networkx().edges.values()))
        edge_w = edge['weight']
        edge_p = edge['error_probability']
        self.decibels_per_w = -math.log10(edge_p / (1 - edge_p)) * 10 / edge_w

    @staticmethod
    def from_task(task: sinter.Task) -> 'CompiledDesaturationSampler':
        dem = task.detector_error_model.flattened()
        num_dets = dem.num_detectors
        gap_circuit = task.circuit.copy()

        # Parse color and basis annotations out of the dem.
        det_coords = dem.get_detector_coordinates()
        det_bases: list[Literal['X', 'Z', '!']] = []
        det_colors: list[Literal['r', 'g', 'b', '_']] = []
        postselected_detectors_hidden_from_matcher = set()
        postselected_detectors_visible_to_matcher = set()

        for d in range(num_dets):
            coords = det_coords[d]
            if len(coords) <= 4 or coords[4] == -9:
                postselected_detectors_hidden_from_matcher.add(d)
                det_bases.append('!')
                det_colors.append('_')
                continue

            coord_annotation = int(coords[4])
            basis = cast(Any, 'XXXZZZXZ'[coord_annotation])
            color = cast(Any, 'rgbrgb__'[coord_annotation])
            det_bases.append(basis)
            det_colors.append(color)
            if (color == 'r' and basis == 'X') or (color == 'g' and basis == 'Z'):
                # Postselect the color code detectors that can be ablated, leaving a matchable code.
                postselected_detectors_visible_to_matcher.add(d)

        # Parse errors out of the dem.
        errors = []
        for inst in dem:
            if inst.type != 'error':
                continue
            errors.append(_DemError.from_error_instruction(inst))

        # Classify each single-basis error's obs flip, to help with decomposition.
        dets_to_obs = {}
        for err in errors:
            bases = {
                det_bases[d]
                for d in err.det_set
            }
            if len(bases) == 1 and len(err.det_set) == 2 and err.obs_mask:
                a, b = err.det_set
                postselected_detectors_hidden_from_matcher.add(a)
                postselected_detectors_hidden_from_matcher.add(b)
            if len(bases) == 1:
                dets_to_obs[err.det_set] = err.obs_mask

        # Find single-basis RGB triplet errors that need to be simplified for the matcher.
        virtual_pair_nodes = set()
        for err in errors:
            colors = {det_colors[d] for d in err.det_set}
            bases = {det_bases[d] for d in err.det_set}
            if len(err.det_set) == 3:
                if len(bases) == 1 and colors == {'r', 'g', 'b'}:
                    a, b, c = err.det_set
                    virtual_pair_nodes.add(frozenset([a, b]))
                    virtual_pair_nodes.add(frozenset([a, c]))
                    virtual_pair_nodes.add(frozenset([b, c]))
            elif len(err.det_set) == 2 and len(bases) == 1 and (colors == {'r', 'g'} or colors == {'r', 'b'} or colors == {'b', 'g'}):
                a, b = err.det_set
                virtual_pair_nodes.add(frozenset([a, b]))

        pair2virtual = {}
        for pair in sorted(virtual_pair_nodes, key=lambda e: tuple(sorted(e))):
            k = len(pair2virtual) + num_dets
            pair2virtual[pair] = k
            a, b = pair
            if a != -1 and b != -1:
                det_coords[k] = [(x + y) / 2 for x, y in list(zip(det_coords[a], det_coords[b]))[:3]]
            else:
                c = a if a != -1 else b
                det_coords[k] = det_coords[c][:3]
                det_coords[k][0] += 0.25
                det_coords[k][1] += 0.25
                det_coords[k][2] += 0.25
            gap_circuit.append('DETECTOR', [], det_coords[k])

        matchable_dem = stim.DetectorErrorModel()
        for k in range(num_dets + len(pair2virtual)):
            matchable_dem.append('detector', det_coords[k], [stim.target_relative_detector_id(k)])
        for err in errors:
            colors = collections.Counter(det_colors[d] for d in err.det_set)
            bases = collections.Counter(det_bases[d] for d in err.det_set)
            if len(err.det_set) == 2 and err.det_set in virtual_pair_nodes:
                # A boundary error at the side of the color code region.

                # Add as a normal error, and also as a virtual-pair-node boundary error.
                virtual_err = _DemError(
                    p=err.p,
                    det_set=frozenset([pair2virtual[err.det_set]]),
                    obs_mask=err.obs_mask,
                )
                matchable_dem.append(err.to_instruction())
                matchable_dem.append(virtual_err.to_instruction())

            elif len(err.det_set) == 3 and len(bases) == 1 and colors == collections.Counter('rgb'):
                # This is a bulk error within the color code region.

                # Split into three node-to-virtual-node-pair errors.
                assert err.obs_mask == 0
                for solo in err.det_set:
                    virtual_err = _DemError(
                        p=err.p,
                        obs_mask=err.obs_mask,
                        det_set=frozenset([solo, pair2virtual[err.det_set ^ frozenset([solo])]]),
                    )
                    matchable_dem.append(virtual_err.to_instruction())

            elif len(err.det_set) <= 2 and (bases.keys() == {'X'} or bases.keys() == {'Z'}):
                # This is a simple matchable error.
                matchable_dem.append(err.to_instruction())

            elif bases['X'] <= 2 and bases['Z'] <= 2:
                # This is a decomposable matchable error.

                # Decompose into X part and Z part.
                xs = frozenset([d for d in err.det_set if det_bases[d] == 'X'])
                zs = frozenset([d for d in err.det_set if det_bases[d] == 'Z'])
                if xs not in dets_to_obs or zs not in dets_to_obs:
                    # Don't know what the individual parts actually do.
                    continue

                obs_x = dets_to_obs[xs]
                obs_z = dets_to_obs[zs]
                if obs_x ^ obs_z != err.obs_mask:
                    # Decomposition failed. Could be due to a distance 3 logical error.
                    continue

                x_part = _DemError(p=err.p, det_set=xs, obs_mask=obs_x)
                z_part = _DemError(p=err.p, det_set=zs, obs_mask=obs_z)
                matchable_dem.append(_DemError.to_separated_instruction([
                    x_part,
                    z_part,
                ]))
            else:
                # Too complicated. Don't tell the matcher about it.
                pass

        clipped_dem = clipped_matchable_dem(matchable_dem, postselected_detectors_hidden_from_matcher)
        clipped_dem_with_det_for_obs = _dem_with_obs_detector(clipped_dem)
        gap_circuit.append("DETECTOR", [], [-9, -9, -9])  # gap observable detector
        assert gap_circuit.num_detectors == clipped_dem_with_det_for_obs.num_detectors

        clipped_dem.append(stim.DemInstruction('detector', [-9, -9, -9], [stim.target_relative_detector_id(clipped_dem_with_det_for_obs.num_detectors - 1)]))
        return CompiledDesaturationSampler(
            task=task,
            gap_dem=clipped_dem_with_det_for_obs,
            postselected_detectors=frozenset(postselected_detectors_hidden_from_matcher | postselected_detectors_visible_to_matcher),
            gap_circuit=gap_circuit,
        )

    def sample(self, shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()
        dets, actual_obs = self.gap_circuit_sampler.sample(shots, separate_observables=True, bit_packed=True)

        keep_mask = ~np.any(dets & self._discard_mask, axis=1)
        dets = dets[keep_mask]
        actual_obs = actual_obs[keep_mask]
        assert actual_obs.shape[1] == 1
        actual_obs = actual_obs[:, 0]
        predictions, gaps = self._decode_batch_overwrite_last_byte(bit_packed_dets=dets)
        errors = predictions ^ actual_obs
        counter = collections.Counter()
        for gap, err in zip(gaps, errors):
            counter[f'E{round(gap)}' if err else f'C{round(gap)}'] += 1
        t1 = time.monotonic()

        return sinter.AnonTaskStats(
            shots=shots,
            errors=np.count_nonzero(errors),
            discards=shots - np.count_nonzero(keep_mask),
            seconds=t1 - t0,
            custom_counts=counter,
        )

    def _decode_batch_overwrite_last_byte(self, bit_packed_dets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        bit_packed_dets[:, -1] |= self._obs_det_byte
        _, on_weights = self.gap_decoder.decode_batch(bit_packed_dets, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True)
        bit_packed_dets[:, -1] ^= self._obs_det_byte
        _, off_weights = self.gap_decoder.decode_batch(bit_packed_dets, return_weights=True, bit_packed_shots=True, bit_packed_predictions=True)
        gaps: np.ndarray = np.abs((on_weights - off_weights) * self.decibels_per_w)
        predictions: np.ndarray = on_weights < off_weights
        return predictions, gaps

    def decode_det_set(self, det_set: set[int]) -> tuple[bool, float]:
        dets = np.zeros(shape=(1, self.num_dets), dtype=np.bool_)
        for d in det_set:
            if d in self.postselected_detectors:
                return False, 0
            dets[0][d] = 1
        predictions, gaps = self._decode_batch_overwrite_last_byte(np.packbits(dets, bitorder='little', axis=1))
        return predictions[0], math.ceil(gaps[0])


def clipped_matchable_dem(flat_dem: stim.DetectorErrorModel, clip: AbstractSet[int]) -> stim.DetectorErrorModel:
    neighbors = collections.defaultdict(dict)

    heap: list[tuple[float, int, int]] = []
    boundaries = set()
    for inst in flat_dem:
        if inst.type == 'error':
            if any(t.is_separator() for t in inst.targets_copy()):
                continue
            err = _DemError.from_error_instruction(inst)
            w = -math.log(err.p / (1 - err.p))
            if len(err.det_set) == 1:
                a, = err.det_set
                heapq.heappush(heap, (w, a, err.obs_mask))
                boundaries.add(a)
            elif len(err.det_set) == 2:
                a, b = err.det_set
                neighbors[a][b] = (w, err.obs_mask)
                neighbors[b][a] = (w, err.obs_mask)

    classification: dict[int, tuple[int, float]] = {}
    while heap:
        cost, node, obs = heapq.heappop(heap)
        if node in classification:
            continue
        classification[node] = (obs, cost)
        for neighbor, (extra_cost, extra_obs) in neighbors[node].items():
            if neighbor not in classification:
                heapq.heappush(heap, (cost + extra_cost, neighbor, obs ^ extra_obs))

    new_dem = stim.DetectorErrorModel()
    for inst in flat_dem:
        if inst.type != 'error' or sum(t.is_relative_detector_id() and t.val in clip for t in inst.targets_copy()) < 2:
            new_dem.append(inst)
    for c in clip:
        if c not in boundaries and c in classification:
            obs, w = classification[c]
            p = math.exp(-w) / (math.exp(-w) + 1)
            targets = [stim.target_relative_detector_id(c)]
            for b in int_to_flipped_bits(obs):
                targets.append(stim.target_logical_observable_id(b))
            new_dem.append('error', [p], targets)
    return new_dem


def _dem_with_obs_detector(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    obs_det = stim.target_relative_detector_id(dem.num_detectors)
    new_dem = stim.DetectorErrorModel()
    new_dem.append('detector', [-10, -10, -10, -10, -10], [obs_det])
    for inst in dem:
        if inst.type == 'error':
            targets = inst.targets_copy()
            new_targets = []
            for t in targets:
                if t.is_logical_observable_id():
                    new_targets.append(obs_det)
                new_targets.append(t)
            new_dem.append('error', inst.args_copy(), new_targets)
        else:
            new_dem.append(inst)
    return new_dem
