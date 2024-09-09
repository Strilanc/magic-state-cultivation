import collections
import dataclasses
import hashlib
import math
from typing import Iterator

import numpy as np
import stim


def int_to_flipped_bits(bits: int) -> list[int]:
    v = []
    if bits < 0:
        v.append(-1)
        bits ^= -1
    while bits:
        b = bits.bit_length() - 1
        v.append(b)
        bits ^= 1 << b
    return v


def bernoulli_sum(p: float, q: float) -> float:
    return p * (1 - q) + q * (1 - p)


def iter_pair_chunks(errs: np.ndarray) -> Iterator[np.ndarray]:
    buf = np.copy(errs)
    for k in range(1, len(errs)):
        w = buf[:k]
        w ^= errs[k]
        yield buf[:k]
        w ^= errs[k]


def iter_triplet_chunks(errs: np.ndarray) -> Iterator[np.ndarray]:
    buf = np.copy(errs)
    for k1 in range(1, len(errs)):
        w = buf[:k1]
        w ^= errs[k1]
        for k2 in range(k1 + 1, len(errs)):
            w ^= errs[k2]
            yield w
            w ^= errs[k2]
        w ^= errs[k1]


def iter_pair_and_triplet_chunks(errs: np.ndarray) -> Iterator[np.ndarray]:
    buf = np.copy(errs)
    for k1 in range(1, len(errs)):
        w = buf[:k1]
        yield w
        w ^= errs[k1]
        for k2 in range(k1 + 1, len(errs)):
            w ^= errs[k2]
            yield w
            w ^= errs[k2]
        w ^= errs[k1]


def iter_combo_chunks(errs: np.ndarray, min_w: int, max_w: int) -> Iterator[np.ndarray]:
    if max_w > 3:
        raise NotImplementedError(f'{max_w=} > 3')

    if min_w <= 0 <= max_w:
        yield errs[:1] & 0
    if min_w <= 1 <= max_w:
        yield errs

    buf = np.copy(errs)
    for k1 in range(1, len(errs)):
        w = buf[:k1]
        w ^= errs[k1]
        if min_w <= 2 <= max_w:
            yield w
        if 3 <= max_w:
            for k2 in range(k1 + 1, len(errs)):
                w ^= errs[k2]
                if min_w <= 3 <= max_w:
                    yield w
                w ^= errs[k2]
        w ^= errs[k1]


def iter_enumerate_combo_chunks(errs: np.ndarray, max_w: int) -> Iterator[tuple[np.ndarray, list[int]]]:
    indices = []
    if max_w >= 4:
        raise NotImplementedError(f'{max_w=} >= 4')

    if max_w >= 0:
        yield errs[:1] & 0, indices

    indices.append(-1)
    if max_w >= 1:
        yield errs, indices

    if max_w >= 2:
        buf = np.copy(errs)
        for k1 in range(1, len(errs)):
            indices.append(k1)
            w = buf[:k1]
            w ^= errs[k1]
            yield w, indices
            if max_w >= 3:
                for k2 in range(k1 + 1, len(errs)):
                    indices.append(k2)
                    w ^= errs[k2]
                    yield w, indices
                    w ^= errs[k2]
                    indices.pop()
            w ^= errs[k1]
            indices.pop()


@dataclasses.dataclass
class DemErrorSet:
    errors: list['DemError']
    probs: np.ndarray
    masks: np.ndarray

    def strong_id(self, max_weight: int) -> str:
        lines = []
        for err in sorted(self.errors):
            lines.append(f'{err.det}:{err.obs}')
        lines.append(f'w={max_weight}')
        return hashlib.sha1('\n'.join(lines).encode('utf8')).hexdigest()

    @staticmethod
    def from_dem(dem: stim.DetectorErrorModel) -> 'DemErrorSet':
        if dem.num_observables > 1:
            raise NotImplementedError(f'{dem.num_observables=} > 1')
        num_dets = dem.num_detectors
        acc = collections.defaultdict(float)
        for instruction in dem.flattened():
            if instruction.type == 'error':
                err = DemError.from_error_instruction(instruction)
                key = (err.det, err.obs)
                acc[key] = bernoulli_sum(acc[key], err.p)
        words = math.ceil((num_dets + 1) / 64)
        shape = (len(acc),)
        if num_dets < 8:
            dtype = np.uint8
        elif num_dets < 16:
            dtype = np.uint16
        elif num_dets < 32:
            dtype = np.uint32
        elif num_dets < 64:
            dtype = np.uint64
        else:
            shape = (len(acc), words)
            dtype = np.uint64
        masks = np.zeros(shape=shape, dtype=dtype)
        probs = np.zeros(shape=(len(acc),), dtype=np.float64)
        errors = []
        for k, ((det, obs), p) in enumerate(sorted(acc.items())):
            v = obs | (det << 1)
            if len(shape) == 1:
                masks[k] = v
            else:
                for k2 in range(words):
                    masks[k, k2] = v & ((1 << 64) - 1)
                    v >>= 64
            probs[k] = p
            errors.append(DemError(p=p, det=det, obs=obs))
        return DemErrorSet(masks=masks, probs=probs, errors=errors)

    def find_masks_reached_by_errors_up_to(self, *, max_distance: int, opposing_masks: set[int] | None = None) -> set[int]:
        if max_distance > 3:
            raise NotImplementedError(f'{max_distance=} > 3')

        result: set[int] = set()
        if opposing_masks is None:
            if len(self.masks.shape) == 1:
                for chunk in iter_combo_chunks(self.masks, min_w=0, max_w=max_distance):
                    result.update(chunk)
            else:
                for chunk in iter_combo_chunks(self.masks, min_w=0, max_w=max_distance):
                    for e in chunk:
                        val = 0
                        for ke in range(len(e)):
                            val ^= int(e[ke]) << (ke * 64)
                        result.add(val)
        else:
            if len(self.masks.shape) == 1:
                for chunk in iter_combo_chunks(self.masks, min_w=0, max_w=max_distance):
                    for e in chunk:
                        val = int(e)
                        if val ^ 1 in opposing_masks:
                            result.add(val | 1)
            else:
                for chunk in iter_combo_chunks(self.masks, min_w=0, max_w=max_distance):
                    for e in chunk:
                        val = 0
                        for ke in range(len(e)):
                            val ^= int(e[ke]) << (ke * 64)
                        if val ^ 1 in opposing_masks:
                            result.add(val | 1)
        return result

    def find_errors_for_midpoint_masks(self, midpoints: set[int], max_distance: int) -> dict[int, set[frozenset[int, ...]]]:
        result: dict[int, set[frozenset[int, ...]]] = {}
        if len(self.masks.shape) == 1:
            for chunk, indices in iter_enumerate_combo_chunks(self.masks, max_w=max_distance):
                for k in range(len(chunk)):
                    e = chunk[k]
                    val = int(e)
                    if val | 1 in midpoints:
                        if val not in result:
                            result[val] = set()
                        if indices:
                            indices[0] = k
                        result[val].add(frozenset(indices))
        else:
            for chunk, indices in iter_enumerate_combo_chunks(self.masks, max_w=max_distance):
                for k in range(len(chunk)):
                    e = chunk[k]
                    val = 0
                    for ke in range(len(e)):
                        val ^= int(e[ke]) << (ke * 64)
                    if val | 1 in midpoints:
                        if val not in result:
                            result[val] = set()
                        if indices:
                            indices[0] = k
                        result[val].add(frozenset(indices))
        return result

    def combine_midpoint_errors(self, midpoint_errors: dict[int, set[frozenset[int, ...]]]) -> set[frozenset[int, ...]]:
        result = set()
        for k, vs0 in midpoint_errors.items():
            vs1 = midpoint_errors.get(int(k) ^ 1, ())
            for v1 in vs1:
                for v0 in vs0:
                    result.add(v0 ^ v1)
        return result

    def find_logical_errors(self, max_distance: int) -> list[tuple[int, ...]]:
        if max_distance > 6:
            raise NotImplementedError(f'{max_distance} > 6')
        store_w = max_distance // 2
        search_w = max_distance - store_w
        potential_midpoints = self.find_masks_reached_by_errors_up_to(max_distance=store_w)
        actual_midpoints = self.find_masks_reached_by_errors_up_to(max_distance=search_w, opposing_masks=potential_midpoints)
        groups = self.find_errors_for_midpoint_masks(actual_midpoints, max(store_w, search_w))
        logical_errors = self.combine_midpoint_errors(groups)
        logical_errors = [tuple(sorted(e)) for e in logical_errors if len(e) <= max_distance]
        logical_errors = sorted(logical_errors, key=lambda e: (len(e), e))
        return logical_errors

    def expand_logical_errors(self, logical_errors: list[tuple[int, ...]]) -> list['DemCombinedError']:
        result = []

        no_err_chance = 1
        for e in self.errors:
            no_err_chance *= 1 - e.p

        for logical_error in logical_errors:
            p = no_err_chance
            det_mask = 0
            obs_mask = 0
            for e in logical_error:
                p *= self.probs[e] / (1 - self.probs[e])
                det_mask ^= self.errors[e].det
                obs_mask ^= self.errors[e].obs

            result.append(DemCombinedError(
                src_errors=logical_error,
                det_mask=det_mask,
                obs_mask=obs_mask,
                p=p,
            ))
        return result


def chance_of_exactly_0(ps: list[float]) -> float:
    total = 1
    for p in ps:
        total *= 1 - p
    return total


def chance_of_exactly_1(ps: list[float]) -> float:
    prod = 1
    for p in ps:
        if p != 1:
            prod *= 1 - p

    num_ones = sum(p == 1 for p in ps)
    if num_ones >= 2:
        return 0
    if num_ones == 1:
        return prod

    total = 0
    for p in ps:
        total += prod * p / (1 - p)
    return total


def analyze_solerr_discard_vs_error_rate(error_set: DemErrorSet, logical_errors: list['DemCombinedError']):
    e2f = {}
    for k_cond in range(len(error_set.errors)):
        p = 0
        for logical_err in logical_errors:
            p += math.prod(error_set.probs[k] for k in logical_err.src_errors if k_cond != k)
        e2f[k_cond] = p

    xs = []
    ys = []
    allowed_singleton_errors = set()
    selected_errors = set(range(len(error_set.errors)))
    for k, v in sorted(e2f.items(), key=lambda e: (e[1], e[0])):
        allowed_singleton_errors.add(k)
        selected_errors.remove(k)
        p0a = chance_of_exactly_0([error_set.probs[k] for k in selected_errors])
        p0b = chance_of_exactly_0([error_set.probs[k] for k in allowed_singleton_errors])
        p1b = chance_of_exactly_1([error_set.probs[k] for k in allowed_singleton_errors])
        keep_rate = p0a * (p0b + p1b)

        fail_sets = set()
        for logical_err in logical_errors:
            for k_cond in logical_err.src_errors:
                if k_cond in allowed_singleton_errors:
                    fail_sets.add(frozenset(e for e in logical_err.src_errors if e != k_cond))
                else:
                    fail_sets.add(frozenset(logical_err.src_errors))
        fail_rate = 0
        for fail_set in fail_sets:
            fail_rate += math.prod(error_set.probs[e] for e in fail_set)

        xs.append(1 - keep_rate)
        ys.append(fail_rate)
    return xs, ys


@dataclasses.dataclass(frozen=True)
class DemCombinedError:
    src_errors: tuple[int, ...]
    det_mask: int
    obs_mask: int
    p: float

    @property
    def det_list(self) -> list[int]:
        return int_to_flipped_bits(self.det_mask)

    @property
    def obs_list(self) -> list[int]:
        return int_to_flipped_bits(self.obs_mask)


@dataclasses.dataclass(frozen=True)
class DemError:
    p: float
    det: int
    obs: int

    def __lt__(self, other):
        if not isinstance(other, DemError):
            return NotImplemented
        return (self.det, self.obs, self.p) < (other.det, other.obs, other.p)

    def det_list(self) -> list[int]:
        return int_to_flipped_bits(self.det)

    def obs_list(self) -> list[int]:
        return int_to_flipped_bits(self.obs)

    def to_error_instruction(self) -> 'stim.DemInstruction':
        targets = []
        for d in self.det_list():
            targets.append(stim.target_relative_detector_id(d))
        for b in self.obs_list():
            targets.append(stim.target_logical_observable_id(b))
        return stim.DemInstruction('error', [self.p], targets)

    @staticmethod
    def from_error_instruction(instruction: stim.DemInstruction) -> 'DemError':
        p = instruction.args_copy()[0]
        det = 0
        obs = 0
        for target in instruction.targets_copy():
            if target.is_logical_observable_id():
                obs ^= 1 << target.val
            elif target.is_relative_detector_id():
                det ^= 1 << target.val
            elif not target.is_separator():
                raise NotImplementedError(f'{instruction}')
        return DemError(p=p, det=det, obs=obs)
