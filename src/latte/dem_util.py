import collections
import dataclasses
import itertools
from typing import Callable, DefaultDict

import sinter
import stim

import gen


def verify_can_gap_sample_dem(dem: stim.DetectorErrorModel):
    for inst in dem:
        if inst.type == 'repeat':
            assert isinstance(inst, stim.DemRepeatBlock)
            verify_can_gap_sample_dem(inst.body_copy())
        elif inst.type == 'error':
            cur = []
            for target in inst.targets_copy() + [stim.target_separator()]:
                if target.is_separator():
                    assert 1 <= len(cur) <= 2
                    cur.clear()
                else:
                    cur.append(target)


def circuit_with_super_postselected_detectors_cleared(circuit: stim.Circuit) -> stim.Circuit:
    new_circuit = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name == 'DETECTOR':
            args = inst.gate_args_copy()
            if len(args) > 4 and args[4] in [-99]:
                new_circuit.append("DETECTOR", [], args)
            else:
                new_circuit.append(inst)
        else:
            new_circuit.append(inst)
    return new_circuit


@dataclasses.dataclass(frozen=True)
class Symptom:
    obs_mask: int = 0
    dets: frozenset[int] = frozenset()

    @staticmethod
    def from_dem_targets(targets: list[stim.DemTarget]) -> 'Symptom':
        obs_mask = 0
        dets = []
        for t in targets:
            if t.is_separator():
                pass
            elif t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                obs_mask ^= 1 << t.val
            else:
                raise NotImplementedError(f'{t=}')
        return Symptom(obs_mask=obs_mask, dets=frozenset(gen.xor_sorted(dets)))

    def __mul__(self, other: 'Symptom') -> 'Symptom':
        return Symptom(
            obs_mask=self.obs_mask ^ other.obs_mask,
            dets=self.dets ^ other.dets,
        )


def bernoulli_sum(p: float, q: float) -> float:
    return p * (1 - q) + q * (1 - p)


def bernoulli_combo(*,
                    errors: dict[Symptom, float],
                    compressed_dets: frozenset[int],
                    max_errors: int,
                    error_size_cutoff: int,
                    detection_event_cutoff: int) -> dict[int, tuple[int, float]]:
    if max_errors == 0:
        return {}

    errors = {
        k: v
        for k, v in errors.items()
        if 1 <= len(k.dets) <= error_size_cutoff
        if not k.dets.isdisjoint(compressed_dets)
    }

    levels = [
        {},
        errors,
    ]
    while (len(levels) - 1) * 2 < max_errors:
        lvl = len(levels) + 1
        next_level: DefaultDict[Symptom, float] = collections.defaultdict(float)
        for s1, p1 in levels[-1].items():
            for s2, p2 in errors.items():
                s3: Symptom = s1 * s2
                p3: float = p1 * p2
                if len(s3.dets) <= detection_event_cutoff:
                    next_level[s3] += p3 / lvl
        if Symptom() in next_level:
            del next_level[Symptom()]
        levels.append(next_level)
    groups = [
        sinter.group_by(level.items(), key=lambda e: e[0].dets & compressed_dets)
        for level in levels
    ]
    while len(levels) <= max_errors:
        lvl = len(levels)
        next_level: DefaultDict[Symptom, float] = collections.defaultdict(float)
        if lvl % 2 == 0:
            g = groups[lvl // 2]
            divisor = lvl * lvl / 4
            for matches in g.values():
                for (s1, p1), (s2, p2) in itertools.combinations(matches, 2):
                    net_symptom = s1 * s2
                    net_p = p1 * p2
                    next_level[net_symptom] += net_p / divisor
        else:
            k1 = lvl // 2
            k2 = lvl - lvl // 2
            g1 = groups[k1]
            g2 = groups[k2]
            divisor = k1 * k2
            for det_key, matches1 in g1.items():
                matches2 = g2.get(det_key)
                if matches2 is None:
                    continue
                for s1, p1 in matches1:
                    for s2, p2 in matches2:
                        net_symptom = s1 * s2
                        net_p = p1 * p2
                        next_level[net_symptom] += net_p / divisor
        levels.append(next_level)

    total = collections.defaultdict(lambda: collections.defaultdict(float))
    for level in levels:
        for s, p in level.items():
            if len(s.dets) == 1 and s.dets.isdisjoint(compressed_dets):
                d, = s.dets
                total[d][s.obs_mask] += p

    result = {}
    for d, obs_p in total.items():
        obs, p = max(obs_p.items(), key=lambda e: e[1])
        result[d] = obs, p

    return result


def dem_with_compressed_detectors(
        dem: stim.DetectorErrorModel,
        compressed_detector_predicate: Callable[[list[float]], bool],
        max_compressed_errors: int,
        error_size_cutoff: int,
        detection_event_cutoff: int,
) -> stim.DetectorErrorModel:
    d2c = dem.get_detector_coordinates()
    out_dem = stim.DetectorErrorModel()
    compressed_dets = frozenset(
        det
        for det in range(dem.num_detectors)
        if compressed_detector_predicate(d2c.get(det, ()))
    )
    if not compressed_dets:
        return dem

    dem = dem.flattened()
    error_sets: DefaultDict[Symptom, float] = collections.defaultdict(float)
    for inst in dem:
        if inst.type != 'error':
            out_dem.append(inst)
            continue
        targets = inst.targets_copy()
        if not any(d.is_relative_detector_id() and d.val in compressed_dets for d in targets):
            out_dem.append(inst)
            continue
        symptom = Symptom.from_dem_targets(targets)
        error_sets[symptom] = bernoulli_sum(error_sets[symptom], inst.args_copy()[0])

    combos = bernoulli_combo(
        errors=error_sets,
        compressed_dets=compressed_dets,
        max_errors=max_compressed_errors,
        error_size_cutoff=error_size_cutoff,
        detection_event_cutoff=detection_event_cutoff,
    )

    for det, (obs, p) in combos.items():
        targets = [stim.target_relative_detector_id(det)]
        while obs:
            k = obs.bit_length() - 1
            targets.append(stim.target_logical_observable_id(k))
            obs ^= 1 << k
        out_dem.append('error', p, targets)

    for det in compressed_dets:
        out_dem.append('error', 0.5, [stim.target_relative_detector_id(det)])

    return out_dem


def dem_with_replaced_targets(
        dem: stim.DetectorErrorModel,
        replacements: dict[stim.DemTarget, stim.DemTarget | None],
) -> stim.DetectorErrorModel:
    new_dem = stim.DetectorErrorModel()

    dem = dem.flattened()
    for instruction in dem:
        old_targets = instruction.targets_copy()
        args = instruction.args_copy()
        new_targets = [replacements.get(t, t) for t in old_targets]

        # Delete targets mapped to None
        if any(t is None for t in new_targets):
            new_targets = [t for t in new_targets if t is not None]
            # Fix separator issues created by removing the target.
            while new_targets and new_targets[0].is_separator():
                new_targets.pop(0)
            while new_targets and new_targets[-1].is_separator():
                new_targets.pop()
            for k in range(len(new_targets) - 1)[::-1]:
                if new_targets[k].is_separator() and new_targets[k + 1].is_separator():
                    new_targets.pop(k + 1)
            if old_targets and not new_targets:
                continue

        new_targets = [t for t in new_targets if t is not None]
        if instruction.type == 'error':
            new_dem.append(instruction.type, args, new_targets)
        elif instruction.type == 'logical_observable' or instruction.type == 'detector':
            obs = [t for t in new_targets if t.is_logical_observable_id()]
            det = [t for t in new_targets if t.is_relative_detector_id()]
            if obs:
                new_dem.append('logical_observable', [], obs)
            if det:
                new_dem.append('detector', args, det)
        else:
            raise NotImplementedError(f'{instruction=}')

    return new_dem
