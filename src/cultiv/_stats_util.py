#!/usr/bin/env python3

import collections
import dataclasses
import math
from typing import Callable

import sinter
import stim


@dataclasses.dataclass
class GapArg:
    source: sinter.TaskStats
    gap: int
    cur: sinter.AnonTaskStats
    less: sinter.AnonTaskStats
    more: sinter.AnonTaskStats
    at_least: sinter.AnonTaskStats
    at_most: sinter.AnonTaskStats


def compute_expected_injection_growth_volume(
        circuit: stim.Circuit,
        *,
        discard_rate: float | None = None,
):
    assert circuit != circuit.without_noise()
    det_times = {}
    postselected = set()
    coords = circuit.get_detector_coordinates()
    saw_dissipation = False
    num_layers = 0
    layer_qubits = collections.defaultdict(set)
    for inst in circuit.flattened():
        if inst.name in ['RX', 'MX', 'R', 'M', 'RY', 'MX', 'MPP']:
            saw_dissipation = True
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    layer_qubits[num_layers].add(t.qubit_value)
        elif inst.name in ['CZ', 'CX', 'S', 'H', 'S_DAG']:
            num_layers += saw_dissipation
            saw_dissipation = False
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    layer_qubits[num_layers].add(t.qubit_value)
        elif inst.name in ['TICK', 'QUBIT_COORDS', 'X_ERROR', 'DEPOLARIZE1', 'DEPOLARIZE2', 'Z_ERROR', 'OBSERVABLE_INCLUDE']:
            pass
        elif inst.name == 'DETECTOR':
            d = len(det_times)
            det_times[d] = num_layers
            if len(coords[d]) < 4 or coords[d][-1] == -9:
                postselected.add(d)
        else:
            raise NotImplementedError(f'{inst=}')

    for l in sorted(layer_qubits.keys()):
        layer_qubits[l] |= layer_qubits[l - 1]
    layer_passes = collections.defaultdict(lambda: 1)
    dem = circuit.detector_error_model()
    for inst in dem.flattened():
        if inst.type == 'error':
            layer = min((
                det_times[target.val]
                for target in inst.targets_copy()
                if target.is_relative_detector_id()
                if target.val in postselected
            ), default=None)
            if layer is not None:
                layer_passes[layer] *= 1 - inst.args_copy()[0]

    cost = 0
    survived = 1
    for layer in range(1, num_layers + 1):
        survived *= layer_passes[layer]
        cost += survived * len(layer_qubits[layer])
    if discard_rate is not None:
        survived = 1 - discard_rate
    return cost / survived


def stat_to_gap_stats(
        stats: list[sinter.TaskStats],
        rounding: int,
        func: Callable[[GapArg], sinter.AnonTaskStats],
) -> list[sinter.TaskStats]:
    return [
        gap_stat
        for stat in stats
        for gap_stat in _stat_to_gap_stats_single(
            stat,
            rounding=rounding,
            func=func
        )
    ]


def split_by_gap_threshold(stats: list[sinter.TaskStats], *, gap_rounding: int, keep_zero: bool = False) -> list[sinter.TaskStats]:
    stats = stat_to_gap_stats(
        stats,
        rounding=gap_rounding,
        func=lambda arg: sinter.AnonTaskStats(
            shots=arg.source.shots,
            discards=arg.at_least.discards + arg.less.shots,
            errors=arg.at_least.errors,
        ),
    )
    return [
        stat
        for stat in stats
        if keep_zero or stat.json_metadata.get('gap', 1) > 0
    ]


def split_into_gap_distribution(stats: list[sinter.TaskStats], *, gap_rounding: int) -> list[sinter.TaskStats]:
    stats = stat_to_gap_stats(
        stats,
        rounding=gap_rounding,
        func=lambda arg: sinter.AnonTaskStats(
            shots=arg.cur.shots,
            discards=arg.cur.discards,
            errors=arg.cur.errors,
        ),
    )
    return [
        stat.with_edits(
            errors=stat.errors if e else stat.shots - stat.errors,
            discards=0,
            shots=stat.json_metadata['src_shots'],
            strong_id=stat.strong_id + ':' + str(e),
            json_metadata={
                **stat.json_metadata,
                'gap': stat.json_metadata['gap'] * (-1 if e else +1),
            }
        )
        for stat in stats
        for e in [False, True]
    ]


def split_by_gap(stats: list[sinter.TaskStats], *, gap_rounding: int) -> list[sinter.TaskStats]:
    stats = stat_to_gap_stats(
        stats,
        rounding=gap_rounding,
        func=lambda arg: sinter.AnonTaskStats(
            shots=arg.cur.shots,
            discards=arg.cur.discards,
            errors=arg.cur.errors,
        ),
    )
    return [
        stat
        for stat in stats
    ]


def split_by_custom_count(
        stats: list[sinter.TaskStats],
) -> list[sinter.TaskStats]:
    result = []
    for stat in stats:
        for key, count  in stat.custom_counts.items():
            k, v = key.split('=')
            try:
                v = int(v)
            except TypeError:
                try:
                    v = float(v)
                except TypeError:
                    pass
            result.append(stat.with_edits(json_metadata={**stat.json_metadata, k: v, 'hits': count}, strong_id=stat.strong_id + f'{k},{v}'))
    return result


def sub_anon(a: sinter.AnonTaskStats, b: sinter.AnonTaskStats) -> sinter.AnonTaskStats:
    return sinter.AnonTaskStats(
        shots=a.shots - b.shots,
        errors=a.errors - b.errors,
        discards=a.discards - b.discards,
        seconds=a.seconds - b.seconds,
        custom_counts=a.custom_counts - b.custom_counts,
    )


def _stat_to_gap_stats_single(
        stat: sinter.TaskStats,
        rounding: int,
        func: Callable[[GapArg], sinter.AnonTaskStats],
) -> list[sinter.TaskStats]:
    if not stat.custom_counts:
        return [stat]
    global_discards = stat.discards
    anon_stat = stat.to_anon_stats()
    anon_stat = sinter.AnonTaskStats(shots=anon_stat.shots, errors=anon_stat.errors, discards=0, seconds=anon_stat.seconds, custom_counts=anon_stat.custom_counts)
    gap_stats = collections.defaultdict(sinter.AnonTaskStats)
    max_gap = max(int(k[1:]) for k in anon_stat.custom_counts.keys() if k.startswith('C') or k.startswith('E'))
    for cor_gap, hits in anon_stat.custom_counts.items():
        if cor_gap.startswith('C'):
            is_error = False
        elif cor_gap.startswith('E'):
            is_error = True
        else:
            raise NotImplementedError(f'{cor_gap=}')
        gap = int(cor_gap[1:])
        gap = max_gap if gap == max_gap else min(max_gap, round(gap / rounding) * rounding)
        assert gap >= 0
        gap_stats[gap] += sinter.AnonTaskStats(shots=hits, errors=hits * is_error)
    if not gap_stats:
        gap_stats[0] = anon_stat

    result = []
    less = sinter.AnonTaskStats()
    more = anon_stat
    more += sinter.AnonTaskStats(discards=-more.discards)
    for gap in sorted(gap_stats.keys()):
        gap_stat = gap_stats[gap]
        more = sub_anon(more, gap_stat)
        choice = func(GapArg(source=stat, gap=gap, cur=gap_stat, less=less, more=more, at_least=gap_stat + more, at_most=gap_stat + less))
        less += gap_stat
        result.append(sinter.TaskStats(
            strong_id=stat.strong_id + f':gap{gap}',
            decoder=stat.decoder,
            json_metadata={
                **stat.json_metadata,
                'gap': gap,
                'src_errors': stat.errors,
                'src_discards': stat.discards,
                'src_shots': stat.shots,
            },
            shots=choice.shots,
            errors=choice.errors,
            discards=choice.discards + global_discards,
            seconds=choice.seconds,
            custom_counts=choice.custom_counts,
        ))

    return result


def preprocess_intercepted_simulation_stats(stats: list[sinter.TaskStats]) -> list[sinter.TaskStats]:
    """
    - Adds 'sim' to metadata, based on the decoder field.
    - Overwrites the intercepted-basis of the circuit (b=Y) with the gate that the decoder used (b=T, S, or Z) based on the decoder field.
    """

    result = []
    for stat in stats:
        if 'intercept' not in stat.decoder:
            continue
        if stat.decoder == 'twirl_intercept_t' or stat.decoder == 'vec_intercept_t':
            continue
        result.append(stat.with_edits(
            json_metadata={
                **stat.json_metadata,
                'b': stat.decoder[-1].upper(),
                'sim': 'Vector' if 'vec' in stat.decoder else 'Stabilizer',
            },
            decoder=stat.decoder[:-2].replace('vec_intercept', 'Vector Sim').replace('twirl_intercept', 'Stabilizer Sim'),
        ))

    # Add synthetic S+Z data point
    # decoder_groups = sinter.group_by(stats, key=lambda stat: stat.decoder)
    # group1 = decoder_groups.get('twirl_intercept_s')
    # group2 = decoder_groups.get('twirl_intercept_z')
    # if group1 is not None and group2 is not None:
    #     ps1 = sinter.group_by(group1, key=lambda stat: stat.json_metadata['p'])
    #     ps2 = sinter.group_by(group2, key=lambda stat: stat.json_metadata['p'])
    #     for ps in ps1.keys() & ps2.keys():
    #         a, = ps1[ps]
    #         b, = ps2[ps]
    #         combined_error_rate = a.errors / (a.shots - a.discards) + b.errors / (b.shots - b.discards)
    #         shots = min(a.shots, b.shots)
    #         discard_rate = max(a.discards / a.shots, b.discards / b.shots)
    #         result.append(
    #             a.with_edits(
    #                 errors=math.ceil(shots * combined_error_rate * (1 - discard_rate)),
    #                 discards=math.ceil(shots * discard_rate),
    #                 shots=shots,
    #                 decoder=a.decoder[:-2].replace('vec_intercept', 'Vector Sim').replace('twirl_intercept', 'Stabilizer Sim'),
    #                 json_metadata={**a.json_metadata, 'b': 'S+Z', 'sim': 'Vector' if 'vec' in a.decoder else 'Stabilizer'},
    #                 strong_id=a.strong_id + ':' + b.strong_id,
    #             )
    #         )

    return result
