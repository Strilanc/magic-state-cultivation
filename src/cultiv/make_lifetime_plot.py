import argparse
import collections
import math
import pathlib

import numpy as np
import stim
from matplotlib import pyplot as plt

import gen
import cultiv


def circuit_to_layers(circuit: stim.Circuit) -> list[stim.Circuit]:
    cur_layer = stim.Circuit()
    prev_layers = []
    saw_two_qubit_gate = False
    saw_measurement = False
    for inst in circuit.flattened():
        data = stim.GateData(inst.name)
        if inst.name == 'QUBIT_COORDS':
            continue
        elif data.is_two_qubit_gate:
            if saw_measurement:
                saw_measurement = False
                prev_layers.append(cur_layer)
                cur_layer = stim.Circuit()
            saw_two_qubit_gate = True
            cur_layer.append(inst)
        elif data.produces_measurements:
            cur_layer.append(inst)
            saw_measurement = True
        elif data.is_reset or data.produces_measurements:
            if saw_two_qubit_gate:
                saw_two_qubit_gate = False
                saw_measurement = False
                prev_layers.append(cur_layer)
                cur_layer = stim.Circuit()
            cur_layer.append(inst)
        else:
            cur_layer.append(inst)
    if len(cur_layer):
        prev_layers.append(cur_layer)
    return prev_layers


def sample_times(circuit: stim.Circuit, shots: int) -> tuple[collections.Counter, list[int]]:
    sim = stim.FlipSimulator(batch_size=1024, num_qubits=circuit.num_qubits)
    layers = circuit_to_layers(circuit)

    qubit_counts = []
    used_qubits = set()
    for layer in layers:
        for inst in layer:
            if inst.name in ['R', 'RX']:
                for t in inst.targets_copy():
                    used_qubits.add(t.qubit_value)
        qubit_counts.append(len(used_qubits))
    survivors = np.zeros(1024, dtype=np.bool_)
    postselected_detectors = set()
    for det, coord in circuit.get_detector_coordinates().items():
        if len(coord) == 3 or coord[-1] == -9 or coord[4] == 0 or coord[4] == 4:
            postselected_detectors.add(det)
    counts = collections.Counter()

    shots_left = shots
    while shots_left > 0:
        sim.clear()
        cur_det = 0
        tick = 0
        survivors[:] = True
        for layer in layers:
            tick += 1
            sim.do(layer)
            if cur_det < sim.num_detectors:
                while cur_det < sim.num_detectors:
                    if cur_det in postselected_detectors:
                        fired = sim.get_detector_flips(detector_index=cur_det)
                        survivors &= ~fired
                    cur_det += 1
            counts[tick] += np.count_nonzero(survivors)

        counts[0] += 1024
        shots_left -= 1024

    return counts, qubit_counts


def desc(n: float) -> str:
    power = math.floor(math.log10(n))
    while 10**(power + 1) <= n:
        power += 1
    base = round(n / 10**power * 10) / 10
    base = str(base).ljust(3, '0')
    return fr'${base} \cdot 10^{{{power}}}$'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=None)
    parser.add_argument('--show', action='store_true')
    args = parser.parse_args()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    for dcolor in [3, 5]:
        ax: plt.Axes
        ax = ax1 if dcolor == 3 else ax2
        circuit = cultiv.make_end2end_cultivation_circuit(
            dcolor=dcolor,
            dsurface=15,
            basis='Y',
            r_growing=dcolor,
            r_end=10,
            inject_style='unitary',
        )
        circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
        num_shots = 1024*100
        ts, qs = sample_times(circuit, num_shots)
        if dcolor == 5:
            success_rate = 0.01  # Taken from the rejection-vs-logical-error plot near high end of rejection.
        elif dcolor == 3:
            success_rate = 0.25  # Taken from the rejection-vs-logical-error plot near high end of rejection.
        else:
            raise NotImplementedError()
        ts[max(ts.keys())] = num_shots * success_rate
        ts[max(ts.keys()) - 1] = ts[max(ts.keys())]
        xs_1 = []
        ys_1 = []
        xs_2 = []
        ys_2 = []

        cost = 0
        for k in range(max(ts.keys()) + 1):
            cost += (ts[k] * qs[min(k, len(qs) - 1)] / num_shots) / success_rate

        for k in range(max(ts.keys()) + 1):
            if k > 0:
                xs_1.append(k)
                ys_1.append(ys_1[-1])
            xs_1.append(k)
            ys_1.append(ts[k])
        for k, q in enumerate(qs):
            xs_2.append(k)
            xs_2.append(k + 1)
            ys_2.append(q)
            ys_2.append(q)
        xs_1 = np.array(xs_1)
        ys_1 = np.array(ys_1) / num_shots
        xs_2 = np.array(xs_2)
        ys_2 = np.array(ys_2) / max(qs)
        ax.plot(xs_1, ys_1, label='Surviving Shots')
        ax.plot(xs_2, ys_2, label='Qubits Activated')
        ax.fill_between(xs_1, ys_1 * 0, ys_1, alpha=0.2, color='C0')
        ax.fill_between(xs_2, ys_2 * 0, ys_2, alpha=0.2, color='C1')
        ax.set_ylim(0, 1.01)
        ax.set_xlim(0, max(xs_1))
        labels=[
            'Encode T',
            'Stabilize',
            'Check T',
            'Check T',
            *([
                  'Stabilize',
                  'Stabilize',
                  'Stabilize',
                  'Check T',
                  'Check T',
              ] * (dcolor == 5)),
            *(['Stabilize'] * dcolor),
            'Escaped!',
            *(['[wait for gap]'] * 9),
            'ready',
        ]
        ax.set_xticks(range(len(labels) + 1), [''] * (len(labels) + 1))
        ax.set_xticks([e + 0.5 for e in range(len(labels))], labels, rotation=90, minor=True)
        ax.xaxis.set_tick_params(length=0, which='minor')
        ax.grid()
        ax.legend()
        ax.set_yticks([x*0.1 for x in range(11)])
        ax.set_ylabel('Proportion')
        ax.set_title(f"Life of a Fault-Distance-{dcolor} Cultivation\n(p=0.001, d2=15, r1=d1, r2=10, noise=uniform)\n(expected qubit·rounds={desc(cost)})")
    fig.set_size_inches(1024 / 100, 512 / 100)
    fig.set_dpi(100)
    fig.tight_layout()
    if args.out is not None:
        fig.savefig(args.out)
        print(f'wrote file://{pathlib.Path(args.out).absolute()}')
    if args.show or args.out is None:
        plt.show()


if __name__ == '__main__':
    main()
