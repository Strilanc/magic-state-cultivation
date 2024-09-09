import random
import time

import sinter
import stim

import gen
from latte.vec_sim import VecSim


class VecInterceptSampler(sinter.Sampler):
    """Samples while overriding S rotations with powers of T.

    This sampler is highly specialized for injection circuits where
    consistent powers of T all distill correctly.

    Uses a vector simulator to make it possible to perform non
    stabilizer gates.
    """

    def __init__(self, turns: float, sweep_bit_randomization: bool):
        self.turns = turns
        self.sweep_bit_randomization = sweep_bit_randomization

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledVecInterceptSampler(task, self.turns, self.sweep_bit_randomization)


class CompiledVecInterceptSampler(sinter.CompiledSampler):
    def __init__(self, task: sinter.Task, turns: float, sweep_bit_randomization: bool):
        self.task = task
        self.turns = turns
        self.sweep_bit_randomization = sweep_bit_randomization

    def sample(self, shots: int) -> sinter.AnonTaskStats:
        result = sinter.AnonTaskStats()
        for _ in range(shots):
            result += sample_circuit_with_vec_sim(
                self.task.circuit,
                self.turns,
                self.sweep_bit_randomization,
            )
        return result


def sample_circuit_with_vec_sim(circuit: stim.Circuit, turns: float, sweep_bit_randomization: bool) -> sinter.AnonTaskStats:
    t0 = time.monotonic()
    assert turns % 0.25 == 0
    turns %= 2
    t_count = round(turns * 4)
    sim = VecSim()
    measurements = []
    detectors = []
    observables = []
    sweep_bits = {
        b: sweep_bit_randomization and random.random() < 0.5
        for b in range(circuit.num_sweep_bits)
    }
    discard_shot = False
    for q in range(circuit.num_qubits):
        sim.do_qalloc_z(q)
    for inst in circuit:
        if inst.name == 'S':
            for q in inst.targets_copy():
                for _ in range(t_count):
                    sim.do_t(q.qubit_value)
        elif inst.name == 'S_DAG':
            for q in inst.targets_copy():
                for _ in range(t_count):
                    sim.do_t_dag(q.qubit_value)
        elif inst.name == 'MPP':
            for terms in inst.target_groups():
                combined_targets = []
                for term in terms:
                    combined_targets.append(term)
                    combined_targets.append(stim.target_combiner())
                combined_targets.pop()
                if all(term.is_y_target for term in terms):
                    for term in terms:
                        for _ in range(t_count):
                            sim.do_t_dag(term.qubit_value)
                        sim.do_s(term.qubit_value)
                sim.do_stim_instruction(
                    stim.CircuitInstruction('MPP', combined_targets, inst.gate_args_copy()),
                    sweep_bits=sweep_bits,
                    out_measurements=measurements,
                    out_detectors=detectors,
                    out_observables=observables,
                )
                if all(term.is_y_target for term in terms):
                    for term in terms:
                        sim.do_s_dag(term.qubit_value)
                        for _ in range(t_count):
                            sim.do_t(term.qubit_value)
        elif inst.name == 'DETECTOR':
            b = False
            for q in inst.targets_copy():
                assert q.is_measurement_record_target
                b ^= measurements[q.value]
            if b:
                discard_shot = True
                break
        else:
            sim.do_stim_instruction(
                inst,
                sweep_bits=sweep_bits,
                out_measurements=measurements,
                out_detectors=detectors,
                out_observables=observables,
            )
    t1 = time.monotonic()
    if discard_shot:
        return sinter.AnonTaskStats(discards=1, shots=1, seconds=t1 - t0)
    if any(observables):
        return sinter.AnonTaskStats(errors=1, shots=1, seconds=t1 - t0)
    return sinter.AnonTaskStats(shots=1, seconds=t1 - t0)
