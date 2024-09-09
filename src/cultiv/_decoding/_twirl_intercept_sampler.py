import time

import numpy as np
import sinter
import stim


class TwirlInterceptSampler(sinter.Sampler):
    """Samples while overriding S rotations with powers of T.

    This sampler is highly specialized for injection circuits where
    consistent powers of T all distill correctly.

    This sampler turns X or Y errors crossing a T into 50% X 50% Y
    errors, as suggested in https://arxiv.org/abs/2003.03049 . THIS
    SEEMS TO WORK VERY POORLY BE VERY CAREFUL USING THIS SAMPLER.
    """
    def __init__(self, turns: float):
        self.turns = turns

    def compiled_sampler_for_task(self, task: sinter.Task) -> sinter.CompiledSampler:
        return CompiledTwirlInterceptSampler(task, self.turns)


class CompiledTwirlInterceptSampler(sinter.CompiledSampler):
    def __init__(self, task: sinter.Task, turns: float):
        assert turns % 0.25 == 0 and 0 <= turns < 2
        self.task = task
        self.turns = turns
        self.num_qubits = task.circuit.num_qubits
        self.instructions = self.task.circuit.flattened()
        self.is_t_like = self.turns % 0.5 == 0.25
        self.is_s_like = self.turns % 1 == 0.5
        self.is_z_like = self.turns % 1 == 0
        self.simulator: stim.FlipSimulator | None = None

    def _t_twirl(self, qubits: list[int]):
        assert len(set(qubits)) == len(qubits)

        xy_twirl = np.zeros((self.num_qubits, self.simulator.batch_size), dtype=np.bool_)

        # Only twirl targeted qubits.
        for q in qubits:
            xy_twirl[q, :] = np.random.randint(0, 2, self.simulator.batch_size, dtype=np.bool_)

        # Don't twirl unless there is an X error or Y error present.
        for k, flip in enumerate(self.simulator.peek_pauli_flips()):
            xs,  _ = flip.to_numpy(bit_packed=False)
            xy_twirl[:, k] &= xs

        self.simulator.broadcast_pauli_errors(pauli='Z', mask=xy_twirl)

    def _do_t_or_s_or_z(self, targets: list[stim.GateTarget]):
        if self.is_t_like:
            # Twirled T gate randomizes X-vs-Y error distinction.
            self._t_twirl([t.qubit_value for t in targets])
        elif self.is_s_like:
            # S gates do the normal thing.
            self.simulator.do(stim.CircuitInstruction("S", [t.qubit_value for t in targets]))
        elif self.is_z_like:
            # Errors not changed by a Z gate.
            pass
        else:
            raise NotImplementedError(f'{self.turns=}')

    def _sample_once(self, shots: int) -> sinter.AnonTaskStats:
        shots = min(shots, 256)
        if self.simulator is None or shots != self.simulator.batch_size:
            self.simulator = stim.FlipSimulator(
                batch_size=shots,
                num_qubits=self.num_qubits,
                disable_stabilizer_randomization=self.is_t_like,
            )
        else:
            self.simulator.reset()

        for inst in self.instructions:
            if inst.name == 'S' or inst.name == 'S_DAG':
                self._do_t_or_s_or_z(inst.targets_copy())
            elif inst.name == 'MPP':
                args = inst.gate_args_copy()
                for terms in inst.target_groups():
                    is_tsz_basis_measurement = all(term.is_y_target for term in terms)

                    sub_targets = []
                    for term in terms:
                        if is_tsz_basis_measurement:
                            sub_targets.append(stim.target_x(term.qubit_value))
                        else:
                            sub_targets.append(term)
                        sub_targets.append(stim.target_combiner())
                    sub_targets.pop()

                    if is_tsz_basis_measurement:
                        self._do_t_or_s_or_z(terms)
                    self.simulator.do(stim.CircuitInstruction('MPP', sub_targets, args))
                    if is_tsz_basis_measurement:
                        self._do_t_or_s_or_z(terms)
            else:
                self.simulator.do(inst)

        discard_mask = np.any(self.simulator.get_detector_flips(bit_packed=False), axis=0)
        error_mask = np.any(self.simulator.get_observable_flips(bit_packed=False), axis=0)
        discards = np.count_nonzero(discard_mask)
        errors = np.count_nonzero(error_mask & ~discard_mask)
        return sinter.AnonTaskStats(shots=shots, errors=errors, discards=discards)

    def sample(self, shots: int) -> sinter.AnonTaskStats:
        t0 = time.monotonic()

        total = sinter.AnonTaskStats()
        shots_left = shots
        while shots_left > 0:
            sample = self._sample_once(shots_left)
            shots_left -= sample.shots
            total += sample
        t1 = time.monotonic()
        total += sinter.AnonTaskStats(seconds=t1 - t0 - total.seconds)
        return total
