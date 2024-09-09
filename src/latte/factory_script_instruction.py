import dataclasses
from typing import Literal, cast, Any, Optional

from latte.vec_sim import VecSim


@dataclasses.dataclass
class PauliStringTarget:
    """Maps qubits to pauli bases.

    Depending on the context, this is either a combined Pauli product operator
    or a set of distinct single-qubit operators.
    """
    q2i: dict[int, Literal['X', 'Y', 'Z']]
    sign: int

    def __str__(self):
        s = '-' if self.sign == -1 else ''
        if len(self.q2i) == 1:
            (q, p), = self.q2i.items()
            return f'{s}{p}{q}'
        else:
            m = max(self.q2i.keys(), default=0) + 1
            return s + ''.join(self.q2i.get(k, '_') for k in range(m))


@dataclasses.dataclass
class TQubitTarget:
    """The X+Y (or X-Y) axis of a qubit."""
    q: int
    dag: bool


@dataclasses.dataclass
class FactoryScriptInstruction:
    name: Literal[
        # Allocates and initializes qubits.
        #     int target: Allocates the indexed qubit and initializes it to |0>.
        #     Pauli string target: Each qubit with a non-identity term is
        #           allocated and initialized to the specified basis.
        #     T target: injects a noisy |T>
        'ALLOC',

        # Performs a noisy T gate.
        #     int target: Applies the T gate to the indexed qubit.
        #     Pauli string target: Applies the T gate to the Pauli product
        #         observable specified by the Pauli string.
        #     T target: not allowed
        'T',

        # Performs noiseless Pauli flips.
        #     int target: not allowed
        #     Pauli string target: Applies the given Pauli string as Pauli
        #           operations.
        #     T target: not allowed
        'FLIP',

        # Performs a noiseless S gate.
        #     int target: Applies the S gate to the indexed qubit.
        #     Pauli string target: Applies the S gate to the Pauli product
        #         observable specified by the Pauli string.
        #     T target: not allowed
        'S',

        # Performs a noiseless H gate.
        #     int target: Applies the H gate to the indexed qubit.
        #     Pauli string target: not allowed
        #     T target: not allowed
        'H',

        # Performs noiseless swap gates.
        #     int target: Must be given an even number. Swaps aligned pairs.
        #     Pauli string target: not allowed
        #     T target: not allowed
        'SWAP',

        # Performs noiseless CX gates.
        #     int target: Must be given an even number. CXs aligned pairs.
        #     Pauli string target: not allowed
        #     T target: not allowed
        'CX',

        # Performs noiseless CZ gates.
        #     int target: Must be given an even number. CZs aligned pairs.
        #     Pauli string target: not allowed
        #     T target: not allowed
        'CZ',

        # Measures the given targets, checking for errors, and releases them.
        #     int target: Postselect-releases in the Z basis.
        #     Pauli string target: Each qubit with a non-identity term is
        #           measured in that basis (rejecting the factory run if the
        #           qubit is in the -1 eigenspace of that basis) then released.
        #     T target: Uses a noisy T gate to transform from |T> to |+>, then
        #           performs an X measurement to verify |+> instead of |->.
        'POSTSELECT_RELEASE',

        # Measures the given targets, checking for errors.
        #     int target: Postselects single qubits in the Z basis.
        #     Pauli string target: The given pauli product is measured as a
        #           product (as opposed to as individual qubits). The factory
        #           run is rejected if the result is not in the +1 eigenspace
        #           of the operator.
        #     T target: Uses a noisy T gate to transform from |T> to |+>, then
        #           performs an X measurement to verify |+> instead of |->.
        'POSTSELECT',

        # Yields the given targets as results, expected to be specific states.
        #     int target: The factory fails unless the qubit is |0>.
        #           Releases the qubit.
        #     Pauli string target: Each qubit with a non-identity term is
        #           measured in that basis (the factory fails if the
        #           qubit is in the -1 eigenspace of that basis) then released.
        #     T target: The factory fails unless the qubit is |T>. The simulator
        #           uses a noiseless T gate to check this. The qubit releases.
        'OUTPUT_RELEASE',

        # Releases qubits without checking or relying on them.
        #     int target: Releases the indexed qubit.
        #     Pauli string target: Each qubit with a non-identity term is
        #           released.
        #     T target: Releases the indexed qubit.
        'RELEASE',

        # Measures the given pauli product observable, forcing the result 0.
        #
        # Non-magical prep would measure the observable and do feedback if the
        # result was 1 instead of 0.
        #
        # Useful when initially trying to write a factory, but not yet wanting
        # to deal with solving for feedback operations.
        'MAGIC_DETERMINISTIC_PREP',

        # Performs a noiseless T gate. Useful for verifying some output states.
        #     int target: Applies the T gate to the indexed qubit.
        #     Pauli string target: Applies the T gate to the Pauli product
        #         observable specified by the Pauli string.
        #     T target: not allowed
        'MAGIC_PERFECT_T',

        # Performs a noiseless CCZ gate.
        #     int target: Must be given three. The qubits to CCZ.
        #     Pauli string target: not allowed
        #     T target: not allowed
        'MAGIC_PERFECT_CCZ',

        # Performs a noiseless CS gate.
        #     int target: Must be given two. The qubits to CS.
        #     Pauli string target: not allowed
        #     T target: not allowed
        'MAGIC_PERFECT_CS',
    ]
    targets: tuple[PauliStringTarget | int | TQubitTarget, ...]

    def __str__(self):
        return self.name + ''.join(' ' + str(e) for e in self.targets)

    @staticmethod
    def parse_instructions(text: str) -> tuple['FactoryScriptInstruction', ...]:
        instructions = []
        for instruction in text.splitlines():
            e = FactoryScriptInstruction.from_line(instruction)
            if e is not None:
                instructions.append(e)
        return tuple(instructions)

    @staticmethod
    def target_from_word(word: str) -> PauliStringTarget | int | TQubitTarget | None:
        if not word.strip():
            return None
        if word.startswith('-T'):
            return TQubitTarget(int(word[2:]), dag=True)
        if word.startswith('T'):
            return TQubitTarget(int(word[1:]), dag=False)
        if set(word) <= set('0123456789'):
            return int(word)
        if set(word) <= set('XYZI_+-'):
            d: dict[int, Literal['X', 'Y', 'Z']] = {}
            sign = +1
            if word.startswith('+'):
                word = word[1:]
            elif word.startswith('-'):
                word = word[1:]
                sign = -1
            for q, p in enumerate(word):
                if p in 'XYZ':
                    d[q] = cast(Literal['X', 'Y', 'Z'], p)
            return PauliStringTarget(d, sign)
        if set(word) <= set('XYZ0123456789*-+'):
            d: dict[int, Literal['X', 'Y', 'Z']] = {}
            sign = +1
            if word.startswith('+'):
                word = word[1:]
            elif word.startswith('-'):
                word = word[1:]
                sign = -1
            for pk in word.split('*'):
                pauli = pk[0]
                qubit = int(pk[1:])
                assert pauli in 'XYZ'
                assert qubit not in d
                d[qubit] = cast(Literal['X', 'Y', 'Z'], pauli)
            return PauliStringTarget(d, sign)
        raise NotImplementedError(f'{word=}')

    @staticmethod
    def from_line(line: str) -> Optional['FactoryScriptInstruction']:
        if '#' in line:
            line = line.split('#')[0]
        line = line.strip()
        if not line:
            return None
        terms = line.split()
        name = terms[0]
        targets = []
        for word in terms[1:]:
            target = FactoryScriptInstruction.target_from_word(word)
            if target is not None:
                targets.append(target)
        return FactoryScriptInstruction(name, tuple(targets))

    def apply_to(self, sim: VecSim, prefer_output_result: bool | None = None, prefer_check_result: bool | None = None) -> list[Any]:
        if self.name == 'ALLOC':
            for t in self.targets:
                if isinstance(t, PauliStringTarget):
                    for q, p in t.q2i.items():
                        sim.do_qalloc_p(q, p)
                elif isinstance(t, TQubitTarget):
                    sim.do_qalloc_x(t.q)
                    if t.dag:
                        sim.do_t_dag(t.q)
                    else:
                        sim.do_t(t.q)
                elif isinstance(t, int):
                    sim.do_qalloc_z(t)
                else:
                    raise NotImplementedError(f'{self=}')
            return []
        elif self.name == 'T' or self.name == 'MAGIC_PERFECT_T':
            for t in self.targets:
                if isinstance(t, PauliStringTarget):
                    sim.do_t_obs(t.q2i, sign=t.sign)
                elif isinstance(t, int):
                    sim.do_t(t)
                else:
                    raise NotImplementedError(f'{self=}')
            return []
        elif self.name == 'FLIP':
            for t in self.targets:
                if isinstance(t, PauliStringTarget):
                    sim.do_paulis(t.q2i)
                else:
                    raise NotImplementedError(f'{self=}')
            return []
        elif self.name == 'S':
            for t in self.targets:
                if isinstance(t, PauliStringTarget):
                    sim.do_s_obs(t.q2i, sign=t.sign)
                elif isinstance(t, int):
                    sim.do_s(t)
                else:
                    raise NotImplementedError(f'{self=}')
            return []
        elif self.name == 'SWAP':
            assert len(self.targets) % 2 == 0 and all(isinstance(e, int) for e in self.targets)
            for k in range(0, len(self.targets), 2):
                a = self.targets[k]
                b = self.targets[k + 1]
                sim.do_swap(a, b)
            return []
        elif self.name == 'MAGIC_PERFECT_CCZ':
            assert len(self.targets) == 3
            a, b, c = self.targets
            sim.do_ccz(a, b, c)
            return []
        elif self.name == 'MAGIC_PERFECT_CS':
            assert len(self.targets) == 2
            a, b = self.targets
            sim.do_cs(a, b)
            return []
        elif self.name == 'CX':
            assert len(self.targets) % 2 == 0 and all(isinstance(e, int) for e in self.targets)
            for k in range(0, len(self.targets), 2):
                a = self.targets[k]
                b = self.targets[k + 1]
                sim.do_cx(a, b)
            return []
        elif self.name == 'CZ':
            assert len(self.targets) % 2 == 0 and all(isinstance(e, int) for e in self.targets)
            for k in range(0, len(self.targets), 2):
                a = self.targets[k]
                b = self.targets[k + 1]
                sim.do_cz(a, b)
            return []
        elif self.name == 'POSTSELECT_RELEASE':
            outputs = []
            for t in self.targets:
                if isinstance(t, PauliStringTarget):
                    for q, p in t.q2i.items():
                        outputs.append(sim.do_mp_discard(q, p, prefer_result=prefer_check_result))
                elif isinstance(t, int):
                    outputs.append(sim.do_mz_discard(t, prefer_result=prefer_check_result))
                elif isinstance(t, TQubitTarget):
                    if t.dag:
                        sim.do_t(t.q)
                    else:
                        sim.do_t_dag(t.q)
                    outputs.append(sim.do_mx_discard(t.q, prefer_result=prefer_check_result))
                else:
                    raise NotImplementedError(f'{self=}')
            return [('CHECK', out) for out in outputs]
        elif self.name == 'POSTSELECT':
            outputs = []
            for t in self.targets:
                if isinstance(t, PauliStringTarget):
                    outputs.append(sim.do_measure_obs(t.q2i, sign=t.sign, prefer_result=prefer_output_result))
                elif isinstance(t, int):
                    outputs.append(sim.do_mz(t, prefer_result=prefer_output_result))
                else:
                    raise NotImplementedError(f'{self=}')
            return [('CHECK', out) for out in outputs]
        elif self.name == 'OUTPUT_RELEASE' or self.name == 'RELEASE':
            outputs = []
            for t in self.targets:
                if isinstance(t, PauliStringTarget):
                    for q, p in t.q2i.items():
                        outputs.append(sim.do_mp_discard(q, p, prefer_result=prefer_output_result))
                elif isinstance(t, int):
                    outputs.append(sim.do_mz_discard(t, prefer_result=prefer_output_result))
                elif isinstance(t, TQubitTarget):
                    if t.dag:
                        sim.do_t(t.q)
                    else:
                        sim.do_t_dag(t.q)
                    outputs.append(sim.do_mx_discard(t.q, prefer_result=prefer_output_result))
                else:
                    raise NotImplementedError(f'{self=}')
            if self.name == 'RELEASE':
                outputs.clear()
            return [('OUTPUT', out) for out in outputs]
        elif self.name == 'MAGIC_DETERMINISTIC_PREP':
            outputs = []
            for t in self.targets:
                if isinstance(t, PauliStringTarget):
                    outputs.append(sim.do_measure_obs(t.q2i, sign=t.sign, prefer_result=False))
                else:
                    raise NotImplementedError(f'{self=}')
            if any(outputs):
                raise ValueError("MAGIC_DETERMINISTIC_PREP deterministically failed")
            return []
        else:
            raise NotImplementedError(f'{self=}')
