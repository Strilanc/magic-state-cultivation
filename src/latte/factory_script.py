import collections
import itertools
import json
import pathlib
import urllib.parse
from typing import Literal, Any, Iterable, cast

from latte.factory_script_instruction import FactoryScriptInstruction, \
    PauliStringTarget, TQubitTarget
from latte.vec_sim import VecSim


class FactoryScript:
    """Stores a parsed script defining a distillation factory.

    An example script:

        PROMISE num_t_outputs=1
        PROMISE distance=3
        PROMISE assume_checks_fail_with_certainty=True
        PROMISE num_t_used=15
        PROMISE max_storage=5
        PROMISE num_t_checks=4
        ALLOC XXXXX
        T ZZZ__
        T ZZ_Z_
        T ZZ__Z
        T Z_ZZ_
        T Z_Z_Z
        T Z__ZZ
        T _ZZZ_
        T _ZZ_Z
        T _Z_ZZ
        T __ZZZ
        T ZZZZZ
        FLIP XXXXX
        POSTSELECT_RELEASE T0 T1 T2 T3
        OUTPUT_RELEASE T4
    """

    def __init__(
        self,
        *,
        instructions: Iterable['FactoryScriptInstruction'],
        name: str,
        distance: int = -1,
        num_t_used: int | Literal['auto'] = 'auto',
        num_t_outputs: int | Literal['auto'] = 'auto',
        num_checks: int | Literal['auto'] = 'auto',
        max_storage: int | Literal['auto'] = 'auto',
        num_t_checks: int | Literal['auto'] = 'auto',
        assume_checks_fail_with_certainty: bool = False,
    ):
        self.instructions = tuple(instructions)
        assert all(isinstance(e, FactoryScriptInstruction) for e in self.instructions)
        self.name = name
        self.distance = distance
        self.num_t_used: int = num_t_used if num_t_used != 'auto' else self._recompute_num_t_used()
        self.num_t_checks: int = num_t_checks if num_t_checks != 'auto' else self._recompute_num_t_checks()
        self.num_t_outputs: int = num_t_outputs if num_t_outputs != 'auto' else self._recompute_num_t_outputs()
        self.num_checks: int = num_checks if num_checks != 'auto' else self._recompute_num_checks()
        self.max_storage: int = max_storage if max_storage != 'auto' else self._recompute_max_storage()
        self.assume_checks_fail_with_certainty = assume_checks_fail_with_certainty
        self.max_qubit_index = self._recompute_max_qubit_index()

    @staticmethod
    def read_from_path(path: str | pathlib.Path):
        with open(path, 'r') as f:
            contents = f.read()
        return FactoryScript.read_from_file_contents(name=pathlib.Path(path).name, contents=contents)

    @staticmethod
    def read_from_file_contents(*, name: str, contents: str):
        instructions = []
        promises = {}
        for line in contents.splitlines():
            if '#' in line:
                line = line.split('#')[0]
            line = line.strip()
            if line.lower().startswith('promise'):
                line = line[7:].strip()
                k, v = line.split('=')
                if v == 'True':
                    v = True
                else:
                    v = int(v)
                promises[k.strip()] = v
            else:
                parsed = FactoryScriptInstruction.from_line(line)
                if parsed is not None:
                    instructions.append(parsed)
        num_t_used = promises.pop('num_t_used', 'auto')
        num_t_outputs = promises.pop('num_t_outputs', 'auto')
        num_checks = promises.pop('num_checks', 'auto')
        max_storage = promises.pop('max_storage', 'auto')
        num_t_checks = promises.pop('num_t_checks', 'auto')
        distance = promises.pop('distance', -1)
        assume_checks_fail_with_certainty = promises.pop('assume_checks_fail_with_certainty', False)
        if promises:
            raise NotImplementedError(f'{promises=}')
        return FactoryScript(
            instructions=instructions,
            name=name,
            distance=distance,
            num_t_used=num_t_used,
            num_t_outputs=num_t_outputs,
            num_checks=num_checks,
            max_storage=max_storage,
            num_t_checks=num_t_checks,
            assume_checks_fail_with_certainty=assume_checks_fail_with_certainty,
        )

    @staticmethod
    def from_quirk_url_approx(url: str) -> 'FactoryScript':
        url = urllib.parse.unquote(url)
        assert '#circuit=' in url
        circuit_text = url.split('#circuit=')[1]
        cols = json.loads(circuit_text)['cols']
        instructions = []
        for col in cols:
            controls = []
            swaps = []
            parities: dict[int, Literal['X', 'Y', 'Z']] = {}
            global_phase = 0
            for k, c in enumerate(col):
                if c == '•':
                    controls.append(k)
                elif c == 'Swap':
                    swaps.append(k)
                elif c == 'xpar':
                    parities[k] = 'X'
                elif c == 'ypar':
                    parities[k] = 'Y'
                elif c == 'zpar':
                    parities[k] = 'Z'
                elif c == '√i':
                    global_phase += 1
                elif c == '√-i':
                    global_phase -= 1
                elif c == 'i':
                    global_phase += 2
                elif c == '-i':
                    global_phase -= 2
                else:
                    continue
                col[k] = 1

            if controls and global_phase:
                raise NotImplementedError()
            if len(swaps) == 2:
                instructions.append(FactoryScriptInstruction(name='SWAP', targets=tuple(swaps)))
            elif len(swaps) == 0:
                pass
            else:
                raise NotImplementedError(f'{swaps=}')
            if global_phase == 0:
                pass
            elif global_phase == 1:
                instructions.append(FactoryScriptInstruction(name='T', targets=(PauliStringTarget(parities, sign=+1),)))
            elif global_phase == -1:
                instructions.append(FactoryScriptInstruction(name='T', targets=(PauliStringTarget(parities, sign=-1),)))
            else:
                raise NotImplementedError(f'{global_phase=}')

            for k, c in enumerate(col):
                if c == 1:
                    continue
                elif c == 'H':
                    instructions.append(FactoryScriptInstruction(name='H', targets=(k,)))
                elif c == 'X' or c == 'Z':
                    assert not parities
                    if len(controls) == 1:
                        instructions.append(FactoryScriptInstruction(name=cast(Any, f'C{c}'), targets=(controls[0], k)))
                    elif len(controls) == 0:
                        instructions.append(FactoryScriptInstruction(name='FLIP', targets=(PauliStringTarget({k: c}, sign=1),)))
                    else:
                        raise NotImplementedError(f'{c=}, {controls=}')
                elif c == '|0⟩⟨0|':
                    assert not parities and not controls
                    instructions.append(FactoryScriptInstruction(name='POSTSELECT', targets=(k,)))
                elif c == '|+⟩⟨+|':
                    assert not parities and not controls
                    instructions.append(FactoryScriptInstruction(name='POSTSELECT', targets=(PauliStringTarget({k: 'X'}, sign=1),)))
                elif c == 'Z^-¼':
                    assert not controls and not parities
                    instructions.append(FactoryScriptInstruction(name='T', targets=(PauliStringTarget({k: 'Z'}, sign=-1),)))
                elif c == 'Z^¼':
                    assert not controls and not parities
                    instructions.append(FactoryScriptInstruction(name='T', targets=(PauliStringTarget({k: 'Z'}, sign=1),)))
                elif c == '…' or c == 'Amps1' or c == 'Amps3':
                    pass
                else:
                    raise NotImplementedError(c)
        return FactoryScript(name='quirk', instructions=instructions)

    def _recompute_num_t_used(self) -> int:
        n = 0
        for instruction in self.instructions:
            if instruction.name == 'T':
                n += len(instruction.targets)
            elif instruction.name == 'POSTSELECT_RELEASE' or instruction.name == 'ALLOC':
                for t in instruction.targets:
                    if isinstance(t, TQubitTarget):
                        n += 1
        return n

    def _recompute_num_t_outputs(self) -> int:
        n = 0
        for instruction in self.instructions:
            if instruction.name == 'OUTPUT_RELEASE':
                for t in instruction.targets:
                    if isinstance(t, TQubitTarget):
                        n += 1
        return n

    def _recompute_num_t_checks(self) -> int:
        n = 0
        for instruction in self.instructions:
            if instruction.name == 'POSTSELECT_RELEASE':
                for t in instruction.targets:
                    if isinstance(t, TQubitTarget):
                        n += 1
        return n

    def _recompute_num_checks(self) -> int:
        n = 0
        for instruction in self.instructions:
            if instruction.name == 'POSTSELECT_RELEASE':
                for t in instruction.targets:
                    if isinstance(t, PauliStringTarget):
                        n += len(t.q2i)
                    else:
                        n += 1
            elif instruction.name == 'POSTSELECT':
                n += len(instruction.targets)

        return n

    def _recompute_max_storage(self) -> int:
        q = 0
        max_q = 0
        for instruction in self.instructions:
            if instruction.name == 'ALLOC':
                for t in instruction.targets:
                    if isinstance(t, PauliStringTarget):
                        q += len(t.q2i)
                    else:
                        q += 1
                max_q = max(max_q, q)
            if instruction.name == 'POSTSELECT_RELEASE' or instruction.name == 'OUTPUT_RELEASE' or instruction.name == 'RELEASE':
                for t in instruction.targets:
                    if isinstance(t, PauliStringTarget):
                        q -= len(t.q2i)
                    elif isinstance(t, (int, TQubitTarget)):
                        q -= 1
        return max_q

    def _recompute_max_qubit_index(self) -> int:
        max_q = 0
        for instruction in self.instructions:
            for t in instruction.targets:
                if isinstance(t, int):
                    max_q = max(t, max_q)
                elif isinstance(t, TQubitTarget):
                    max_q = max(t.q, max_q)
                elif isinstance(t, PauliStringTarget):
                    max_q = max(max(t.q2i.keys()), max_q)
                else:
                    raise NotImplementedError(f'{instruction=}')
        return max_q

    def verify(
            self,
            *,
            count_uncaught_benign_as_error: bool,
            require_distance_exact: bool,
            skip_distance_check: bool = False,
    ):
        if self.num_t_used != self._recompute_num_t_used():
            raise ValueError(f'{self.num_t_used=} != {self._recompute_num_t_used()=}')
        if self.num_t_checks != self._recompute_num_t_checks():
            raise ValueError(f'{self.num_t_checks=} != {self._recompute_num_t_checks()=}')
        if self.num_t_outputs != self._recompute_num_t_outputs():
            raise ValueError(f'{self.num_t_outputs=} != {self._recompute_num_t_outputs()=}')
        if self.num_checks != self._recompute_num_checks():
            raise ValueError(f'{self.num_checks=} != {self._recompute_num_checks()=}')
        if self.max_storage != self._recompute_max_storage():
            raise ValueError(f'{self.max_storage=} != {self._recompute_max_storage()=}')

        outs = self.simulate_with_injected_t_errors(
            set(),
            prefer_check_result=True,
            prefer_output_result=True,
        )
        if any(flag for _, flag in outs):
            raise ValueError("Noiseless execution didn't succeed unconditionally.")

        if not skip_distance_check:
            self._verify_distance(
                count_uncaught_benign_as_error=count_uncaught_benign_as_error,
                require_distance_exact=require_distance_exact,
            )

    def _verify_distance(self, *, count_uncaught_benign_as_error: bool, require_distance_exact: bool):
        def _fail_distance_verify(err_msg: str):
            raise ValueError(f"{err_msg}\n"
                             f"    name={self.name}\n"
                             f"    errors={err_set_str(inject_key, self.num_t_used)}\n"
                             f"    checks={err_set_str(caught_key, self.num_checks)}\n"
                             f"    output={err_set_str(fail_key, self.num_t_outputs)}")

        seen_failures = collections.defaultdict(list)
        saw_exact_distance_count = 0
        for d in range(max(1, self.distance + require_distance_exact)):
            if self.assume_checks_fail_with_certainty and d > self.distance / 2 + (0.5 if require_distance_exact else 0):
                break

            for injected in itertools.combinations(range(self.num_t_used), d):
                inject_key = 0
                for j in injected:
                    inject_key |= 1 << j
                outs = self.simulate_with_injected_t_errors(
                    set(injected),
                    prefer_check_result=None if injected else True,
                    prefer_output_result=None if injected else True,
                )
                caught_key = 0
                fail_key = 0
                caught_index = 0
                fail_index = 0
                for out in outs:
                    if out[0] == 'CHECK':
                        if out[1]:
                            caught_key |= 1 << caught_index
                        caught_index += 1
                    elif out[0] == 'OUTPUT':
                        if out[1]:
                            fail_key |= 1 << fail_index
                        fail_index += 1
                    else:
                        raise NotImplementedError(f'{out=}')
                if fail_key and not inject_key:
                    _fail_distance_verify("Noiseless case had bad outputs.")
                elif caught_key and not inject_key:
                    _fail_distance_verify("Noiseless case had detection events.")
                elif fail_key and not caught_key:
                    if len(injected) == d:
                        saw_exact_distance_count += 1
                    else:
                        _fail_distance_verify("Failed to catch a bad output.")
                elif not caught_key and inject_key and count_uncaught_benign_as_error:
                    if len(injected) == d:
                        saw_exact_distance_count += 1
                    else:
                        _fail_distance_verify("Failed to catch an injected error, even though it was benign, because count_uncaught_benign_as_error=True.")

                if self.assume_checks_fail_with_certainty:
                    self_inject_key = inject_key
                    self_caught_key = caught_key
                    self_fail_key = fail_key
                    if self_caught_key in seen_failures:
                        for other_inject_key, other_fail_key in seen_failures[self_caught_key]:
                            inject_key = self_inject_key ^ other_inject_key
                            caught_key = 0
                            fail_key = self_fail_key ^ other_fail_key
                            if fail_key:
                                if inject_key.bit_count() < self.distance:
                                    _fail_distance_verify("Failed to catch a bad output.")
                                elif inject_key.bit_count() == self.distance:
                                    saw_exact_distance_count += 1

                        if count_uncaught_benign_as_error:
                            for other_inject_key, other_fail_key in seen_failures[self_caught_key]:
                                if other_inject_key:
                                    inject_key = self_inject_key ^ other_inject_key
                                    caught_key = 0
                                    fail_key = self_fail_key ^ other_fail_key
                                    if inject_key.bit_count() < self.distance:
                                        _fail_distance_verify("Failed to catch an injected error, even though it was benign, because count_uncaught_benign_as_error=True.")
                                    elif inject_key.bit_count() == self.distance:
                                        saw_exact_distance_count += 1
                    seen_failures[self_caught_key].append((self_inject_key, self_fail_key))
        if require_distance_exact and saw_exact_distance_count == 0 and self.distance != -1:
            raise ValueError("Distance is too low.")

    def __str__(self) -> str:
        return self.name

    def to_instructions(self) -> str:
        return '\n'.join(str(e) for e in self.instructions)

    def to_quirk_url(self) -> str:
        circuit_cols = []
        for instruction in self.instructions:
            cols = [{}, {}, {}]
            col = cols[0]
            if instruction.name == 'ALLOC':
                for t in instruction.targets:
                    if isinstance(t, PauliStringTarget):
                        for q, p in t.q2i.items():
                            if p == 'X':
                                col[q] = 'H'
                            elif p == 'Z':
                                col[q] = 1
                            elif p == 'Y':
                                col[q] = 'X^½'
                    elif isinstance(t, TQubitTarget):
                        cols[0][t.q] = 'H'
                        cols[1][t.q] = 'Z^-¼' if t.dag else 'Z^¼'
                    else:
                        raise NotImplementedError(f'{instruction=}')
            elif instruction.name == 'FLIP':
                for t in instruction.targets:
                    if isinstance(t, PauliStringTarget):
                        for q, p in t.q2i.items():
                            col[q] = p
                    else:
                        raise NotImplementedError(f'{instruction=}')
            elif instruction.name == 'CX' or instruction.name == 'CZ':
                assert all(isinstance(e, int) for e in instruction.targets)
                assert len(instruction.targets) % 2 == 0
                for k in range(0, len(instruction.targets), 2):
                    a = instruction.targets[k]
                    b = instruction.targets[k + 1]
                    while len(cols) <= k // 2:
                        cols.append({})
                    cols[k // 2][a] = '•'
                    cols[k // 2][b] = 'X' if instruction.name == 'CX' else 'Z'
            elif instruction.name == 'SWAP':
                assert all(isinstance(e, int) for e in instruction.targets)
                assert len(instruction.targets) % 2 == 0
                for k in range(0, len(instruction.targets), 2):
                    a = instruction.targets[k]
                    b = instruction.targets[k + 1]
                    while len(cols) <= k // 2:
                        cols.append({})
                    if a > b:
                        a, b = b, a
                    if b == a + 1:
                        cols[k // 2][a] = '<<2'
                    else:
                        cols[k // 2][a] = 'Swap'
                        cols[k // 2][b] = 'Swap'
            elif instruction.name == 'T' or instruction.name == 'MAGIC_PERFECT_T':
                for t in instruction.targets:
                    if isinstance(t, PauliStringTarget):
                        free_q = set(range(min(16, self.max_qubit_index + 2)))
                        for q, p in t.q2i.items():
                            col[q] = p.lower() + 'par'
                            free_q.remove(q)
                        anc_q = max(free_q)
                        col[anc_q] = '√-i' if t.sign == -1 else '√i'
                    else:
                        raise NotImplementedError(f'{instruction=}')
            elif instruction.name == 'H':
                for t in instruction.targets:
                    if isinstance(t, int):
                        col[t] = 'H'
                    else:
                        raise NotImplementedError(f'{instruction=}')
            elif instruction.name == 'MAGIC_PERFECT_CCZ':
                a, b, c = instruction.targets
                col[a] = '•'
                col[b] = '•'
                col[c] = 'Z'
            elif instruction.name == 'MAGIC_PERFECT_CS':
                a, b = instruction.targets
                col[a] = '•'
                col[b] = 'Z^½'
            elif instruction.name == 'S':
                for t in instruction.targets:
                    if isinstance(t, PauliStringTarget):
                        if len(t.q2i) == 1:
                            (q, p), = t.q2i.items()
                            col[q] = p + '^' + ('-' if t.sign == -1 else '') + '½'
                        else:
                            free_q = set(range(min(16, self.max_qubit_index + 2)))
                            for q, p in t.q2i.items():
                                col[q] = p.lower() + 'par'
                                free_q.remove(q)
                            anc_q = max(free_q)
                            col[anc_q] = '-i' if t.sign == -1 else 'i'
                    else:
                        raise NotImplementedError(f'{instruction=}')
            elif instruction.name == 'MAGIC_DETERMINISTIC_PREP' or instruction.name == 'POSTSELECT':
                for t in instruction.targets:
                    if isinstance(t, PauliStringTarget):

                        free_q = set(range(min(16, self.max_qubit_index + 2)))
                        for q, p in t.q2i.items():
                            if len(t.q2i) == 1:
                                if p == 'X':
                                    col[q] = '|+⟩⟨+|'
                                elif p == 'Y':
                                    col[q] = '|X⟩⟨X|'
                                elif p == 'Z':
                                    col[q] = '|0⟩⟨0|'
                                else:
                                    raise NotImplementedError(f'{instruction=}')
                            else:
                                col[q] = p.lower() + 'par'
                            free_q.remove(q)
                        if not free_q:
                            if cols[0][0] == 'xpar':
                                cols[0][0] = 'Z'
                                cols[1][0] = '|+⟩⟨+|'
                            elif cols[0][0] == 'ypar':
                                cols[0][0] = 'Z'
                                cols[1][0] = '|X⟩⟨X|'
                            elif cols[0][0] == 'zpar':
                                cols[0][0] = 'X'
                                cols[1][0] = '|0⟩⟨0|'
                            cols[2] = cols[0]
                        elif len(t.q2i) != 1:
                            col[max(free_q)] = '0'
                    else:
                        raise NotImplementedError(f'{instruction=}')
            elif instruction.name == 'POSTSELECT_RELEASE':
                for t in instruction.targets:
                    if isinstance(t, TQubitTarget):
                        cols[0][t.q] = 'Z^¼' if t.dag else 'Z^-¼'
                        cols[1][t.q] = 'H'
                        cols[2][t.q] = '|0⟩⟨0|'
                    elif isinstance(t, PauliStringTarget):
                        for q, p in t.q2i.items():
                            if p == 'X':
                                cols[1][q] = 'H'
                                cols[2][q] = '|0⟩⟨0|'
                            elif p == 'Y':
                                cols[1][q] = 'X^½'
                                cols[2][q] = '|0⟩⟨0|'
                            elif p == 'Z':
                                cols[2][q] = '|0⟩⟨0|'
                            else:
                                raise NotImplementedError(f'{instruction=}')

                    else:
                        raise NotImplementedError(f'{instruction=}')
            elif instruction.name == 'OUTPUT_RELEASE':
                for t in instruction.targets:
                    if isinstance(t, TQubitTarget):
                        cols[0][t.q] = 'Z^¼' if t.dag else 'Z^-¼'
                        cols[1][t.q] = 'H'
                        cols[2][t.q] = '…'
                    else:
                        raise NotImplementedError(f'{instruction=}')
            elif instruction.name == 'RELEASE':
                for t in instruction.targets:
                    if isinstance(t, TQubitTarget):
                        cols[0][t.q] = 'Z^¼' if t.dag else 'Z^-¼'
                        cols[1][t.q] = 'H'
                        cols[2][t.q] = 'NeGate'
                    elif isinstance(t, PauliStringTarget):
                        for q, p in t.q2i.items():
                            if p == 'X':
                                cols[1][q] = 'H'
                                cols[2][q] = 'NeGate'
                            elif p == 'Y':
                                cols[1][q] = 'X^½'
                                cols[2][q] = 'NeGate'
                            elif p == 'Z':
                                cols[2][q] = 'NeGate'
                            else:
                                raise NotImplementedError(f'{instruction=}')
                    else:
                        raise NotImplementedError(f'{instruction=}')
            else:
                raise NotImplementedError(f'{instruction=}')
            for c in cols:
                if not c:
                    continue
                col2 = []
                for k, v in c.items():
                    while len(col2) <= k:
                        col2.append(1)
                    col2[k] = v
                circuit_cols.append(col2)
        return 'https://algassert.com/quirk#circuit=' + urllib.parse.quote(json.dumps({'cols': circuit_cols}))

    def simulate_with_injected_t_errors(
        self,
        injected: set[int],
        *,
        prefer_check_result: bool | None = None,
        prefer_output_result: bool | None = None,
    ) -> list[Any]:
        sim = VecSim()
        err_index = 0
        outs = []

        def should_apply_next_error() -> bool:
            nonlocal err_index
            result = err_index in injected
            err_index += 1
            return result

        def forward(sub_instruction: FactoryScriptInstruction):
            outs.extend(sub_instruction.apply_to(
                sim,
                prefer_check_result=prefer_check_result,
                prefer_output_result=prefer_output_result))

        for instruction in self.instructions:
            if instruction.name == 'T':
                for t in instruction.targets:
                    if should_apply_next_error():
                        if isinstance(t, PauliStringTarget):
                            sim.do_paulis(t.q2i)
                        elif isinstance(t, int):
                            sim.do_z(t)
                        else:
                            raise NotImplementedError(f'{t=}')
                    forward(FactoryScriptInstruction(instruction.name, (t,)))
            elif instruction.name == 'POSTSELECT_RELEASE':
                for t in instruction.targets:
                    if isinstance(t, TQubitTarget):
                        if should_apply_next_error():
                            sim.do_z(t.q)
                    forward(FactoryScriptInstruction(instruction.name, (t,)))
            elif instruction.name == 'ALLOC':
                for t in instruction.targets:
                    forward(FactoryScriptInstruction(instruction.name, (t,)))
                    if isinstance(t, TQubitTarget):
                        if should_apply_next_error():
                            sim.do_z(t.q)
            else:
                forward(instruction)
        return outs


def err_set_str(x: int, n: int) -> str:
    assert x < 2**n, (x, 2**n)
    if n == 0:
        return ''
    ks = []
    for k in range(n):
        if x & (1 << k):
            ks.append(str(k))
    return bin(x)[2:].rjust(n, '0').replace('0', '_').replace('1', 'E')[::-1] + '{' + ','.join(ks) + '}'
