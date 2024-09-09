import dataclasses
from typing import Iterable, Literal, cast, Any

from latte.lattice_surgery_layer import LatticeSurgeryLayer, InjectedError
from latte.vec_sim import VecSim


@dataclasses.dataclass
class Expression:
    variables: tuple[str, ...]
    op: Literal['', '^', '&', '<', '+', '&&', '^^']

    def __str__(self):
        return self.op.join(self.variables)

    def eval(self, state: dict[str, bool | int]) -> int | complex:
        v = self.try_eval(state)
        if v is None:
            for r in self.variables:
                if r not in state:
                    raise ValueError(f"Variable not defined yet: {r}")
            assert False, (self, state)
        return v

    def try_eval(self, state: dict[str, bool | int]) -> int | complex | None:
        vals = []
        for k in self.variables:
            try:
                vals.append(int(k))
            except ValueError:
                try:
                    vals.append(complex(k))
                except ValueError:
                    if k not in state:
                        return None
                    vals.append(state[k])

        if self.op == '':
            assert len(vals) == 1
            return vals[0]
        elif self.op == '^':
            acc = 0
            for v in vals:
                acc ^= v
            return acc
        elif self.op == '^^':
            acc = False
            for v in vals:
                acc ^= bool(v)
            return bool(acc)
        elif self.op == '+':
            acc = 0
            for v in vals:
                acc += v
            return acc
        elif self.op == '<':
            acc = 1
            for k in range(1, len(vals)):
                acc &= vals[k - 1] < vals[k]
            return acc
        elif self.op == '&':
            acc = -1
            for v in vals:
                acc &= v
            return acc
        elif self.op == '&&':
            for v in vals:
                if not v:
                    return False
            return True
        else:
            raise NotImplementedError(f'{self.op=}')


@dataclasses.dataclass
class DiscardShotAction:
    condition: Expression


@dataclasses.dataclass
class PrintAction:
    condition: Expression


@dataclasses.dataclass
class MeasureAction:
    name: str
    ground: bool


@dataclasses.dataclass
class ReplaceAction:
    pattern: str
    false_result: str
    true_result: str
    expression: Expression


@dataclasses.dataclass
class ResolveAction:
    location: complex
    expression: Expression


@dataclasses.dataclass
class LetAction:
    name: str
    expression: Expression

    def try_assign(self, state: dict[str, bool | int]) -> bool:
        new_val = self.expression.try_eval(state)
        if new_val is None:
            return False
        old_val = state.setdefault(self.name, new_val)
        assert old_val == new_val
        return True


@dataclasses.dataclass
class FeedbackAction:
    target: Expression
    condition: Expression
    action: Literal['X', 'Y', 'Z', 'S', 'H', 'SQRT_X', 'SQRT_X_DAG']


def parse_single_op_expression(line: str, variables: list[str]) -> Expression:
    seen_ops = set()
    var_line = ''.join(variables)
    for candidate in ['+', '^', '<', '&', '&&', '^^']:
        if candidate in var_line:
            seen_ops.add(candidate)
    if '&&' in seen_ops:
        seen_ops.remove('&')
    if '^^' in seen_ops:
        seen_ops.remove('^')
    if len(seen_ops) > 1:
        raise NotImplementedError(f'{line=}')
    if not seen_ops:
        assert len(variables) == 1
        return Expression(tuple(variables), op='')

    op: Literal['+', '^', '<', '&', '&&', '^^'] = cast(Any, next(iter(seen_ops)))
    variables = [e for e in ' '.join(variables).replace(op, ' ' + op + ' ').split(' ') if e]
    assert all(e == op for e in variables[1::2])
    assert len(variables) % 2 == 1
    return Expression(variables=tuple(variables[0::2]), op=op)


class LatticeSurgeryLayerWithFeedback:
    def __init__(
            self,
            *,
            layer_diagram: str,
            measure_actions: Iterable[MeasureAction],
            outputs: Iterable[str],
            checks: Iterable[DiscardShotAction],
            replace_actions: Iterable[ReplaceAction],
            resolve_actions: Iterable[ResolveAction],
            let_actions: Iterable[LetAction],
            feedback_actions: Iterable[FeedbackAction],
            print_actions: Iterable[PrintAction],
    ):
        self.layer_diagram: str = layer_diagram
        self.measure_actions: tuple[MeasureAction, ...] = tuple(measure_actions)
        self.replace_actions: tuple[ReplaceAction, ...] = tuple(replace_actions)
        self.resolve_actions: tuple[ResolveAction, ...] = tuple(resolve_actions)
        self.let_actions: tuple[LetAction, ...] = tuple(let_actions)
        self.feedback_actions: tuple[FeedbackAction, ...] = tuple(feedback_actions)
        self.outputs: tuple[str, ...] = tuple(outputs)
        self.checks: tuple[DiscardShotAction, ...] = tuple(checks)
        self.print_actions: tuple[PrintAction, ...] = tuple(print_actions)

        self._fixed_layer: LatticeSurgeryLayer | None = None

    def run(self, sim: VecSim, state: dict[str, bool], *, injected_errors: frozenset[InjectedError]):
        layer = self.make_layer(state)
        ground = any(m.ground for m in self.measure_actions)
        layer_results = sim.do_lattice_surgery_layer(layer, prefer_result=False if ground else None, injected_errors=injected_errors)

        # Record named measurement results.
        if len(layer_results) != len(self.measure_actions):
            raise ValueError(f'The number of "measures" annotations differs from the number of measurements.\n'
                             f'measurements={[e.name for e in self.measure_actions]}.\n'
                             f'{len(layer_results)=} != {len(self.measure_actions)=}')
        for k, v in zip(self.measure_actions, layer_results):
            state[k.name] = bool(v)

        # Assign variables.
        for action in self.let_actions:
            worked = action.try_assign(state)
            assert worked, action

        # Run feedback expressions.
        for action in self.feedback_actions:
            target = action.target.eval(state)
            condition = action.condition.eval(state)
            if condition:
                if action.action in 'XYZ':
                    sim.do_paulis({target: action.action})
                elif action.action == 'S':
                    sim.do_s(target)
                elif action.action == 'H':
                    sim.do_s(target)
                elif action.action == 'SQRT_X':
                    sim.do_h(target)
                    sim.do_s(target)
                    sim.do_h(target)
                elif action.action == 'SQRT_X_DAG':
                    sim.do_h(target)
                    sim.do_s_dag(target)
                    sim.do_h(target)
                elif action.action == 'T':
                    sim.do_t(target)
                else:
                    raise NotImplementedError(f'{action}')

        # Print values.
        for print_action in self.print_actions:
            print(f'{print_action.condition}={print_action.condition.eval(state)}')

        # Check checks.
        for check in self.checks:
            if check.condition.eval(state):
                return 'reject'

        # Check outputs.
        for out in self.outputs:
            if out.upper() == 'CCZ':
                assert len(sim.q2i) == 3
                a, b, c = sim.q2i
                sim.do_ccz(a, b, c)
                ma = sim.do_mx_discard(a)
                mb = sim.do_mx_discard(b)
                mc = sim.do_mx_discard(c)
                if ma or mb or mc:
                    return 'fail'
            elif out.upper() == 'TT':
                assert len(sim.q2i) == 2
                a, b = sim.q2i
                sim.do_t(a)
                sim.do_t(b)
                ma = sim.do_mx_discard(a)
                mb = sim.do_mx_discard(b)
                if ma or mb:
                    return 'fail'
            elif out == 'print':
                qs = sorted(sim.q2i.keys(), key=lambda e: (e.real, e.imag))
                print()
                print()
                print(sim.state_str(order=qs.index))
                print()
                print()
                assert False, '\n\n' + sim.state_str(order=qs.index) + '\n\n'
            else:
                raise NotImplementedError(f'{out=}')

        return 'pass'

    def make_layer(self, measurements: dict[str, bool], *, skip_resolving: bool = False) -> LatticeSurgeryLayer:
        if self._fixed_layer is not None:
            return self._fixed_layer

        c = self.layer_diagram
        for r in self.let_actions:
            r.try_assign(measurements)

        for r in self.replace_actions:
            val = r.expression.eval(measurements) & 1
            c = c.replace(r.pattern, (r.true_result if val else r.false_result).replace('_', ' '))

        result = LatticeSurgeryLayer.from_text(c)

        if not skip_resolving:
            for r in self.resolve_actions:
                v = r.expression.eval(measurements)
                c = result.nodes[r.location]
                assert len(c) == 2
                result.nodes[r.location] = cast(Any, c[v & 1])

        if not self.replace_actions and not self.resolve_actions:
            self._fixed_layer = result
        return result

    @staticmethod
    def from_content(content: str) -> 'LatticeSurgeryLayerWithFeedback':
        measure_actions: list[MeasureAction] = []
        outputs: list[str] = []
        checks: list[DiscardShotAction] = []
        prints: list[PrintAction] = []
        replace_actions: list[ReplaceAction] = []
        resolve_actions: list[ResolveAction] = []
        let_actions: list[LetAction] = []
        feedback_actions: list[FeedbackAction] = []

        kept_lines = []
        for line in content.splitlines():
            words = [e for e in line.split(' ') if e]
            if not words:
                words.append('')
            if words[0] == 'measures':
                assert len(words) == 2, words
                measure_actions.append(MeasureAction(name=words[1], ground=False))
            elif words[0] == 'ground_measures':
                assert len(words) == 2
                measure_actions.append(MeasureAction(name=words[1], ground=True))
            elif words[0] == 'resolve_at':
                _resolve_at, location, _with, *variables = words
                assert _resolve_at == 'resolve_at'
                assert _with == 'with'
                expression = parse_single_op_expression(line, variables)
                resolve_actions.append(ResolveAction(location=complex(location), expression=expression))
            elif words[0] == 'replace':
                _replace, pattern, _with, true_result, _else, false_result, _if, *variables = words
                assert _replace == 'replace'
                assert _with == 'with'
                assert _else == 'else'
                assert _if == 'if'
                expression = parse_single_op_expression(line, variables)
                replace_actions.append(ReplaceAction(pattern=pattern, false_result=false_result, true_result=true_result, expression=expression))
            elif words[0] == 'let':
                _let, name, _eq, *variables = words
                assert _let == 'let'
                assert _eq == '='
                expression = parse_single_op_expression(line, variables)
                let_actions.append(LetAction(name=name, expression=expression))
            elif words[0] == 'feedback':
                variable: str | None
                if len(words) == 4:
                    _feedback, action, _at, location = words
                    assert _feedback == 'feedback'
                    assert _at == 'at'
                    condition = Expression(('1',), '')
                else:
                    _feedback, action, _at, location, _if, *variables = words
                    assert _feedback == 'feedback'
                    assert _at == 'at'
                    assert _if == 'if'
                    condition = parse_single_op_expression(line, variables)
                assert action in ['X', 'Y', 'Z', 'S', 'H', 'SQRT_X', 'SQRT_X_DAG', 'T']
                feedback_actions.append(FeedbackAction(
                    condition=condition,
                    target=parse_single_op_expression(line, [location]),
                    action=cast(Any, action),
                ))
            elif words[0] == 'discard_shot_if':
                _check, *variables = words
                condition = parse_single_op_expression(line, variables)
                checks.append(DiscardShotAction(condition))
            elif words[0] == 'print':
                _check, *variables = words
                assert _check == 'print'
                condition = parse_single_op_expression(line, variables)
                prints.append(PrintAction(condition))
            elif words[0] == 'output_should_be':
                assert len(words) == 2
                outputs.append(words[1])
            else:
                kept_lines.append(line)

        return LatticeSurgeryLayerWithFeedback(
            layer_diagram='\n'.join(kept_lines),
            measure_actions=measure_actions,
            outputs=outputs,
            checks=checks,
            replace_actions=replace_actions,
            let_actions=let_actions,
            feedback_actions=feedback_actions,
            print_actions=prints,
            resolve_actions=resolve_actions,
        )
