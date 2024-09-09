import dataclasses
from typing import Iterable, Callable

import stim

from ._flow import Flow


@dataclasses.dataclass
class _FlowRow:
    inp: stim.PauliString
    meas: set[int]
    out: stim.PauliString
    key: int | None


def solve_flow_auto_measurements(
    *, flows: Iterable[Flow], circuit: stim.Circuit, q2i: dict[complex, int]
) -> tuple[Flow, ...]:
    flows = list(flows)
    if all(flow.measurement_indices != "auto" for flow in flows):
        # Skip solving for the generators, when it's not needed.
        return tuple(flows)

    # Create a table of the circuit's stabilizer flow generators.
    table: list[_FlowRow] = []
    num_qubits = circuit.num_qubits
    for flow in circuit.flow_generators():
        inp = flow.input_copy()
        out = flow.output_copy()
        if len(inp) == 0:
            inp = stim.PauliString(num_qubits)
        if len(out) == 0:
            out = stim.PauliString(num_qubits)
        table.append(
            _FlowRow(
                inp=inp,
                meas=set(flow.measurements_copy()),
                out=out,
                key=None,
            )
        )

    # Append flows-to-be-solved to the end of the table.
    for k in range(len(flows)):
        if flows[k].measurement_indices == "auto":
            inp = stim.PauliString(num_qubits)
            for q, p in flows[k].start.qubits.items():
                inp[q2i[q]] = p
            out = stim.PauliString(num_qubits)
            for q, p in flows[k].end.qubits.items():
                out[q2i[q]] = p
            table.append(_FlowRow(inp=inp, meas=set(), out=out, key=k))

    def partial_elim(predicate: Callable[[_FlowRow], bool]):
        nonlocal num_solved
        for k2 in range(num_solved, len(table)):
            if predicate(table[k2]):
                pivot = k2
                break
        else:
            return

        for k2 in range(len(table)):
            if k2 != pivot and predicate(table[k2]):
                table[k2].inp *= table[pivot].inp
                table[k2].meas ^= table[pivot].meas
                table[k2].out *= table[pivot].out
        t0 = table[pivot]
        t1 = table[num_solved]
        table[pivot] = t1
        table[num_solved] = t0
        num_solved += 1

    # Perform Gaussian elimination on the table.
    num_solved = 0
    for q in range(num_qubits):
        partial_elim(lambda f: f.inp[q] & 1)
        partial_elim(lambda f: f.inp[q] & 2)
    for q in range(num_qubits):
        partial_elim(lambda f: f.out[q] & 1)
        partial_elim(lambda f: f.out[q] & 2)

    # Find the flow-to-be-solved rows in the table; now solved.
    for t in table:
        if t.key is not None:
            if t.inp.weight > 0 or t.out.weight > 0:
                raise ValueError(f"Failed to solve {flows[t.key]}")
            flows[t.key] = flows[t.key].with_edits(measurement_indices=t.meas)

    return tuple(flows)
