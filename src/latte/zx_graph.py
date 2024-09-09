import dataclasses
import functools
from typing import Literal, Any, cast, Iterable

import networkx as nx
import stim

import gen

TNode = Any


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class ZXNode:
    key: TNode
    index: int
    kind: Literal['X', 'Z', 'in', 'out', 'H']
    phase: complex


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class ZXEdge:
    n1: int
    n2: int
    key: Any
    col_index: int
    x_sink: bool
    z_sink: bool


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class FeedbackTargets:
    # Bit flip targets.
    xs: frozenset[Any] = frozenset()
    # Phase flip targets.
    zs: frozenset[Any] = frozenset()
    # Measurement flip targets.
    ms: frozenset[int] = frozenset()


@dataclasses.dataclass(frozen=True)
class ExternalStabilizer:
    paulis: stim.PauliString
    sign: int
    inp: gen.PauliMap
    out: gen.PauliMap
    interior: gen.PauliMap
    is_measurement: bool
    sink_edge_cols: tuple[int, ...]
    sink_edge_keys: tuple[complex, ...]
    sink_edge_bases: tuple[Literal['X', 'Z'], ...]
    port_stabilizer: stim.PauliString


class ZXGraph:
    def __init__(
            self,
            *,
            nodes: tuple[ZXNode, ...],
            edges: tuple[ZXEdge, ...],
            num_locations: int,
    ):
        self.nodes = nodes
        self.edges = edges
        self.num_locations = num_locations

    def to_lattice_surgery_error_table(self) -> list[stim.PauliString]:
        """Returns possible distance-d lattice surgery errors as pauli strings.

        Local errors include pipe errors (X or Z error chains crossing an edge)
        as well as junction errors (X or Z errors a junction potentially at a
        diagonal, or piercing the center of the junction).

        The indexing used by the pauli strings comes from `self.nn2e`.
        """
        errors = []
        for node in self.nodes:
            if (node.kind == 'X' or node.kind == 'Z') and node.phase in [1, -1]:
                same_kind = node.kind
                other_kind = 'X' if node.kind == 'Z' else 'Z'

                # Potential errors crossing the junction diagonally.
                neighbors = self.n2neighbors[node.key]
                for k1 in range(len(neighbors)):
                    for k2 in range(k1 + 1, len(neighbors)):
                        n1 = neighbors[k1]
                        n2 = neighbors[k2]
                        i1 = self.nn2e[(node.key, n1.key)].col_index
                        i2 = self.nn2e[(node.key, n2.key)].col_index
                        err = stim.PauliString(self.num_locations)
                        err[i1] = other_kind
                        err[i2] = other_kind
                        errors.append(err)

                # The error piercing the center of the junction.
                err = stim.PauliString(self.num_locations)
                err[node.index] = same_kind
                errors.append(err)
            elif (node.kind == 'X' or node.kind == 'Z') and node.phase in [1j, -1j]:
                pass
            elif node.kind == 'in' or node.kind == 'out':
                # Edge errors on the threshold of the outside world.
                for basis in 'XZ':
                    err = stim.PauliString(self.num_locations)
                    err[node.index] = basis
                    errors.append(err)
            elif node.kind == 'H':
                pass
            else:
                raise NotImplementedError(f'{node=}')

        # Errors crossing from one boundary to another during an edge.
        for edge in self.edges:
            for basis in 'XZ':
                err = stim.PauliString(self.num_locations)
                err[edge.col_index] = basis
                errors.append(err)

        return errors

    def to_stabilizer_flow_table(self, *, include_edges_not_centers: bool) -> list[stim.PauliString]:
        table = []
        for node in self.nodes:
            if node.kind == 'H':
                n1, n2 = self.n2neighbors[node.key]
                i1 = self.nn2e[(node.key, n1.key)].col_index
                i2 = self.nn2e[(node.key, n2.key)].col_index
                crossing = stim.PauliString(self.num_locations)
                crossing[i1] = 'X'
                crossing[i2] = 'Z'
                table.append(crossing)
                crossing = stim.PauliString(self.num_locations)
                crossing[i1] = 'Z'
                crossing[i2] = 'X'
                table.append(crossing)
            elif node.kind == 'X' or node.kind == 'Z':
                if node.phase.imag:
                    neighbor, = self.n2neighbors[node.key]

                    reach = stim.PauliString(self.num_locations)
                    i = self.nn2e[(node.key, neighbor.key)].col_index
                    if not include_edges_not_centers:
                        reach[node.index] = 'Y'
                    reach[i] = 'Y'
                    table.append(reach)
                else:
                    same_kind = node.kind
                    other_kind = 'X' if node.kind == 'Z' else 'Z'
                    neighbors = self.n2neighbors[node.key]
                    for k in range(len(neighbors) - 1):
                        n1 = neighbors[k]
                        n2 = neighbors[k + 1]
                        i1 = self.nn2e[(node.key, n1.key)].col_index
                        i2 = self.nn2e[(node.key, n2.key)].col_index
                        crossing = stim.PauliString(self.num_locations)
                        crossing[i1] = same_kind
                        crossing[i2] = same_kind
                        table.append(crossing)

                    broadcast = stim.PauliString(self.num_locations)
                    if not include_edges_not_centers:
                        broadcast[node.index] = other_kind
                    for neighbor in neighbors:
                        i = self.nn2e[(node.key, neighbor.key)].col_index
                        broadcast[i] = other_kind
                    if node.phase == 1:
                        broadcast.sign = +1
                    elif node.phase == -1:
                        broadcast.sign = -1
                    else:
                        raise NotImplementedError(f'{node=}')
                    table.append(broadcast)
            elif node.kind == 'in' or node.kind == 'out':
                neighbor, = self.n2neighbors[node.key]
                i = self.nn2e[(node.key, neighbor.key)].col_index
                for basis in 'XZ':
                    enter = stim.PauliString(self.num_locations)
                    enter[node.index] = basis
                    enter[i] = basis
                    table.append(enter)
            else:
                raise NotImplementedError(f'{node=}')

        if include_edges_not_centers:
            for node in self.nodes:
                if node.kind == 'X' or node.kind == 'Z':
                    crossing = stim.PauliString(self.num_locations)
                    crossing[node.index] = node.kind
                    crossing[self.nn2e[(node.key, self.n2neighbors[node.key][0].key)].col_index] = node.kind
                    table.append(crossing)

            for n1, n2 in self.internal_edges + self.input_edges + self.output_edges:
                c1, c2 = self.nn2ii[(n1, n2)]
                if c1 > c2:
                    continue
                for basis in 'XZ':
                    edge = stim.PauliString(self.num_locations)
                    edge[c1] = basis
                    edge[c2] = basis
                    table.append(edge)

        return table

    @functools.cached_property
    def n2neighbors(self) -> dict[TNode, tuple[ZXNode, ...]]:
        result = {n.key: [] for n in self.nodes}
        for edge in self.edges:
            result[self.nodes[edge.n1].key].append(self.nodes[edge.n2])
        return result

    @functools.cached_property
    def input_edges(self) -> tuple[tuple[TNode, TNode], ...]:
        result = []
        for edge in self.edges:
            n1 = self.nodes[edge.n1]
            n2 = self.nodes[edge.n2]
            t1 = n1.kind
            t2 = n2.kind
            if t1 == 'in' or t2 == 'in':
                result.append((n1.key, n2.key))
        return tuple(result)

    @functools.cached_property
    def output_edges(self) -> tuple[tuple[TNode, TNode], ...]:
        result = []
        for edge in self.edges:
            n1 = self.nodes[edge.n1]
            n2 = self.nodes[edge.n2]
            t1 = n1.kind
            t2 = n2.kind
            if t1 == 'out' or t2 == 'out':
                result.append((n1.key, n2.key))
        return tuple(result)

    @functools.cached_property
    def internal_edges(self) -> tuple[tuple[TNode, TNode], ...]:
        result = []
        for edge in self.edges:
            n1 = self.nodes[edge.n1]
            n2 = self.nodes[edge.n2]
            t1 = n1.kind
            t2 = n2.kind
            if t1 == 'in' or t2 == 'in' or t1 == 'out' or t2 == 'out':
                continue
            result.append((n1.key, n2.key))
        return tuple(result)

    @functools.cached_property
    def internal_edge_set(self) -> frozenset[tuple[TNode, TNode]]:
        return frozenset(
            edge[::order]
            for order in [-1, +1]
            for edge in self.internal_edges
        )

    @functools.cached_property
    def input_edge_set(self) -> frozenset[tuple[TNode, TNode]]:
        return frozenset(
            edge[::order]
            for order in [-1, +1]
            for edge in self.input_edges
        )

    @functools.cached_property
    def output_edge_set(self) -> frozenset[tuple[TNode, TNode]]:
        return frozenset(
            edge[::order]
            for order in [-1, +1]
            for edge in self.output_edges
        )

    @functools.cached_property
    def nn2e(self) -> dict[tuple[TNode, TNode], ZXEdge]:
        nn2e = {}
        for edge in self.edges:
            nn2e[(self.nodes[edge.n1].key, self.nodes[edge.n2].key)] = edge
        return nn2e

    @functools.cached_property
    def nn2ii(self) -> dict[tuple[TNode, TNode], tuple[int, int]]:
        e2ii: dict[tuple[TNode, TNode], tuple[int, int]] = {}
        for edge in self.edges:
            k1 = self.nodes[edge.n1].key
            k2 = self.nodes[edge.n2].key
            e2ii[(k1, k2)] = (self.nn2e[(k1, k2)].col_index, self.nn2e[(k2, k1)].col_index)
        return e2ii

    @functools.cached_property
    def i2edge(self) -> dict[int, ZXEdge]:
        return {
            edge.col_index: edge
            for edge in self.edges
        }

    @functools.cached_property
    def i2n(self) -> dict[int, ZXNode]:
        return {
            n.index: n
            for n in self.nodes
        }

    def to_stabilizer_flow_table_with_edge_differences_eliminated(self) -> list[stim.PauliString]:
        flow_table = self.to_stabilizer_flow_table(
            include_edges_not_centers=False)
        num_eliminated = eliminate_differences(
            table=flow_table,
            pairs=[self.nn2ii[e] for e in self.internal_edges + self.input_edges + self.output_edges],
        )
        result = [row for row in flow_table[num_eliminated:] if any(row)]
        eliminate_points(
            table=result,
            basis_locs=[
                (cast(Literal['X', 'Z'], b), edge.col_index)
                for edge in self.edges
                for b in ('X' * edge.x_sink + 'Z' * edge.z_sink)
            ] + [
                (cast(Literal['X', 'Z'], b), k)
                for k in range(len(self.port_col_set))
                for b in 'XZ'
            ],
            num_pivotable_rows=len(result),
        )
        return result

    @functools.cached_property
    def port_col_set(self) -> frozenset[int]:
        return frozenset(
            node.index
            for node in self.nodes
            if node.kind == 'in' or node.kind == 'out'
        )

    @functools.cached_property
    def out_port_col_set(self) -> frozenset[int]:
        return frozenset(
            node.index
            for node in self.nodes
            if node.kind == 'out'
        )

    @functools.cached_property
    def in_port_col_set(self) -> frozenset[int]:
        return frozenset(
            node.index
            for node in self.nodes
            if node.kind == 'in'
        )

    @functools.cached_property
    def error_to_feedback_map(self) -> dict[tuple[Any, Literal['X', 'Z']], FeedbackTargets]:
        """Finds ways to push errors inside the graph out of the graph.
        """
        err_rewrite_table = self.to_stabilizer_flow_table(
            include_edges_not_centers=True,
        )

        # Add logical measurement stabilizers to the identities that can be used
        # to simplify the effects of physical errors. This should guarantee that
        # all errors can be pushed to the output. Add columns at the end to keep
        # track of which logical measurements were flipped as part of moving the
        # errors.
        measure_stabilizers = self.measurement_stabilizers
        err_rewrite_table = [
            row + stim.PauliString(len(measure_stabilizers))
            for row in err_rewrite_table
        ]
        for k, m in enumerate(measure_stabilizers):
            p = stim.PauliString(len(err_rewrite_table[0]))
            for c, b in zip(m.sink_edge_cols, m.sink_edge_bases):
                p[c] = b
            p[len(p) - len(measure_stabilizers) + k] = 'X'
            err_rewrite_table.append(p)

        # Rewrite errors by using gaussian elimination on the rewrite table.
        num_error_rewrite_rules = len(err_rewrite_table)
        errs = self.to_lattice_surgery_error_table()
        for err in errs:
            err_rewrite_table.append(err + stim.PauliString(len(measure_stabilizers)))
        eliminate_points(
            table=err_rewrite_table,
            basis_locs=[
                (cast(Literal['X', 'Z'], b), k)
                for k in [
                    *(set(range(self.num_locations)) - self.out_port_col_set - self.in_port_col_set),
                    *self.in_port_col_set,
                    *self.out_port_col_set,
                ]
                for b in 'XZ'
            ],
            num_pivotable_rows=num_error_rewrite_rules,
        )
        rewritten_errors = err_rewrite_table[num_error_rewrite_rules:]

        result = {}
        num_out = len(self.out_port_col_set)
        for err, new_err in zip(errs, rewritten_errors, strict=True):
            p = gen.PauliMap(err)
            if len(p.qubits) != 1:
                continue
            qubit_feedback = new_err[:num_out]
            measure_feedback = new_err[len(new_err) - len(measure_stabilizers):]
            if any(new_err[num_out:len(new_err) - len(measure_stabilizers)]):
                raise ValueError("Leftover")
            measure_indices = frozenset(k for k, p in enumerate(measure_feedback) if p)
            qubit_keys_x = frozenset(self.i2n[k].key for k, p in enumerate(qubit_feedback) if p in [1, 2])
            qubit_keys_z = frozenset(self.i2n[k].key for k, p in enumerate(qubit_feedback) if p in [2, 3])
            (q, b), = p.qubits.items()
            if q in self.i2n:
                q = self.i2n[q].key
            elif q in self.i2edge:
                edge = self.i2edge[q]
                q = (self.nodes[edge.n1].key, self.nodes[edge.n2].key)
            else:
                raise NotImplementedError()
            result[(q, b)] = FeedbackTargets(xs=qubit_keys_x, zs=qubit_keys_z, ms=measure_indices)
        return result

    @functools.cached_property
    def external_stabilizers(self) -> tuple[ExternalStabilizer, ...]:
        """Returns generators for the graph's input-to-output relationships."""

        flow_table = self.to_stabilizer_flow_table_with_edge_differences_eliminated()

        result: list[ExternalStabilizer] = []
        for row in flow_table:
            sign = +1
            for e in self.edges:
                if e.n1 > e.n2:
                    continue
                col1, col2 = self.nn2ii[(self.nodes[e.n1].key, self.nodes[e.n2].key)]
                p1 = row[col1]
                p2 = row[col2]
                assert p1 == p2
                if p1 == p2 == 2:
                    # Each YY edge causes a -1
                    sign *= -1

            row = row * sign

            inp: dict[complex, Literal['X', 'Y', 'Z']] = {}
            out: dict[complex, Literal['X', 'Y', 'Z']] = {}
            interior: dict[complex, Literal['X', 'Y', 'Z']] = {}
            sink_edges = []
            sink_bases: list[Literal['X', 'Z']] = []
            for col, p in enumerate(row):
                if p == 0:
                    continue
                if col in self.i2edge:
                    edge: ZXEdge = self.i2edge[col]
                    a = self.nodes[edge.n1].key
                    b = self.nodes[edge.n2].key
                    if isinstance(a, tuple):
                        a = a[0]
                    if isinstance(b, tuple):
                        b = b[0]
                    c = (a + b) / 2
                    interior[c] = cast(Literal['X', 'Y', 'Z'], '_XYZ'[p])
                    if edge.x_sink and p != 1:
                        sink_edges.append(edge)
                        sink_bases.append('X')
                    if edge.z_sink and p != 3:
                        sink_edges.append(edge)
                        sink_bases.append('Z')
                elif col in self.i2n:
                    n = self.i2n[col]
                    k = n.key
                    if isinstance(k, tuple):
                        k = k[0]
                    if n.kind == 'in':
                        inp[k] = cast(Literal['X', 'Y', 'Z'], '_XYZ'[p])
                    elif n.kind == 'out':
                        out[k] = cast(Literal['X', 'Y', 'Z'], '_XYZ'[p])
                    else:
                        interior[k] = cast(Literal['X', 'Y', 'Z'], '_XYZ'[p])
                else:
                    raise NotImplementedError(f'{col}')

            if not out:
                if len(sink_edges) == 0:
                    raise ValueError(f"A potential excitation can't escape.\n"
                                     f"It's trapped along {interior!r}.\n"
                                     f"Change the graph or annotate a measurement with '*' or '@' along an edge.")

            result.append(ExternalStabilizer(
                paulis=row,
                sign=int((sign * row.sign).real),
                inp=gen.PauliMap(inp),
                out=gen.PauliMap(out),
                interior=gen.PauliMap(interior),
                is_measurement=not bool(out),
                sink_edge_cols=tuple(e.col_index for e in sink_edges),
                sink_edge_keys=tuple(e.key for e in sink_edges),
                sink_edge_bases=tuple(sink_bases),
                port_stabilizer=stim.PauliString(0),
            ))

        result = sorted(
            result,
            key=lambda x: (-1, -1) if x.sink_edge_keys is None else (sum(x.sink_edge_keys).real, sum(x.sink_edge_keys).imag))

        num_ports = len(self.port_col_set)
        assert len(result) == num_ports
        port_stabilizers = [e.paulis[:num_ports] for e in result]
        for k in range(num_ports):
            v = result[k].port_stabilizer
            v += port_stabilizers[k]

        return tuple(result)

    @functools.cached_property
    def measurement_stabilizers(self) -> tuple[ExternalStabilizer, ...]:
        port_relations = self.external_stabilizers
        num_out_ports = len(self.out_port_col_set)
        result = tuple(
            e
            for e in port_relations
            if not any(e.port_stabilizer[:num_out_ports])
        )
        used_sinks = {
            (k, b)
            for m in result
            for k, b in zip(m.sink_edge_keys, m.sink_edge_bases)
        }
        available_sinks = {
            (edge.key, 'X')
            for edge in self.edges
            if edge.x_sink
        } | {
            (edge.key, 'Z')
            for edge in self.edges
            if edge.z_sink
        }
        if used_sinks != available_sinks:
            raise ValueError("There's an annotated measurement that isn't needed.")
        assert used_sinks == available_sinks
        return result

    @staticmethod
    def from_nx_graph(graph: nx.Graph) -> 'ZXGraph':
        def k2c(v: Any) -> complex:
            if isinstance(v, tuple):
                return v[0]
            return v

        nodes = []
        next_index = 0
        n2i = {}
        order = {'S_port': -2, 'T_port': -2, 'in_port': -1, 'out_port': -3}
        for n, vals in sorted(graph.nodes.items(), key=lambda e: order.get(e[1]['type'], 0)):
            t = vals['type']
            phase = vals['phase']
            kind: Literal['X', 'Z', 'in', 'out', 'H']
            if t == 'X':
                kind = 'X'
            elif t == 'Z':
                kind = 'Z'
            elif t == 'in_port' or t == 'T_port' or t == 'S_port':
                kind = 'in'
            elif t == 'out_port':
                kind = 'out'
            elif t == 'H':
                kind = 'H'
            else:
                raise NotImplementedError(f'{n=}, {vals=}')
            nodes.append(ZXNode(
                key=n,
                index=next_index,
                kind=kind,
                phase=phase,
            ))
            n2i[n] = next_index
            next_index += 1

        edges = []
        for (n1, n2), vals in graph.edges.items():
            for orientation in range(2):
                edges.append(ZXEdge(
                    n1=n2i[n1],
                    n2=n2i[n2],
                    col_index=next_index,
                    x_sink=orientation == 0 and bool(vals.get('x_sink')),
                    z_sink=orientation == 0 and bool(vals.get('z_sink')),
                    key=(k2c(n1) + k2c(n2)) / 2,
                ))
                next_index += 1
                n1, n2 = n2, n1

        return ZXGraph(
            nodes=tuple(nodes),
            edges=tuple(edges),
            num_locations=next_index,
        )


def eliminate_points(*, table: list[stim.PauliString], basis_locs: Iterable[tuple[Literal['X', 'Z'], int]], num_pivotable_rows: int) -> int:
    def pxz(p: int) -> int:
        p ^= p >> 1
        return p

    num_solved = 0
    for basis, col in basis_locs:
        m = 1 if basis == 'X' else 2
        for pivot_row in range(num_solved, num_pivotable_rows):
            if pxz(table[pivot_row][col]) & m:
                break
        else:
            continue
        if pivot_row != num_solved:
            table[pivot_row], table[num_solved] = table[num_solved], table[pivot_row]
            pivot_row = num_solved
        for row in range(len(table)):
            if row != pivot_row and pxz(table[row][col]) & m:
                table[row] *= table[pivot_row]
        num_solved += 1

    return num_solved


def eliminate_differences(*, table: list[stim.PauliString], pairs: Iterable[tuple[int, int]]) -> int:
    def pauli_diff_ixzy(p1: int, p2: int) -> int:
        dp = p1 ^ p2
        dp ^= dp >> 1
        return dp

    num_solved = 0
    for col1, col2 in pairs:
        for m in [1, 2]:
            for pivot_row in range(num_solved, len(table)):
                if pauli_diff_ixzy(table[pivot_row][col1], table[pivot_row][col2]) & m:
                    break
            else:
                continue
            if pivot_row != num_solved:
                table[pivot_row], table[num_solved] = table[num_solved], table[pivot_row]
                pivot_row = num_solved
            for row in range(len(table)):
                if row != pivot_row and pauli_diff_ixzy(table[row][col1], table[row][col2]) & m:
                    table[row] *= table[pivot_row]
            num_solved += 1

    return num_solved
