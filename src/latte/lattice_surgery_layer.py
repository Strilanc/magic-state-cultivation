import collections
import dataclasses
import functools
import random
from typing import Literal, Any, cast, Iterable, TYPE_CHECKING, Callable
from typing import Sequence

import numpy as np
import stim

import gen
from latte.lattice_surgery_instruction import LatticeSurgeryInstruction, ErrorSource
from latte.zx_graph import ZXGraph

if TYPE_CHECKING:
    import pygltflib
    import networkx as nx


@dataclasses.dataclass(frozen=True, unsafe_hash=True)
class InjectedError:
    pos: complex
    layer: float
    basis: Literal['X', 'Y', 'Z']

    def time_shifted_by(self, dt: float) -> 'InjectedError':
        return InjectedError(pos=self.pos, layer=self.layer + dt, basis=self.basis)


class LatticeSurgeryLayer:
    def __init__(
            self,
            *,
            nodes: dict[complex, Literal['.', 'X', 'Y', 'Z', 'T', 'S', 'XY', 'YZ', 'XZ']],
            future_edges: Iterable[complex],
            past_edges: Iterable[complex],
            edges: Iterable[complex],
            x_sink_edges: Iterable[complex],
            z_sink_edges: Iterable[complex],
    ):
        """A layer of lattice surgery, with spacelike and timelike edges.

        Args:
            nodes: The locations and types of junctions. The locations should
                be integer complex coordinates. The types are:

                '.': Unused location. Surface code not present here.
                'X': X-type junction. Reveals XX for each edge pair, and Z_all.
                'Z': Z-type junction. Reveals ZZ for each edge pair, and X_all.
                'Y': Y-type leaf, using the construction from
                    https://arxiv.org/abs/2302.07395 . This node must only have
                    timelike edges. If a past edge is present, this is a Y
                    basis measurement. If a future edge is present, this is a Y
                    basis reset.
                'T': T state injection. Randomly prepares T|+> or T†|+>, with
                    the outcome stored in the measurement record.
                'S': S state injection. Randomly prepares |i> or |j>, with
                    the outcome stored in the measurement record. Differs from
                    'Y' in that 'Y' is fault tolerant and considered
                    deterministic.
            future_edges: The set of the nodes that will end up as logical
                qubits sticking around (propagating into the next layer).
            past_edges: The set of the nodes that entered as logical
                qubits propagating from the previous layer.
            edges: The set of spacelike edges between nodes. Each entry should
                be a complex number with one integer coefficient and one half
                integer coefficient. If c is a complex number with integer
                coordinates, then c+0.5 being in this set means there is an
                edge between c and c+1. Similarly, c+0.5j being in this set
                means there is an edge between c and c+1j.
            x_sink_edges: Locations where trapped Pauli X excitations can
                be removed, creating a logical measurement result.
            z_sink_edges: Locations where trapped Pauli Z excitations can
                be removed, creating a logical measurement result.
        """
        self.nodes: dict[complex, Literal['.', 'X', 'Y', 'Z', 'T', 'S', 'XY', 'YZ', 'XZ']] = nodes
        self.future_edges: frozenset[complex] = frozenset(future_edges)
        self.past_edges: frozenset[complex] = frozenset(past_edges)
        self.edges: frozenset[complex] = frozenset(edges)
        self.x_sink_edges: frozenset[complex] = frozenset(x_sink_edges)
        self.z_sink_edges: frozenset[complex] = frozenset(z_sink_edges)

        self._cached_tasks: tuple[LatticeSurgeryInstruction, ...] | None = None
        self._cached_tasks_key = None

    def list_edge_errors(self) -> list[InjectedError]:
        result = []
        for v in self.past_edges:
            for b in 'XYZ':
                result.append(InjectedError(pos=v, layer=-0.5, basis=cast(Any, b)))
        for v in self.future_edges:
            for b in 'XYZ':
                result.append(InjectedError(pos=v, layer=+0.5, basis=cast(Any, b)))
        for e in self.edges:
            for b in 'XYZ':
                result.append(InjectedError(pos=e, layer=0, basis=cast(Any, b)))
        return result

    def to_stim_circuit(self) -> stim.Circuit:
        tasks = self.to_sim_instructions(layer_key='A')
        result = stim.Circuit()
        q2i = {}
        m2i = {}
        group2i = {}

        def key_to_records(k: Any) -> list[stim.GateTarget]:
            if k in group2i:
                ms = group2i[k]
            elif k in m2i:
                ms = [k]
            else:
                raise ValueError(f'{k=}')
            return [stim.target_rec(m2i[m] - len(m2i)) for m in ms]

        coords = {}
        for n in self.nodes:
            q = len(q2i)
            q2i[n] = q
            coords[n] = (n.real, n.imag)

        t_measure_waiting_for_x_herald = None
        obs_index = 0
        cur_bit = []
        for task in tasks:
            if t_measure_waiting_for_x_herald is not None and 'herald' not in task.action:
                t_measure_waiting_for_x_herald = None
            a = task.target
            b = task.target2
            c = task.action
            if c == 'mxx':
                result.append('MXX', [q2i[a], q2i[b]])
                m2i[task.measure_key] = len(m2i)
            elif c == 'mzz':
                result.append('MZZ', [q2i[a], q2i[b]])
                m2i[task.measure_key] = len(m2i)
            elif c == 'cx':
                result.append('CX', [q2i[a], q2i[b]])
            elif c == 'qalloc_x':
                result.append('RX', [q2i[a]])
            elif c == 'qalloc_y':
                result.append('RY', [q2i[a]])
            elif c == 'qalloc_z':
                result.append('RZ', [q2i[a]])
            elif c == 'm_discard_x':
                result.append('MRX', [q2i[a]])
                m2i[task.measure_key] = len(m2i)
            elif c == 'm_discard_y':
                result.append('MRY', [q2i[a]])
                m2i[task.measure_key] = len(m2i)
            elif c == 'm_discard_z':
                result.append('MRZ', [q2i[a]])
                m2i[task.measure_key] = len(m2i)
            elif c == 't':
                q = ('T_INJECT', len(q2i))
                q2i[q] = len(q2i)
                coords[q] = (733, 57473, len(q2i))
                result.append('CX', [q2i[a], q2i[q]])
                result.append('MR', [q2i[q]])
                m2i[q] = len(m2i)
                t_measure_waiting_for_x_herald = q
            elif c == 'heralded_random_x':
                if t_measure_waiting_for_x_herald is not None:
                    m2i[task.measure_key] = m2i[t_measure_waiting_for_x_herald]
                    del m2i[t_measure_waiting_for_x_herald]
                    t_measure_waiting_for_x_herald = None
                else:
                    result.append('X_ERROR', [q2i[a]], 1)
                    result.append('MPAD', [1])
                    m2i[task.measure_key] = len(m2i)
                    # result.append('HERALDED_PAULI_CHANNEL_1', [q2i[a]], [0, 0.5, 0, 0])
            elif c == 'heralded_random_z':
                result.append('Z_ERROR', [q2i[a]], 0)
                result.append('MPAD', [0])
                # result.append('HERALDED_PAULI_CHANNEL_1', [q2i[a]], [0, 0, 0, 0.5])
                m2i[task.measure_key] = len(m2i)
            elif c == 'accumulator_bit_clear':
                cur_bit.clear()
            elif c == 'accumulator_bit_xor':
                cur_bit.append(task.measure_key)
            elif c == 'accumulator_bit_save':
                group2i[task.measure_key] = cur_bit
                result.append('OBSERVABLE_INCLUDE', key_to_records(task.measure_key), obs_index)
                obs_index += 1
            elif c == 'feedback_m2x':
                result.append('CX', [
                    x
                    for t in key_to_records(task.measure_key)
                    for x in [t, q2i[task.target]]
                ])
            elif c == 'feedback_m2y':
                result.append('CY', [
                    x
                    for t in key_to_records(task.measure_key)
                    for x in [t, q2i[task.target]]
                ])
            elif c == 'feedback_m2z':
                result.append('CZ', [
                    x
                    for t in key_to_records(task.measure_key)
                    for x in [t, q2i[task.target]]
                ])
            else:
                raise NotImplementedError(f'{task=}')
        header = stim.Circuit()
        for q, c in coords.items():
            header.append('QUBIT_COORDS', q2i[q], c)
        return header + result

    def compute_node_degrees(self) -> collections.Counter:
        degrees = collections.Counter()
        for e in self.future_edges:
            degrees[e] += 1
        for e in self.past_edges:
            degrees[e] += 1
        for e in self.edges:
            if e.real % 1 == 0.5:
                degrees[e + 0.5] += 1
                degrees[e - 0.5] += 1
            if e.imag % 1 == 0.5:
                degrees[e + 0.5j] += 1
                degrees[e - 0.5j] += 1
        return degrees

    @staticmethod
    def solve_lattice_surgery_orientations(
            layers: Sequence['LatticeSurgeryLayer'],
            ignore_contradictions: bool = False,
    ) -> dict[tuple[complex, float], bool]:
        result = {}
        progress = True
        while progress:
            progress = False
            for k in range(len(layers)):
                progress |= layers[k]._partial_solve_lattice_surgery_orientations(k, result, ignore_contradictions=ignore_contradictions)
        return result

    def _partial_solve_lattice_surgery_orientations(
            self, layer: int, known: dict[tuple[complex, float], bool],
            *,
            ignore_contradictions: bool) -> bool:
        degrees = self.compute_node_degrees()
        new_values = {}

        for node, c in self.nodes.items():
            if degrees[node] >= 3:
                assert c == 'X' or c == 'Z'
                has_real = node + 0.5 in self.edges or node - 0.5 in self.edges
                has_imag = node + 0.5j in self.edges or node - 0.5j in self.edges
                has_time = node in self.past_edges or node in self.future_edges
                if has_real + has_time + has_imag != 2:
                    if ignore_contradictions:
                        continue
                    raise ValueError("Lattice surgery junctions must be planar.")
                for d in [-0.5, 0.5]:
                    if node + d in self.edges:
                        new_values[(node + d, layer)] = (c == 'X') ^ has_time
                for d in [-0.5j, 0.5j]:
                    if node + d in self.edges:
                        new_values[(node + d, layer)] = (c == 'X') ^ has_time
                if node in self.future_edges:
                    new_values[(node, layer + 0.5)] = (c == 'X') ^ has_imag
                if node in self.past_edges:
                    new_values[(node, layer - 0.5)] = (c == 'X') ^ has_imag
        progress = False
        for k, v in new_values.items():
            v2 = known.get(k, None)
            if v2 is None:
                known[k] = v
                progress = True
            elif v != v2 and not ignore_contradictions:
                raise ValueError("Inconsistent boundary requirements for lattice surgery.")
        new_values.clear()

        for node, c in self.nodes.items():
            if degrees[node] == 2:
                has_time = node in self.past_edges or node in self.future_edges
                has_imag = node + 0.5j in self.edges or node - 0.5j in self.edges
                has_real = node + 0.5 in self.edges or node - 0.5 in self.edges
                inverted_keys = set()
                keys = [
                    (node + d, float(layer))
                    for d in [-0.5, 0.5, -0.5j, 0.5j]
                    if node + d in self.edges
                ]
                if node in self.past_edges:
                    keys.append((node, layer - 0.5))
                if node in self.future_edges:
                    keys.append((node, layer + 0.5))
                if has_time and has_imag:
                    inverted_keys = set(keys)
                elif has_time and has_real:
                    inverted_keys = {k for k in keys if k[1] % 1 == 0.5}
                v = None
                for k in keys:
                    v = known.get(k)
                    if v is not None:
                        v ^= k in inverted_keys
                        break
                if v is not None:
                    if c == 'H':
                        v ^= True
                    for k in keys:
                        new_values[k] = v ^ (k in inverted_keys)

        for k, v in new_values.items():
            v2 = known.get(k, None)
            if v2 is None:
                known[k] = v
                progress = True
            elif v != v2 and not ignore_contradictions:
                raise ValueError("Inconsistent boundary requirements for lattice surgery.")
        new_values.clear()
        return progress

    @staticmethod
    def combined_3d_model_gltf(
            layers: Iterable['LatticeSurgeryLayer'],
            *,
            wireframe: bool = False,
            spacing: float = 2,
            ignore_contradictions: bool = False,
            injected_errors: Iterable[InjectedError],
    ) -> 'pygltflib.GLTF2':
        layers = tuple(layers)
        injected_errors = frozenset(injected_errors)
        wireframe |= bool(injected_errors)
        triangles = []
        lines = []
        if wireframe:
            for k in range(len(layers)):
                layers[k].add_wireframe_to_3d_skeleton_gltf(
                    k,
                    out_triangles=triangles,
                    out_lines=lines,
                    spacing=spacing,
                    injected_errors=frozenset(
                        err.time_shifted_by(-k)
                        for err in injected_errors
                        if err.layer == k or err.layer == k - 0.5
                    ),
                )
        else:
            orientations = LatticeSurgeryLayer.solve_lattice_surgery_orientations(layers, ignore_contradictions=ignore_contradictions)
            for k in range(len(layers)):
                layers[k].add_to_3d_model_gltf(k, orientations, out_triangles=triangles, out_lines=lines, spacing=spacing)

        return gen.gltf_model_from_colored_triangle_data(triangles, colored_line_data=lines)

    def to_3d_model_gltf(self, *, wireframe: bool = False) -> 'pygltflib.GLTF2':
        return LatticeSurgeryLayer.combined_3d_model_gltf([self], wireframe=wireframe, injected_errors=frozenset())

    def add_to_3d_model_gltf(
            self,
            layer: int,
            orientations: dict[tuple[complex, float], bool],
            out_triangles: list[gen.ColoredTriangleData],
            out_lines: list[gen.ColoredLineData],
            spacing: float,
    ):
        degrees = self.compute_node_degrees()
        pitch = spacing + 1

        def xy(v):
            return np.array([-v[2], v[0], -v[1]])
        add_face = functools.partial(add_face_to, xy=xy, out_triangles=out_triangles, out_lines=out_lines)
        add_cube = functools.partial(add_cube_to, xy=xy, out_triangles=out_triangles, out_lines=out_lines)

        for node, c in self.nodes.items():
            x = node.real * pitch
            y = node.imag * pitch
            z = layer * pitch
            if c == 'X' or c == 'Z':
                top_bottom = None
                left_right = None
                front_back = None
                for d in [0.5, -0.5, 0.5j, -0.5j]:
                    v = orientations.get((node + d, layer))
                    if v is not None:
                        top_bottom = v
                        if d.real == 0:
                            left_right = not v
                        else:
                            front_back = not v
                v = orientations.get((node, layer - 0.5))
                if v is None:
                    v = orientations.get((node, layer + 0.5))
                if v is not None:
                    front_back = v
                    left_right = not v
                if left_right is None:
                    left_right = c == 'X'
                if front_back is None:
                    front_back = c == 'X'
                if top_bottom is None:
                    top_bottom = c == 'X'
                z_scale = 1
                if degrees[node] == 1:
                    has_real = node + 0.5 in self.edges or node - 0.5 in self.edges
                    has_imag = node + 0.5j in self.edges or node - 0.5j in self.edges
                    has_time = node in self.past_edges or node in self.future_edges
                    if has_real:
                        left_right = c != 'X'
                    if has_imag:
                        front_back = c != 'X'
                    if has_time:
                        top_bottom = c != 'X'
                    if node in self.past_edges:
                        z_scale = 0.1
                left_right = (1, 0, 0, 1) if left_right else (0, 0, 1, 1)
                top_bottom = (1, 0, 0, 1) if top_bottom else (0, 0, 1, 1)
                front_back = (1, 0, 0, 1) if front_back else (0, 0, 1, 1)

                for d in [-0.5, 0.5]:
                    if node + d not in self.edges:
                        add_face([x + d, y, z - (1 - z_scale) / 2], d1=(0, 1, 0), d2=(0, 0, z_scale), rgba=left_right, include_lines=True, include_triangles=True)
                    if node + d * 1j not in self.edges:
                        add_face([x, y + d, z - (1 - z_scale) / 2], d1=(1, 0, 0), d2=(0, 0, z_scale), rgba=front_back, include_lines=True, include_triangles=True)
                for (b, d) in [(node in self.past_edges, -z_scale*0.5), (node in self.future_edges, +z_scale*0.5)]:
                    if not b:
                        add_face([x, y, z + d - (1 - z_scale) / 2], d1=(1, 0, 0), d2=(0, 1, 0), rgba=top_bottom, include_lines=True, include_triangles=True)
                add_cube([x, y, z - (1 - z_scale) / 2], sz=z_scale, rgba=(0, 0, 0, 1), include_triangles=False, include_lines=True)
            elif c == 'H':
                add_cube([x, y, z], rgba=(1, 1, 0, 1), include_triangles=True, include_lines=True)
            elif len(c) >= 2 and set(c) <= set('XYZ'):
                if c[0] == 'X':
                    rgba1 = (0, 0, 1, 1) if degrees[node] == 1 else (1, 0, 0, 1)
                elif c[0] == 'Z':
                    rgba1 = (0, 0, 1, 1) if degrees[node] != 1 else (1, 0, 0, 1)
                elif c[0] == 'Y':
                    rgba1 = (0, 1, 0, 1)
                else:
                    rgba1 = (0.5, 0.5, 0.5, 1)
                if c[1] == 'X':
                    rgba2 = (0, 0, 1, 1) if degrees[node] == 1 else (1, 0, 0, 1)
                elif c[1] == 'Z':
                    rgba2 = (0, 0, 1, 1) if degrees[node] != 1 else (1, 0, 0, 1)
                elif c[1] == 'Y':
                    rgba2 = (0, 1, 0, 1)
                else:
                    rgba2 = (0.5, 0.5, 0.5, 1)
                _draw_interlocking_double_color_cube(
                    x=x,
                    y=y,
                    z=z - 0.25 if degrees[node] == 1 else 0.5,
                    sx=0.5,
                    sy=0.5,
                    sz=0.25 if degrees[node] == 1 else 0.5,
                    xy=xy,
                    rgba1=rgba1,
                    rgba2=rgba2,
                    out_triangles=out_triangles,
                    out_lines=out_lines,
                )
                continue
            elif c != '.':
                if c == 'T':
                    rgba = (1, 0, 1, 1)
                elif c == 'Y':
                    rgba = (0, 1, 0, 1)
                    did_work = False
                    if node in self.past_edges:
                        _draw_y_node(
                            xz_orientation=orientations.get((node, layer - 0.5)),
                            xy=lambda dx, dy, dz: xy([x + dx, y + dy, z - 0.1 + dz * 0.8]),
                            out_lines=out_lines,
                            out_triangles=out_triangles,
                        )
                        did_work = True
                    if node in self.future_edges:
                        _draw_y_node(
                            xz_orientation=orientations.get((node, layer + 0.5)),
                            xy=lambda dx, dy, dz: xy([x + dx, y + dy, z + 0.1 - dz * 0.8]),
                            out_lines=out_lines,
                            out_triangles=out_triangles,
                        )
                        did_work = True
                    if did_work:
                        continue
                else:
                    rgba = (0, 0, 0, 1)
                add_cube([x, y, z], rgba=rgba, include_triangles=True, include_lines=True)

        for node in self.nodes:
            for (b, dz) in [(node in self.past_edges, -1), (node in self.future_edges, +1)]:
                if not b:
                    continue
                orientation = orientations.get((node, layer + dz*0.5))
                x = node.real * pitch
                y = node.imag * pitch
                z = layer * pitch
                c1 = (0, 0, 0, 1) if orientation is None else (1, 0, 0, 1) if orientation else (0, 0, 1, 1)
                c2 = (0, 0, 0, 1) if orientation is None else (0, 0, 1, 1) if orientation else (1, 0, 0, 1)
                s2 = spacing / 2
                add_face((x, y - 0.5, z + dz*0.5 + s2*dz*0.5), d1=(1, 0, 0), d2=(0, 0, s2), rgba=c1, include_lines=False, include_triangles=True)
                add_face((x, y + 0.5, z + dz*0.5 + s2*dz*0.5), d1=(1, 0, 0), d2=(0, 0, s2), rgba=c1, include_lines=False, include_triangles=True)
                add_face((x - 0.5, y, z + dz*0.5 + s2*dz*0.5), d1=(0, 1, 0), d2=(0, 0, s2), rgba=c2, include_lines=False, include_triangles=True)
                add_face((x + 0.5, y, z + dz*0.5 + s2*dz*0.5), d1=(0, 1, 0), d2=(0, 0, s2), rgba=c2, include_lines=False, include_triangles=True)
                out_lines.append(gen.ColoredLineData(
                    rgba=(0, 0, 0, 1),
                    edge_list=np.array(
                        [
                            xy([x2, y2, z2])
                            for x2 in [x - 0.5, x + 0.5]
                            for y2 in [y - 0.5, y + 0.5]
                            for z2 in [z + dz*0.5, z + dz*0.5 + dz*s2]
                        ],
                        dtype=np.float32,
                    ),
                ))
                add_face((x, y - 0.5, z + dz*0.5 + s2*dz*0.5), d1=(1, 0, 0), d2=(0, 0, s2), rgba=c1, include_lines=False, include_triangles=True)
                add_face((x, y + 0.5, z + dz*0.5 + s2*dz*0.5), d1=(1, 0, 0), d2=(0, 0, s2), rgba=c1, include_lines=False, include_triangles=True)
                add_face((x - 0.5, y, z + dz*0.5 + s2*dz*0.5), d1=(0, 1, 0), d2=(0, 0, s2), rgba=c2, include_lines=False, include_triangles=True)
                add_face((x + 0.5, y, z + dz*0.5 + s2*dz*0.5), d1=(0, 1, 0), d2=(0, 0, s2), rgba=c2, include_lines=False, include_triangles=True)
        for edge in self.edges:
            x = edge.real * pitch
            y = edge.imag * pitch
            z = layer * pitch
            orientation = orientations.get((edge, layer))
            d = 1 if edge.real % 1 == 0.5 else 1j
            c1 = (0, 0, 0, 1) if orientation is None else (1, 0, 0, 1) if orientation else (0, 0, 1, 1)
            c2 = (0, 0, 0, 1) if orientation is None else (0, 0, 1, 1) if orientation else (1, 0, 0, 1)
            add_face((x, y, z - 0.5), d1=(d.real*spacing, d.imag*spacing, 0), d2=(-d.imag, d.real, 0), rgba=c1, include_lines=True, include_triangles=True)
            add_face((x, y, z + 0.5), d1=(d.real*spacing, d.imag*spacing, 0), d2=(-d.imag, d.real, 0), rgba=c1, include_lines=True, include_triangles=True)
            add_face((x-d.imag*0.5, y+d.real*0.5, z), d1=(d.real*spacing, d.imag*spacing, 0), d2=(0, 0, 1), rgba=c2, include_lines=True, include_triangles=True)
            add_face((x+d.imag*0.5, y-d.real*0.5, z), d1=(d.real*spacing, d.imag*spacing, 0), d2=(0, 0, 1), rgba=c2, include_lines=True, include_triangles=True)

    def add_wireframe_to_3d_skeleton_gltf(
        self,
        layer: int,
        spacing: float,
        out_lines: list[gen.ColoredLineData],
        out_triangles: list[gen.ColoredTriangleData],
        injected_errors: frozenset[InjectedError],
    ) -> None:
        pitch = spacing + 1

        def xy(v):
            return np.array([-v[2], v[0], -v[1]])

        degs = self.compute_node_degrees()
        for node, c in self.nodes.items():
            if degs[node] == 2 and (c == 'X' or c == 'Z'):
                continue
            x = node.real * pitch
            y = node.imag * pitch
            z = layer * pitch
            if c != '.':
                if c == 'X':
                    rgba = (1, 1, 1, 1)
                elif c == 'Z':
                    rgba = (0, 0, 0, 1)
                elif c == 'H':
                    rgba = (1, 1, 0, 1)
                elif c == 'T':
                    rgba = (1, 0, 1, 1)
                elif c == 'Y':
                    rgba = (0, 1, 0, 1)
                elif len(c) == 2 and set(c) <= set('XYZ'):
                    if c[0] == 'X':
                        rgba1 = (1, 1, 1, 1)
                    elif c[0] == 'Z':
                        rgba1 = (0, 0, 0, 1)
                    elif c[0] == 'Y':
                        rgba1 = (0, 1, 0, 1)
                    else:
                        rgba1 = (0.5, 0.5, 0.5, 1)
                    if c[1] == 'X':
                        rgba2 = (1, 1, 1, 1)
                    elif c[1] == 'Z':
                        rgba2 = (0, 0, 0, 1)
                    elif c[1] == 'Y':
                        rgba2 = (0, 1, 0, 1)
                    else:
                        rgba2 = (0.5, 0.5, 0.5, 1)
                    _draw_interlocking_double_color_cube(
                        x=x,
                        y=y,
                        z=z,
                        sx=0.25,
                        sy=0.25,
                        sz=0.25,
                        xy=xy,
                        rgba1=rgba1,
                        rgba2=rgba2,
                        out_triangles=out_triangles,
                        out_lines=out_lines,
                    )
                    continue
                else:
                    rgba = (0, 0, 0, 0)
                add_cube_to(
                    [x, y, z],
                    sx=0.5,
                    sy=0.5,
                    sz=0.5,
                    rgba=rgba,
                    xy=xy,
                    out_triangles=out_triangles,
                    out_lines=out_lines,
                    include_lines=True,
                    include_triangles=rgba != (0, 0, 0, 0),
                )

        def jitter() -> float:
            return 0.2 * (random.random() * 2 - 1)

        for err in injected_errors:
            x = err.pos.real
            y = err.pos.imag
            z = layer + err.layer
            if err.basis == 'X':
                rgba = (1, 0, 0, 1)
                x -= 0.1 * (x % 1)
                y -= 0.1 * (y % 1)
                z -= 0.1 * (z % 1)
            elif err.basis == 'Z':
                rgba = (0, 0, 1, 1)
                x += 0.1 * (x % 1)
                y += 0.1 * (y % 1)
                z += 0.1 * (z % 1)
            elif err.basis == 'Y':
                rgba = (0, 1, 0, 1)
            else:
                raise NotImplementedError(f'{err.basis=}')
            x *= pitch
            y *= pitch
            z *= pitch
            out_triangles.append(gen.ColoredTriangleData(
                rgba=rgba,
                triangle_list=np.array([
                    xy([x + jitter(), y + jitter(), z + jitter()])
                    for _ in range(48)
                ], dtype=np.float32),
            ))

        edge_xyzs = []
        for edge in self.edges:
            edge_xyzs.append((edge.real, edge.imag, layer))
        for edge in self.future_edges:
            edge_xyzs.append((edge.real, edge.imag, layer + 0.5))
        for edge in self.past_edges:
            edge_xyzs.append((edge.real, edge.imag, layer - 0.5))
        for x, y, z in edge_xyzs:
            dx = dy = dz = 0
            if x % 1 == 0.5:
                dx = 0.5
            if y % 1 == 0.5:
                dy = 0.5
            if z % 1 == 0.5:
                dz = 0.5
            c1 = xy([x - dx, y - dy, z - dz]) * pitch
            c2 = xy([x + dx, y + dy, z + dz]) * pitch
            out_lines.append(gen.ColoredLineData(
                rgba=(0, 0, 0, 0),
                edge_list=np.array([c1, c2], dtype=np.float32)
            ))

    def to_nx_zx_graph(self, *, show: bool = False) -> 'nx.Graph':
        """Converts the lattice surgery into a nearly-ZX-calculus graph.

        The graph has the advantage of not treating space and time differently.
        """
        import networkx as nx
        graph = nx.Graph()
        for node, c in self.nodes.items():
            if c == 'Z':
                graph.add_node(node, type='Z', phase=+1)
            elif c == 'X':
                graph.add_node(node, type='X', phase=+1)
            elif c == 'Y':
                if node in self.past_edges:
                    graph.add_node(node, type='Z', phase=1j)
                if node in self.future_edges:
                    graph.add_node((node, 'prep'), type='Z', phase=1j)
            elif c == 'T':
                space_deg = sum(node + d / 2 in self.edges for d in [1, -1, 1j, -1j])
                time_deg = (node in self.past_edges) + (node in self.future_edges)

                if time_deg + space_deg == 1:
                    graph.add_node(node, type='T_port', phase=1j**0.5)
                else:
                    graph.add_node(node, type='Z', phase=+1)
                    graph.add_node((node, 'T_in'), type='T_port', phase=1j**0.5)
                    graph.add_edge(node, (node, 'T_in'))
            elif c == 'S':
                space_deg = sum(node + d / 2 in self.edges for d in [1, -1, 1j, -1j])
                time_deg = (node in self.past_edges) + (node in self.future_edges)

                if time_deg + space_deg == 1:
                    graph.add_node(node, type='S_port', phase=1j)
                else:
                    graph.add_node(node, type='Z', phase=+1)
                    graph.add_node((node, 'S_in'), type='S_port', phase=1j)
                    graph.add_edge(node, (node, 'S_in'))
            elif c == 'H':
                space_deg = sum(node + d / 2 in self.edges for d in [1, -1, 1j, -1j])
                time_deg = (node in self.past_edges) + (node in self.future_edges)
                assert space_deg == 0 and time_deg == 2
                graph.add_node(node, type='H', phase=+1)
            elif c == '.':
                pass
            elif len(c) == 2 and set(c) <= set('XYZ'):
                raise ValueError(f"{c} node must be resolved before converting into a graph.")
            else:
                raise NotImplementedError(f'{c=}')

            if node in self.past_edges:
                graph.add_node((node, 'past_in'), type='in_port', phase=+1)
            if node in self.future_edges:
                graph.add_node((node, 'future_out'), type='out_port', phase=+1)

        for node, c in self.nodes.items():
            is_instant_measure = True
            for d in [-1, 1, -1j, +1j]:
                e = node + d/2
                if e in self.edges:
                    graph.add_edge(node, node + d, x_sink=e in self.x_sink_edges, z_sink=e in self.z_sink_edges)
                    is_instant_measure = False
            if node in self.future_edges:
                graph.add_edge((node, 'future_out'), node if c != 'Y' else (node, 'prep'))
                is_instant_measure = False
            if node in self.past_edges:
                graph.add_edge(
                    (node, 'past_in'),
                    node,
                    x_sink=c == 'Y' or (c == 'X' and is_instant_measure),
                    z_sink=c == 'Z' and is_instant_measure)

        if show:
            import matplotlib.pyplot as plt
            n2color = {'X': 'red', 'Z': 'blue', 'in_port': 'white', 'out_port': 'white', 'T_port': 'white'}
            n2pos = {}
            node_labels = {}
            for node, vals in graph.nodes.items():
                if isinstance(node, complex):
                    c = node
                else:
                    c, r = node
                    d = 0.1 - 0.15j
                    c += d * {'T_in': -1, 'future_out': +2, 'future': +1, 'past': -1, 'past_in': -2, 'prep': +0.15}[r]
                n2pos[node] = (c.real, -c.imag)
                t = vals['type']
                if t == 'X' or t == 'Z':
                    if vals['phase'] == 1j:
                        label = '½π'
                    elif vals['phase'] == -1j:
                        label = '-½π'
                    elif vals['phase'] == -1:
                        label = 'π'
                    else:
                        label = ''
                elif t == 'in_port' or t == 'out_port':
                    label = ''
                elif t == 'T_port':
                    label = 'T*'
                elif t == 'S_port':
                    label = 'S*'
                else:
                    raise NotImplementedError(f'{node=}, {vals=}')
                node_labels[node] = label
            nx.draw_networkx(
                graph,
                pos=n2pos,
                labels=node_labels,
                font_color='gray',
                node_color=[n2color[vals['type']] for node, vals in graph.nodes.items()],
            )
            nx.draw_networkx_edge_labels(
                graph,
                pos=n2pos,
                edge_labels={
                    (a, b): '*' * bool(graph.edges[(a, b)].get('x_sink')) + '@' * bool(graph.edges[(a, b)].get('z_sink'))
                    for (a, b), vals in graph.edges.items()
                    if graph.edges[(a, b)].get('x_sink') or graph.edges[(a, b)].get('z_sink')
                },
                font_color='gray',
            )
            plt.show()

        return graph

    def __str__(self) -> str:
        grid = {}
        for n, c in self.nodes.items():
            v = n * 4 + 2 + 2j
            grid[v] = c
        for e in self.future_edges:
            v = e * 4 + 2 + 2j
            grid[v + 1 - 1j] = '/'
        for e in self.past_edges:
            v = e * 4 + 2 + 2j
            grid[v - 1 + 1j] = '/'
        for e in self.edges:
            if e.real % 1 == 0.5:
                v = (e - 0.5) * 4 + 2 + 2j
                grid[v + 1] = '-'
                grid[v + 2] = '-'
                grid[v + 3] = '-'
            elif e.imag % 1 == 0.5:
                v = (e - 0.5j) * 4 + 2 + 2j
                grid[v + 1j] = '|'
                grid[v + 2j] = '|'
                grid[v + 3j] = '|'
            else:
                raise NotImplementedError(f'{e=}')

        max_r = int(max([e.real for e in grid.keys()], default=0))
        max_i = int(max([e.imag for e in grid.keys()], default=0))
        chars = []
        for y in range(max_i + 1):
            for x in range(max_r + 1):
                chars.append(grid.get(x + 1j*y, ' '))
            chars.append('\n')
        return ''.join(chars)

    def to_zx_graph(self) -> ZXGraph:
        return ZXGraph.from_nx_graph(self.to_nx_zx_graph())

    @staticmethod
    def from_text(text: str) -> 'LatticeSurgeryLayer':
        xs = collections.Counter()
        ys = collections.Counter()
        x2x: dict[int, int] = {}
        y2y: dict[int, int] = {}
        c2n: dict[complex, complex] = {}
        nodes: dict[complex, Literal['.', 'X', 'Y', 'Z', 'T']] = {}

        m = {
            x + 1j*y: c
            for y, line in enumerate(text.splitlines())
            for x, c in enumerate(line)
        }

        edges = set()
        x_sink_edges = set()
        z_sink_edges = set()
        future_edges = set()
        past_edges = set()
        ds = [-1, -1j, 1j, 1]
        for p, c in m.items():
            if m.get(p - 1, ' ') in '.XYZTSH':
                pass
            elif c in '.XYZTSH':
                x = int(p.real)
                y = int(p.imag)
                xs[x] += 1
                ys[y] += 1
                x2x.setdefault(x, len(x2x))
                y2y.setdefault(y, len(y2y))
                n = x2x[x] + y2y[y] * 1j
                c2n[x + 1j*y] = n
                offset = 1
                while m.get(p + offset, ' ') in '.XYZTSH':
                    c += m.get(p + offset, ' ')
                    offset += 1
                nodes[n] = c

                for d in ds:
                    edge_char = '-' if d.imag == 0 else '|'
                    k = 1
                    has_x_excitation = False
                    has_z_excitation = False
                    while True:
                        ec = m.get(p + d*k)
                        if ec == edge_char:
                            pass
                        elif ec == '*':
                            has_x_excitation = True
                        elif ec == '@':
                            has_z_excitation = True
                        else:
                            break
                        k += 1
                    if k > 1:
                        edges.add(n + d / 2)
                    if has_x_excitation:
                        x_sink_edges.add(n + d / 2)
                    if has_z_excitation:
                        z_sink_edges.add(n + d / 2)
                if m.get(p - 1 + 1j) == '/':
                    past_edges.add(n)
                if m.get(p + 1 - 1j) == '/':
                    future_edges.add(n)
            elif c in ' -|/*@':
                pass
            else:
                raise NotImplementedError(f'Unrecognized node type: {c=} at {p=} in\n{text}')

        if len(set(xs.values())) != 1 or len(set(ys.values())) != 1:
            raise ValueError(f"The qubits didn't form a complete grid:\n{text}")

        return LatticeSurgeryLayer(
            nodes=nodes,
            edges=frozenset(edges),
            x_sink_edges=frozenset(x_sink_edges),
            z_sink_edges=frozenset(z_sink_edges),
            future_edges=frozenset(future_edges),
            past_edges=frozenset(past_edges),
        )

    def _to_quantum_instructions(
            self,
            *,
            layer_key: Any,
            include_error_mechanisms: bool,
            injected_errors: frozenset[InjectedError],
    ) -> list[LatticeSurgeryInstruction]:
        ds = [-1, -1j, 1j, 1]
        order = sorted(self.nodes, key=lambda n: (
            n in self.future_edges,
            sum(n + d / 2 in self.edges for d in ds),
            n.real,
            n.imag,
        ))

        def ensure_init(n: Any) -> None:
            if n not in initialized:
                tasks.append(LatticeSurgeryInstruction(
                    action=cast(Any, 'qalloc_z' if self.nodes[n] == 'X' else 'qalloc_x'),
                    target=n,
                ))
                initialized.add(n)

        initialized = set(self.past_edges)
        edge_done = set()

        tasks = []

        for n in order:
            if n in self.past_edges:
                for b in 'XYZ':
                    if InjectedError(basis=cast(Any, b), pos=n, layer=-0.5) in injected_errors:
                        tasks.append(LatticeSurgeryInstruction(
                            cast(Any, b.lower()),
                            target=n,
                        ))

                if include_error_mechanisms:
                    for b in 'XZ':
                        tasks.append(LatticeSurgeryInstruction(cast(Any, f'error_mechanism_{b.lower()}'), target=n, error_source=ErrorSource(
                            error_type='timelike_edge_error',
                            error_basis=cast(Any, b),
                            error_location=n,
                            error_initiative='before',
                            error_layer=layer_key,
                        )))

        for n in order:
            c = self.nodes[n]
            if c == 'H':
                tasks.append(LatticeSurgeryInstruction(
                    action='h',
                    target=n,
                ))

        for n in order:
            c = self.nodes[n]
            space_deg = sum(n + d / 2 in self.edges for d in ds)
            time_deg = (n in self.past_edges) + (n in self.future_edges)
            if c == '.':
                assert space_deg == 0
                assert time_deg == 0
            elif c == 'Y':
                assert space_deg == 0

        for n in order:
            c = self.nodes[n]
            if c == 'Y' and n in self.past_edges:
                tasks.append(LatticeSurgeryInstruction(
                    action='m_discard_y',
                    target=n,
                    measure_key=(f'MY', n, layer_key),
                ))

        for n in order:
            c = self.nodes[n]
            if c == 'Y' or c == '.':
                continue

            ensure_init(n)
            if c == 'T' or c == 'S':
                if c == 'T':
                    tasks.append(LatticeSurgeryInstruction('t', n))
                else:
                    tasks.append(LatticeSurgeryInstruction('s', n))
                tasks.append(LatticeSurgeryInstruction('heralded_random_x', n, measure_key=('heralded_random_x', n, layer_key)))
                tasks.append(LatticeSurgeryInstruction('heralded_random_z', n, measure_key=('heralded_random_z', n, layer_key)))
                if include_error_mechanisms:
                    tasks.append(LatticeSurgeryInstruction(
                        'error_mechanism_x',
                        target=n,
                        error_source=ErrorSource(
                            error_type='inject_error',
                            error_basis='X',
                            error_location=n,
                            error_initiative='during',
                            error_layer=layer_key,
                        ),
                    ))
                    tasks.append(LatticeSurgeryInstruction(
                        'error_mechanism_z',
                        target=n,
                        error_source=ErrorSource(
                            error_type='inject_error',
                            error_basis='Z',
                            error_location=n,
                            error_initiative='during',
                            error_layer=layer_key,
                        ),
                    ))
            for d in ds:
                e = n + d / 2
                if e in self.edges and e not in edge_done:
                    edge_done.add(e)

                    n2 = n + d
                    ensure_init(n2)
                    n1 = n
                    b1: Literal['X', 'Z'] = cast(Literal['X', 'Z'], 'X' if c == 'X' else 'Z')
                    b2: Literal['X', 'Z'] = cast(Literal['X', 'Z'], 'X' if self.nodes[n2] == 'X' else 'Z')
                    nc = (n1 + n2) / 2
                    if b1 == b2:
                        # Pair measurement case.
                        if (n1.real, n1.imag) > (n2.real, n2.imag):
                            n1, n2 = n2, n1
                        action = cast(Any, 'mxx' if b1 == 'X' else 'mzz')
                        mkey = (action.upper(), (n1 + n2) / 2, layer_key)
                        other_b = cast(Literal['X', 'Z'], 'Z' if b1 == 'X' else 'X')

                        if include_error_mechanisms:
                            tasks.append(LatticeSurgeryInstruction(
                                'error_mechanism_m',
                                measure_key=mkey,
                                error_source=ErrorSource(
                                    error_type='spacelike_edge_error',
                                    error_basis=other_b,
                                    error_location=nc,
                                    error_initiative='during',
                                    error_layer=layer_key,
                                ),
                            ))
                            tasks.append(LatticeSurgeryInstruction(
                                cast(Any, f'error_mechanism_{b1.lower()}'),
                                target=n1,
                                error_source=ErrorSource(
                                    error_type='spacelike_edge_error',
                                    error_basis=b1,
                                    error_location=nc,
                                    error_initiative='during',
                                    error_layer=layer_key,
                                ),
                            ))
                        # Injected errors on the crossbar can be moved to be before+after the measurement.
                        for b in 'XYZ':
                            if InjectedError(basis=cast(Any, b), pos=nc, layer=0) in injected_errors:
                                tasks.append(LatticeSurgeryInstruction(cast(Any, b.lower()), target=n1))
                        tasks.append(LatticeSurgeryInstruction(
                            action=action,
                            target=n1,
                            target2=n2,
                            measure_key=mkey
                        ))
                        # Injected errors on the crossbar can be moved to be before+after the measurement.
                        for b in 'XYZ':
                            if b != b1 and InjectedError(basis=cast(Any, b), pos=nc, layer=0) in injected_errors:
                                tasks.append(LatticeSurgeryInstruction(cast(Any, other_b.lower()), target=n1))
                    else:
                        # CX case.
                        if b2 == 'Z':
                            n1, n2 = n2, n1
                        tasks.append(LatticeSurgeryInstruction('cx', target=n1, target2=n2))

                        # Injected errors on the crossbar can be moved to the control or target qubit.
                        if InjectedError(basis='X', pos=nc, layer=0) in injected_errors:
                            tasks.append(LatticeSurgeryInstruction('x', target=n2))
                        if InjectedError(basis='Z', pos=nc, layer=0) in injected_errors:
                            tasks.append(LatticeSurgeryInstruction('z', target=n1))
                        if InjectedError(basis='Y', pos=nc, layer=0) in injected_errors:
                            tasks.append(LatticeSurgeryInstruction('x', target=n2))
                            tasks.append(LatticeSurgeryInstruction('z', target=n1))

                        if include_error_mechanisms:
                            tasks.append(LatticeSurgeryInstruction(
                                'error_mechanism_x',
                                target=n2,
                                error_source=ErrorSource(
                                    error_type='spacelike_edge_error',
                                    error_basis='X',
                                    error_location=nc,
                                    error_initiative='during',
                                    error_layer=layer_key,
                                ),
                            ))
                            tasks.append(LatticeSurgeryInstruction(
                                'error_mechanism_z',
                                target=n1,
                                error_source=ErrorSource(
                                    error_type='spacelike_edge_error',
                                    error_basis='Z',
                                    error_location=nc,
                                    error_initiative='during',
                                    error_layer=layer_key,
                                ),
                            ))
            if n not in self.future_edges:
                action = cast(Any, 'm_discard_z' if c == 'X' else 'm_discard_x')
                tasks.append(LatticeSurgeryInstruction(
                    action=action,
                    target=n,
                    measure_key=('MZ' if c == 'X' else 'MX', n, layer_key),
                ))

        for n in order:
            c = self.nodes[n]
            if c == 'Y' and n in self.future_edges:
                tasks.append(LatticeSurgeryInstruction('qalloc_y', target=n))

        for n in order:
            if n in self.future_edges:
                if include_error_mechanisms:
                    tasks.append(LatticeSurgeryInstruction('error_mechanism_x', target=n, error_source=ErrorSource(
                        error_type='timelike_edge_error',
                        error_basis='X',
                        error_location=n,
                        error_initiative='after',
                        error_layer=layer_key,
                    )))
                    tasks.append(LatticeSurgeryInstruction('error_mechanism_z', target=n, error_source=ErrorSource(
                        error_type='timelike_edge_error',
                        error_basis='Z',
                        error_location=n,
                        error_initiative='after',
                        error_layer=layer_key,
                    )))

        return tasks

    def _to_classical_feedback_instructions(self, *, quantum_instructions: list[LatticeSurgeryInstruction], layer_key: Any) -> list[LatticeSurgeryInstruction]:
        g = self.to_zx_graph()
        err_feed_map = g.error_to_feedback_map
        result = []
        m_feedbacks = {
            k: [] for k in range(len(g.measurement_stabilizers))
        }
        for q in quantum_instructions:
            err_key: tuple[Any, Literal['X', 'Z']]
            if q.action == 'm_discard_x':
                err_key = (q.target, 'Z')
            elif q.action == 'm_discard_z':
                err_key = (q.target, 'X')
            elif q.action == 'm_discard_y':
                err_key = ((q.target, 'past_in'), 'X')
            elif q.action == 'mxx':
                err_key = ((q.target, q.target2), 'Z')
            elif q.action == 'mzz':
                err_key = ((q.target, q.target2), 'X')
            elif q.action == 'heralded_random_x':
                err_key = (q.target, 'X')
            elif q.action == 'heralded_random_z':
                err_key = (q.target, 'Z')
            else:
                continue

            feedback = err_feed_map[err_key]
            for m in sorted(feedback.ms):
                m_feedbacks[m].append(q.measure_key)
            for x in sorted(feedback.xs, key=lambda e: (e[0].real, e[0].imag)):
                assert x[1] == 'future_out'
                result.append(LatticeSurgeryInstruction('feedback_m2x', measure_key=q.measure_key, target=x[0]))
            for z in sorted(feedback.zs, key=lambda e: (e[0].real, e[0].imag)):
                assert z[1] == 'future_out'
                result.append(LatticeSurgeryInstruction('feedback_m2z', measure_key=q.measure_key, target=z[0]))
        for k, ms in m_feedbacks.items():
            result.append(LatticeSurgeryInstruction('accumulator_bit_clear', measure_key=None, target=None))
            for m in ms:
                result.append(LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=m, target=None))
            result.append(LatticeSurgeryInstruction('accumulator_bit_save', measure_key=(f'logical', k, layer_key), target=None))
        return result

    def to_sim_instructions(
            self,
            *,
            layer_key: Any,
            include_error_mechanisms: bool = False,
            injected_errors: frozenset[InjectedError] = frozenset(),
    ) -> tuple[LatticeSurgeryInstruction, ...]:
        k = (layer_key, include_error_mechanisms, injected_errors)
        if self._cached_tasks is not None and self._cached_tasks_key == k:
            return self._cached_tasks
        self._cached_tasks_key = k
        quantum_instructions = self._to_quantum_instructions(
            layer_key=layer_key,
            include_error_mechanisms=include_error_mechanisms,
            injected_errors=injected_errors,
        )
        classical_instructions = self._to_classical_feedback_instructions(
            quantum_instructions=quantum_instructions,
            layer_key=layer_key,
        )
        self._cached_tasks = tuple(quantum_instructions + classical_instructions)
        return self._cached_tasks


def add_face_to(
        center: Sequence[float],
        *,
        d1: Sequence[float],
        d2: Sequence[float],
        rgba: tuple[float, float, float, float],
        out_triangles: list[gen.ColoredTriangleData],
        out_lines: list[gen.ColoredLineData],
        xy: Callable[[np.ndarray], np.ndarray],
        include_triangles: bool,
        include_lines: bool,
):
    origin = np.array(center)
    d1 = np.array(d1)
    d2 = np.array(d2)
    origin -= d1 / 2
    origin -= d2 / 2
    origin = xy(origin)
    d1 = xy(d1)
    d2 = xy(d2)
    if include_triangles:
        out_triangles.append(gen.ColoredTriangleData(
            rgba=rgba,
            triangle_list=np.array([
                origin,
                origin + d1,
                origin + d2,

                origin + d1 + d2,
                origin + d1,
                origin + d2,
            ], dtype=np.float32)
        ))
    if include_lines:
        out_lines.append(gen.ColoredLineData(
            rgba=(0, 0, 0, 1),
            edge_list=np.array([
                origin,
                origin + d1,

                origin + d1 + d2,
                origin + d1,

                origin,
                origin + d2,

                origin + d1 + d2,
                origin + d2,
            ], dtype=np.float32)
        ))


def add_cube_to(
        center: list[float],
        *,
        sx: float = 1,
        sy: float = 1,
        sz: float = 1,
        rgba: tuple[float, float, float, float],
        out_triangles: list[gen.ColoredTriangleData],
        out_lines: list[gen.ColoredLineData],
        xy: Callable[[np.ndarray], np.ndarray],
        include_triangles: bool,
        include_lines: bool,
):
    x, y, z = center
    f = functools.partial(add_face_to, rgba=rgba, xy=xy, out_triangles=out_triangles, out_lines=out_lines, include_lines=include_lines, include_triangles=include_triangles)
    f([x, y, z - sz/2], d1=[sx, 0, 0], d2=[0, sy, 0])
    f([x, y, z + sz/2], d1=[sx, 0, 0], d2=[0, sy, 0])
    f([x, y - sy/2, z], d1=[sx, 0, 0], d2=[0, 0, sz])
    f([x, y + sy/2, z], d1=[sx, 0, 0], d2=[0, 0, sz])
    f([x - sx/2, y, z], d1=[0, sy, 0], d2=[0, 0, sz])
    f([x + sx/2, y, z], d1=[0, sy, 0], d2=[0, 0, sz])


def _draw_y_node(
        *,
        xz_orientation: bool,
        xy: Callable[[float, float, float], np.ndarray],
        out_lines: list[gen.ColoredLineData],
        out_triangles: list[gen.ColoredTriangleData],
):
    c1 = (1, 0, 0, 1) if xz_orientation else (0, 0, 1, 1)
    c2 = (1, 0, 0, 1) if not xz_orientation else (0, 0, 1, 1)
    out_lines.append(gen.ColoredLineData(
        rgba=(0, 0, 0, 0),
        edge_list=np.array([
            xy(-0.5, +0.5, -0.5),
            xy(+0.5, +0.5, -0.5),
            xy(-0.5, -0.5, -0.5),
            xy(+0.5, -0.5, -0.5),
            xy(+0.5, -0.5, -0.5),
            xy(+0.5, +0.5, -0.5),
            xy(-0.5, -0.5, -0.5),
            xy(-0.5, +0.5, -0.5),

            xy(-0.5, +0.5, -0.4),
            xy(+0.5, +0.5, -0.4),
            xy(-0.5, -0.5, -0.4),
            xy(+0.5, -0.5, -0.4),
            xy(+0.5, -0.5, -0.4),
            xy(+0.5, +0.5, -0.4),
            xy(-0.5, -0.5, -0.4),
            xy(-0.5, +0.5, -0.4),

            xy(-0.5, +0.5, 0),
            xy(+0.5, +0.5, 0),
            xy(-0.5, -0.5, 0),
            xy(+0.5, -0.5, 0),
            xy(+0.5, -0.5, 0),
            xy(+0.5, +0.5, 0),
            xy(-0.5, -0.5, 0),
            xy(-0.5, +0.5, 0),

            xy(-0.5, +0.5, -0.5),
            xy(-0.5, +0.5, 0),
            xy(+0.5, +0.5, -0.5),
            xy(+0.5, +0.5, 0),
            xy(-0.5, -0.5, -0.5),
            xy(-0.5, -0.5, 0),
            xy(+0.5, -0.5, -0.5),
            xy(+0.5, -0.5, 0),

            xy(-0.5, +0.5, 0),
            xy(+0.5, -0.5, 0),
        ], dtype=np.float32),
    ))
    out_triangles.append(gen.ColoredTriangleData(
        rgba=c1,
        triangle_list=np.array([
            xy(-0.5, +0.5, 0 - 0.5),
            xy(+0.5, +0.5, 0 - 0.5),
            xy(+0.5, +0.5, 0),

            xy(-0.5, +0.5, 0),
            xy(+0.5, +0.5, 0),
            xy(-0.5, +0.5, 0 - 0.5),

            xy(+0.5, -0.5, 0 - 0.4),
            xy(+0.5, +0.5, 0 - 0.4),
            xy(+0.5, +0.5, 0),

            xy(+0.5, -0.5, 0),
            xy(+0.5, +0.5, 0),
            xy(+0.5, -0.5, 0 - 0.4),

            xy(+0.5, -0.5, 0),
            xy(+0.5, +0.5, 0),
            xy(-0.5, +0.5, 0),
        ], dtype=np.float32)
    ))
    out_triangles.append(gen.ColoredTriangleData(
        rgba=(1, 1, 0, 1),
        triangle_list=np.array([
            xy(-0.5, -0.5, -0.5),
            xy(+0.5, -0.5, -0.5),
            xy(+0.5, -0.5, -0.4),

            xy(-0.5, -0.5, -0.4),
            xy(+0.5, -0.5, -0.4),
            xy(-0.5, -0.5, -0.5),

            xy(+0.5, -0.5, -0.5),
            xy(+0.5, +0.5, -0.5),
            xy(+0.5, +0.5, -0.4),

            xy(+0.5, -0.5, -0.4),
            xy(+0.5, +0.5, -0.4),
            xy(+0.5, -0.5, -0.5),

            xy(+0.5, +0.5, -0.45),
            xy(+0.5, -0.5, -0.45),
            xy(-0.5, -0.5, -0.45),
        ], dtype=np.float32)
    ))
    out_triangles.append(gen.ColoredTriangleData(
        rgba=c2,
        triangle_list=np.array([
            xy(-0.5, -0.5, 0 - 0.4),
            xy(+0.5, -0.5, 0 - 0.4),
            xy(+0.5, -0.5, 0),

            xy(-0.5, -0.5, 0),
            xy(+0.5, -0.5, 0),
            xy(-0.5, -0.5, 0 - 0.4),

            xy(-0.5, -0.5, 0 - 0.5),
            xy(-0.5, +0.5, 0 - 0.5),
            xy(-0.5, +0.5, 0),

            xy(-0.5, -0.5, 0),
            xy(-0.5, +0.5, 0),
            xy(-0.5, -0.5, 0 - 0.5),

            xy(-0.5, +0.5, 0),
            xy(-0.5, -0.5, 0),
            xy(+0.5, -0.5, 0),
        ], dtype=np.float32)
    ))


def _draw_interlocking_double_color_cube(
        *,
        x: float,
        y: float,
        z: float,
        sx: float = 1,
        sy: float = 1,
        sz: float = 1,
        xy: Callable[[list[float]], np.ndarray],
        rgba1: tuple[float, float, float, float],
        rgba2: tuple[float, float, float, float],
        out_triangles: list[gen.ColoredTriangleData],
        out_lines: list[gen.ColoredLineData],
):
    add_cube_to(
        center=[x, y, z],
        sx=sx * 2,
        sy=sy * 2,
        sz=sz * 2,
        xy=xy,
        rgba=(0, 0, 0, 1),
        out_triangles=out_triangles,
        out_lines=out_lines,
        include_lines=True,
        include_triangles=False,
    )
    out_triangles.append(gen.ColoredTriangleData(
        rgba=rgba1,
        triangle_list=np.array([
            xy([x - sx, y - sy, z - sz]),
            xy([x - sx, y - sy, z + sz]),
            xy([x - sx, y + sy, z + sz]),

            xy([x + sx, y - sy, z + sz]),
            xy([x + sx, y - sy, z - sz]),
            xy([x - sx, y - sy, z - sz]),

            xy([x + sx, y + sy, z + sz]),
            xy([x + sx, y - sy, z + sz]),
            xy([x - sx, y - sy, z + sz]),

            xy([x + sx, y + sy, z + sz]),
            xy([x + sx, y + sy, z - sz]),
            xy([x + sx, y - sy, z - sz]),

            xy([x - sx, y + sy, z - sz]),
            xy([x - sx, y + sy, z + sz]),
            xy([x + sx, y + sy, z + sz]),

            xy([x - sx, y - sy, z - sz]),
            xy([x - sx, y + sy, z - sz]),
            xy([x + sx, y + sy, z - sz]),
        ], dtype=np.float32)
    ))
    out_triangles.append(gen.ColoredTriangleData(
        rgba=rgba2,
        triangle_list=np.array([
            xy([x - sx, y + sy, z + sz]),
            xy([x - sx, y + sy, z - sz]),
            xy([x - sx, y - sy, z - sz]),

            xy([x - sx, y - sy, z - sz]),
            xy([x - sx, y - sy, z + sz]),
            xy([x + sx, y - sy, z + sz]),

            xy([x - sx, y - sy, z + sz]),
            xy([x - sx, y + sy, z + sz]),
            xy([x + sx, y + sy, z + sz]),

            xy([x + sx, y - sy, z - sz]),
            xy([x + sx, y - sy, z + sz]),
            xy([x + sx, y + sy, z + sz]),

            xy([x + sx, y + sy, z + sz]),
            xy([x + sx, y + sy, z - sz]),
            xy([x - sx, y + sy, z - sz]),

            xy([x + sx, y + sy, z - sz]),
            xy([x + sx, y - sy, z - sz]),
            xy([x - sx, y - sy, z - sz]),
        ], dtype=np.float32)
    ))
