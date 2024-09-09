import collections
from typing import Union, Literal, Iterable, Callable, Any

import stim

from ._chunk import Chunk
from ._chunk_loop import ChunkLoop
from ._chunk_reflow import ChunkReflow
from ._complex_util import sorted_complex
from ._flow import Flow
from ._keyed_pauli_map import KeyedPauliMap
from ._pauli_map import PauliMap


class ChunkCompiler:
    """Compiles appended chunks into a unified circuit."""

    def __init__(
        self,
        *,
        flow_to_extra_coords_func: Callable[[Flow], Iterable[float]] = lambda _: (),
    ):
        """
        Args:
            flow_to_extra_coords_func: Determines coordinate data appended to detectors
                (after x, y, and t).
        """
        self.open_flows: dict[
            PauliMap | KeyedPauliMap, Union[Flow, Literal["discard"]]
        ] = {}
        self.num_measurements: int = 0
        self.circuit: stim.Circuit = stim.Circuit()
        self.q2i: dict[complex, int] = {}
        self.o2i: dict[Any, int] = {}
        self.discarded_observables: set[int] = set()
        self.flow_to_extra_coords_func: Callable[[Flow], Iterable[float]] = (
            flow_to_extra_coords_func
        )

    def ensure_qubits_included(self, qubits: Iterable[complex]):
        """Adds the given qubit positions to the indexed positions, if they aren't already."""
        for q in sorted_complex(qubits):
            if q not in self.q2i:
                self.q2i[q] = len(self.q2i)

    def copy(self) -> "ChunkCompiler":
        """Returns a deep copy of the compiler's state."""
        result = ChunkCompiler(flow_to_extra_coords_func=self.flow_to_extra_coords_func)
        result.open_flows = dict(self.open_flows)
        result.num_measurements = self.num_measurements
        result.circuit = self.circuit.copy()
        result.q2i = dict(self.q2i)
        result.o2i = dict(self.o2i)
        result.discarded_observables = set(self.discarded_observables)
        return result

    def __str__(self) -> str:
        lines = [
            "ChunkCompiler {",
            "    discard_flows {",
        ]

        for key, flow in self.open_flows.items():
            if flow == "discard":
                lines.append(f"        {key}")
        lines.append("    }")

        lines.append("    det_flows {")
        for key, flow in self.open_flows.items():
            if flow != "discard" and flow.obs_key is None:
                lines.append(f"        {flow.end}, ms={flow.measurement_indices}")
        lines.append("    }")

        lines.append("    obs_flows {")
        for key, flow in self.open_flows.items():
            if flow != "discard" and flow.obs_key is not None:
                lines.append(f"        {flow.key_end}: ms={flow.measurement_indices}")
        lines.append("    }")

        lines.append(f"    num_measurements = {self.num_measurements}")
        lines.append("}")
        return "\n".join(lines)

    def finish_circuit(self) -> stim.Circuit:
        """Returns the circuit built by the compiler.

        Performs some final translation steps:
        - Re-indexing the qubits to be in a sorted order.
        - Re-indexing the observables to omit discarded observable flows.
        """

        if self.open_flows:
            raise ValueError(
                f"Some flows were unterminated when finishing the circuit:\n\n{self}"
            )

        obs2i = {}
        next_obs_index = 0
        used_indices = set()
        for obs_key, obs_index in self.o2i.items():
            if obs_key in self.discarded_observables:
                obs2i[obs_index] = "discard"
            elif isinstance(obs_key, int):
                obs2i[obs_index] = obs_key
                used_indices.add(obs_key)
        for obs_key, obs_index in sorted(self.o2i.items()):
            if obs_index not in obs2i:
                while next_obs_index in used_indices:
                    next_obs_index += 1
                obs2i[obs_index] = next_obs_index
                next_obs_index += 1

        new_q2i = {q: i for i, q in enumerate(sorted_complex(self.q2i.keys()))}
        final_circuit = stim.Circuit()
        for q, i in new_q2i.items():
            final_circuit.append("QUBIT_COORDS", [i], [q.real, q.imag])
        _append_to_circuit_with_reindexing(
            circuit=self.circuit,
            out=final_circuit,
            new_q2i=new_q2i,
            old_q2i=self.q2i,
            control_time_shift=False,
            obs2i=obs2i,
        )
        if len(final_circuit) > 0 and final_circuit[-1].name == "SHIFT_COORDS":
            final_circuit = final_circuit[:-1]
        return final_circuit

    def append(
        self,
        appended: Chunk | ChunkLoop | ChunkReflow,
    ) -> None:
        """Appends a chunk to the circuit being built.

        The input flows of the appended chunk must exactly match the open outgoing flows of the
        circuit so far.
        """
        if isinstance(appended, Chunk):
            self._append_chunk(chunk=appended)
        elif isinstance(appended, ChunkReflow):
            self._append_chunk_reflow(chunk_reflow=appended)
        elif isinstance(appended, ChunkLoop):
            self._append_chunk_loop(chunk_loop=appended)
        else:
            raise NotImplementedError(f"{appended=}")

    def _append_chunk_reflow(
        self,
        *,
        chunk_reflow: ChunkReflow,
    ) -> None:
        for ps in chunk_reflow.discard_in:
            if isinstance(ps, KeyedPauliMap):
                self.discarded_observables.add(ps.key)
        next_flows: dict[PauliMap | KeyedPauliMap, Union[Flow, Literal["discard"]]] = {}
        for output, inputs in chunk_reflow.out2in.items():
            measurements = set()
            centers = []
            flags = set()
            discarded = False
            for inp_key in inputs:
                if inp_key not in self.open_flows:
                    msg = [
                        f"Missing reflow input: {inp_key=}",
                        "Needed inputs {",
                    ]
                    for ps in inputs:
                        msg.append(f"    {ps}")
                    msg.append("}")
                    msg.append("Actual inputs {")
                    for ps in self.open_flows.keys():
                        msg.append(f"    {ps}")
                    msg.append("}")
                    raise ValueError("\n".join(msg))
                inp = self.open_flows[inp_key]
                if inp == "discard":
                    discarded = True
                else:
                    assert isinstance(inp, Flow)
                    assert not inp.start
                    measurements ^= frozenset(inp.measurement_indices)
                    centers.append(inp.center)
                    flags |= inp.flags

            next_flows[output] = (
                "discard"
                if discarded
                else Flow(
                    start=None,
                    end=(
                        output.pauli_string
                        if isinstance(output, KeyedPauliMap)
                        else output
                    ),
                    measurement_indices=tuple(sorted(measurements)),
                    obs_key=output.key if isinstance(output, KeyedPauliMap) else None,
                    flags=flags,
                    center=sum(centers) / len(centers),
                )
            )

        for k, v in self.open_flows.items():
            if k in chunk_reflow.removed_inputs:
                continue
            assert k not in next_flows
            next_flows[k] = v

        self.open_flows = next_flows

    def _append_chunk(
        self,
        *,
        chunk: Chunk,
    ) -> None:
        # Index any new locations.
        self.ensure_qubits_included(chunk.q2i.keys())

        # Ensure chunks are always separated by a TICK.
        if (
            len(self.circuit) > 0
            and self.circuit[-1].name != "TICK"
            and self.circuit[-1].name != "REPEAT"
        ):
            self.circuit.append("TICK")

        # Attach new flows to existing flows.
        next_flows, completed_flows = self._compute_next_flows(chunk=chunk)
        for ps in chunk.discarded_inputs:
            if isinstance(ps, KeyedPauliMap):
                self.discarded_observables.add(ps.key)
        for ps in chunk.discarded_outputs:
            if isinstance(ps, KeyedPauliMap):
                self.discarded_observables.add(ps.key)
        self.num_measurements += chunk.circuit.num_measurements
        self.open_flows = next_flows

        # Grow the compiled circuit.
        _append_to_circuit_with_reindexing(
            circuit=chunk.circuit,
            out=self.circuit,
            old_q2i=chunk.q2i,
            new_q2i=self.q2i,
            control_time_shift=True,
            obs2i={},
        )
        self._append_detectors(
            completed_flows=completed_flows,
        )

    def _append_chunk_loop(
        self,
        *,
        chunk_loop: ChunkLoop,
    ) -> None:
        past_circuit = self.circuit

        def compute_relative_flow_state():
            return {
                k: (
                    v
                    if isinstance(v, str)
                    else v.with_edits(
                        measurement_indices=[
                            m - self.num_measurements for m in v.measurement_indices
                        ]
                    )
                )
                for k, v in self.open_flows.items()
            }

        iteration_circuits = []
        measure_offset_start_of_loop = self.num_measurements
        prev_rel_flow_state = compute_relative_flow_state()
        while len(iteration_circuits) < chunk_loop.repetitions:
            # Perform an iteration the hard way.
            self.circuit = stim.Circuit()
            for chunk in chunk_loop.chunks:
                self.append(chunk)
            self.circuit.append("TICK")
            iteration_circuits.append(self.circuit)

            # Check if we can fold the rest.
            new_rel_flow_state = compute_relative_flow_state()
            has_pre_loop_measurement = any(
                m < measure_offset_start_of_loop
                for flow in self.open_flows.values()
                if isinstance(flow, Flow)
                for m in flow.measurement_indices
            )
            have_reached_steady_state = (
                not has_pre_loop_measurement
                and new_rel_flow_state == prev_rel_flow_state
            )
            if have_reached_steady_state:
                break
            prev_rel_flow_state = new_rel_flow_state

        # Found a repeating iteration.
        leftover_reps = chunk_loop.repetitions - len(iteration_circuits)
        if leftover_reps > 0:
            measurements_skipped = (
                iteration_circuits[-1].num_measurements * leftover_reps
            )

            # Fold identical repetitions at the end.
            while (
                len(iteration_circuits) > 1
                and iteration_circuits[-1] == iteration_circuits[-2]
            ):
                leftover_reps += 1
                iteration_circuits.pop()
            iteration_circuits[-1] *= leftover_reps + 1

            self.num_measurements += measurements_skipped
            self.open_flows = {
                k: v.with_edits(
                    measurement_indices=[
                        m + measurements_skipped for m in v.measurement_indices
                    ]
                )
                for k, v in self.open_flows.items()
            }

        # Fuse iterations that happened to be equal.
        self.circuit = past_circuit
        if (
            self.circuit
            and self.circuit[-1].name != "TICK"
            and self.circuit[-1].name != "REPEAT"
            and iteration_circuits
            and iteration_circuits[0]
            and iteration_circuits[0][0].name != "TICK"
        ):
            self.circuit.append("TICK")
        k = 0
        while k < len(iteration_circuits):
            k2 = k + 1
            while (
                k2 < len(iteration_circuits)
                and iteration_circuits[k2] == iteration_circuits[k]
            ):
                k2 += 1
            self.circuit += iteration_circuits[k] * (k2 - k)
            k = k2

    def _append_detectors(
        self,
        *,
        completed_flows: list[Flow],
    ):
        inserted_ops = stim.Circuit()

        # Dump observable changes.
        for key, flow in list(self.open_flows.items()):
            if (
                isinstance(key, KeyedPauliMap)
                and not isinstance(flow, str)
                and flow.measurement_indices
            ):
                targets = []
                for m in flow.measurement_indices:
                    targets.append(stim.target_rec(m - self.num_measurements))
                obs_index = self.o2i.setdefault(flow.obs_key, len(self.o2i))
                inserted_ops.append("OBSERVABLE_INCLUDE", targets, obs_index)
                self.open_flows[key] = flow.with_edits(measurement_indices=[])

        # Append detector and observable annotations for the completed flows.
        detector_pos_usage_counts = collections.Counter()
        for flow in completed_flows:
            targets = []
            for m in flow.measurement_indices:
                targets.append(stim.target_rec(m - self.num_measurements))
            if flow.obs_key is None:
                dt = detector_pos_usage_counts[flow.center]
                detector_pos_usage_counts[flow.center] += 1
                coords = (flow.center.real, flow.center.imag, dt) + tuple(
                    self.flow_to_extra_coords_func(flow)
                )
                inserted_ops.append("DETECTOR", targets, coords)
            else:
                obs_index = self.o2i.setdefault(flow.obs_key, len(self.o2i))
                inserted_ops.append("OBSERVABLE_INCLUDE", targets, obs_index)

        if inserted_ops:
            insert_index = len(self.circuit)
            while (
                insert_index > 0
                and self.circuit[insert_index - 1].num_measurements == 0
            ):
                insert_index -= 1
            self.circuit.insert(insert_index, inserted_ops)

        # Shift the time coordinate so future chunks' detectors are further along the time axis.
        det_offset = max(detector_pos_usage_counts.values(), default=0)
        if det_offset > 0:
            self.circuit.append("SHIFT_COORDS", [], (0, 0, det_offset))

    def _compute_next_flows(
        self,
        *,
        chunk: Chunk,
    ) -> tuple[
        dict[PauliMap | KeyedPauliMap, Union[Flow, Literal["discard"]]], list[Flow]
    ]:
        attached_flows, outgoing_discards = self._compute_attached_flows_and_discards(
            chunk=chunk
        )

        next_flows: dict[PauliMap | KeyedPauliMap, Union[Flow, Literal["discard"]]] = {}
        completed_flows: list[Flow] = []
        for flow in attached_flows:
            assert not flow.start
            if flow.end:
                next_flows[flow.key_end] = flow
            else:
                completed_flows.append(flow)

        for discarded in outgoing_discards:
            next_flows[discarded] = "discard"
        for discarded in chunk.discarded_outputs:
            assert discarded not in next_flows
            next_flows[discarded] = "discard"

        return next_flows, completed_flows

    def _compute_attached_flows_and_discards(
        self,
        *,
        chunk: Chunk,
    ) -> tuple[list[Flow], list[PauliMap]]:
        result: list[Flow] = []
        old_flows = dict(self.open_flows)

        # Drop existing flows explicitly discarded by the chunk.
        for discarded in chunk.discarded_inputs:
            old_flows.pop(discarded, None)
        outgoing_discards = []

        # Attach the chunk's flows to the existing flows.
        for new_flow in chunk.flows:
            prev = old_flows.pop(new_flow.key_start, None)
            if prev == "discard":
                # Okay, discard it.
                if new_flow.end:
                    outgoing_discards.append(new_flow.key_end)
            elif prev is not None:
                # Matched! Fuse them together.
                result.append(
                    prev.fuse_with_next_flow(
                        new_flow,
                        next_flow_measure_offset=self.num_measurements,
                    )
                )
            elif not new_flow.start:
                # Flow started inside the new chunk, so doesn't need to be matched.
                result.append(
                    new_flow.with_edits(
                        measurement_indices=[
                            m + self.num_measurements
                            for m in new_flow.measurement_indices
                        ]
                    )
                )
            else:
                # Failed to match. Describe the problem.
                lines = [
                    "A flow input wasn't satisfied.",
                    f"   Expected input: {new_flow.key_start}",
                    f"   Available inputs:",
                ]
                for prev_avail in old_flows.keys():
                    lines.append(f"       {prev_avail}")
                raise ValueError("\n".join(lines))

        # Check for any unmatched flows.
        dangling_flows: list[Flow] = [
            val for val in old_flows.values() if val != "discard"
        ]
        if dangling_flows:
            lines = ["Some flow outputs were unmatched when appending a new chunk:"]
            for flow in dangling_flows:
                lines.append(f"   {flow.key_end}")
            raise ValueError("\n".join(lines))

        return result, outgoing_discards


def _append_to_circuit_with_reindexing(
    *,
    circuit: stim.Circuit,
    old_q2i: dict[complex, int],
    new_q2i: dict[complex, int],
    obs2i: dict[int, int | Literal["discard"]],
    out: stim.Circuit,
    control_time_shift: bool,
) -> None:
    i2i = {i: new_q2i[q] for q, i in old_q2i.items()}

    det_offset_needed = 0
    for inst in circuit:
        if inst.name == "REPEAT":
            block = stim.Circuit()
            _append_to_circuit_with_reindexing(
                circuit=inst.body_copy(),
                old_q2i=old_q2i,
                new_q2i=new_q2i,
                out=block,
                control_time_shift=control_time_shift,
                obs2i=obs2i,
            )
            out.append(
                stim.CircuitRepeatBlock(repeat_count=inst.repeat_count, body=block)
            )
        elif inst.name == "QUBIT_COORDS":
            continue
        elif inst.name == "SHIFT_COORDS":
            if control_time_shift:
                args = inst.gate_args_copy()
                if len(args) > 2:
                    det_offset_needed -= args[2]
                    out.append("SHIFT_COORDS", [], [0, 0, args[2]])
            else:
                out.append(inst)
        elif inst.name == "OBSERVABLE_INCLUDE":
            (obs_index,) = inst.gate_args_copy()
            obs_index = int(round(obs_index))
            obs_index = obs2i.get(obs_index, obs_index)
            if obs_index != "discard":
                out.append("OBSERVABLE_INCLUDE", inst.targets_copy(), obs_index)
        elif inst.name == "DETECTOR":
            args = inst.gate_args_copy()
            t = args[2] if len(args) > 2 else 0
            det_offset_needed = max(det_offset_needed, t + 1)
            out.append(inst)
        else:
            targets = []
            for t in inst.targets_copy():
                if t.is_qubit_target:
                    targets.append(i2i[t.value])
                elif t.is_x_target:
                    targets.append(stim.target_x(i2i[t.value]))
                elif t.is_y_target:
                    targets.append(stim.target_y(i2i[t.value]))
                elif t.is_z_target:
                    targets.append(stim.target_z(i2i[t.value]))
                elif t.is_combiner:
                    targets.append(t)
                elif t.is_measurement_record_target:
                    targets.append(t)
                elif t.is_sweep_bit_target:
                    targets.append(t)
                else:
                    raise NotImplementedError(f"{inst=}")
            out.append(inst.name, targets, inst.gate_args_copy())

    if control_time_shift and det_offset_needed > 0:
        out.append("SHIFT_COORDS", [], (0, 0, det_offset_needed))


def compile_chunks_into_circuit(
    chunks: list[Union[Chunk, ChunkLoop]],
    *,
    add_mpp_boundaries: bool = False,
    flow_to_extra_coords_func: Callable[[Flow], Iterable[float]] = lambda _: (),
) -> stim.Circuit:
    compiler = ChunkCompiler(flow_to_extra_coords_func=flow_to_extra_coords_func)

    if add_mpp_boundaries and chunks:
        compiler.append(chunks[0].mpp_init_chunk())

    for k, chunk in enumerate(chunks):
        try:
            compiler.append(chunk)
        except (ValueError, NotImplementedError) as ex:
            raise type(ex)(
                f"Encountered error while appending chunk index {k}:\n{ex}"
            ) from ex

    if add_mpp_boundaries and chunks:
        compiler.append(chunks[-1].mpp_end_chunk())

    return compiler.finish_circuit()
