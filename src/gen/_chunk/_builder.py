from typing import Iterable, Callable, Any, TYPE_CHECKING, Union, Sequence

import stim

from ._measurement_tracker import MeasurementTracker
from ._pauli_map import PauliMap
from ._keyed_pauli_map import KeyedPauliMap
from ._complex_util import sorted_complex

if TYPE_CHECKING:
    import gen


_SWAP_CONJUGATED_MAP = {
    "XCZ": "CX",
    "YCZ": "CY",
    "YCX": "XCY",
    "SWAPCX": "CXSWAP",
}


class Builder:
    """Helper class for building stim circuits.

    Handles qubit indexing (complex -> int conversion).
    Handles measurement tracking (naming results and referring to them by name).
    """

    def __init__(
        self,
        *,
        q2i: dict[complex, int],
        circuit: stim.Circuit,
        tracker: MeasurementTracker,
    ):
        """
        Args:
            q2i: Maps qubit positions to qubit indices in the circuit.
            circuit: The circuit to append operations into.
            tracker: Where to lookup measurement record indices by key.
        """
        self.q2i = q2i
        self.circuit = circuit
        self.tracker = tracker

    @staticmethod
    def for_qubits(
        qubits: Iterable[complex],
    ) -> "Builder":
        """Creates a builder for a circuit operating on the given positions.

        Args:
            qubits: The qubit positions operated on by the circuit. The builder
                will automatically index these, and record them into the circuit
                using QUBIT_COORDS instructions.

        Returns:
            A new builder, with an empty circuit (except for QUBIT_COORDS
            instructions) and an empty measurement tracker.
        """
        q2i = {q: i for i, q in enumerate(sorted_complex(set(qubits)))}
        circuit = stim.Circuit()

        for q, i in q2i.items():
            circuit.append("QUBIT_COORDS", [i], [q.real, q.imag])

        return Builder(
            q2i=q2i,
            circuit=circuit,
            tracker=MeasurementTracker(),
        )

    def lookup_recs(self, keys: Iterable[Any]) -> list[int]:
        """Looks up measurement record indices by key.

        These keys are specified when appending the measurement into the circuit
        via the builder's append method. By default they are the position of the
        qubit, but can be customized by the append call.
        """
        return self.tracker.lookup_recs(keys)

    def append(
        self,
        gate: str,
        targets: Iterable[Union[complex, Sequence[complex], "gen.PauliMap", Any]] = (),
        *,
        arg: float | Iterable[float] | None = None,
        measure_key_func: (
            Callable[[Union[complex, tuple[complex, complex], "gen.PauliMap"]], Any]
            | None
        ) = lambda e: e,
    ) -> None:
        """Appends an instruction to the builder's circuit.

        This method differs from `stim.Circuit.append` in the following ways:

        1) It targets qubits by position instead of by index. Also, it takes two
        qubit targets as pairs instead of interleaved. For example, instead of
        saying

            a = builder.q2i[5 + 1j]
            b = builder.q2i[5]
            c = builder.q2i[0]
            d = builder.q2i[1j]
            builder.circuit.append('CZ', [a, b, c, d])

        you can say

            builder.append('CZ', [(5+1j, 5), (0, 1j)])

        2) It canonicalizes. For example, it will sort the targets of an H gate,
        sort the pairs of targets of a CX gate, sort the targets within each
        pair for symmetric gates like CZ, replace less common gates like XCZ
        with more common gates like CX by reversing the order within each target
        pair, not bothering to append the instruction when it does nothing due
        to having zero targets, and so forth. Canonicalization makes the form of
        the final circuit stable, despite things like python's `set` data
        structure having inconsistent iteration orders, so that the output is
        easier to unit test and more viable to store under source control.

        3) It tracks measurements. When appending a measurement, its index is
        stored in the measurement tracker keyed by the position of the qubit
        being measured (or by a custom key, if measure_key_func is specified).
        This makes it much easier to refer to measurements later on, when
        defining detectors or flows.

        Args:
            gate: The name of the gate to append, such as "H" or "M" or "CX".
            targets: The qubit positions that the gate operates on. For single
                qubit gates like H or M this should be an iterable of complex
                numbers. For two qubit gates like CX or MXX it should be an
                iterable of pairs of complex numbers. For MPP it should be an
                iterable of gen.PauliMap instances.
            arg: Optional. The parens argument or arguments used for the gate
                instruction. For example, for a measurement gate, this is the
                probability of the incorrect result being reported.
            measure_key_func: Customizes the keys used to track the indices of
                measurement results. By default, measurements are keyed by
                position, but thus won't work if a circuit measures the same
                qubit multiple times. This function can transform that position
                into a different value (for example, you might set
                `measure_key_func=lambda pos: (pos, 'first_cycle')` for
                measurements during the first cycle of the circuit.
        """
        __tracebackhide__ = True
        data = stim.gate_data(gate)

        if data.name == "TICK":
            assert not targets
            assert arg is None
            self.circuit.append("TICK")
        elif data.name == "SHIFT_COORDS":
            assert arg and not targets
            self.circuit.append("SHIFT_COORDS", [], arg)
        elif data.name == "OBSERVABLE_INCLUDE" or data.name == "DETECTOR":
            self.circuit.append(
                data.name,
                self.tracker.current_measurement_record_targets_for(targets),
                arg,
            )
        elif data.name == "MPP":
            if not targets:
                return
            if arg == 0:
                arg = None
            if isinstance(targets, PauliMap) or isinstance(targets, KeyedPauliMap):
                raise ValueError(
                    f"{gate=} but {targets=} isn't a list of gen.PauliMap."
                )
            for target in targets:
                if not isinstance(target, PauliMap) and not isinstance(
                    target, KeyedPauliMap
                ):
                    raise ValueError(f"{gate=} but {target=} isn't a gen.PauliMap.")

            # Canonicalize qubit ordering of the pauli strings.
            stim_targets = []
            for pauli_map in targets:
                if not pauli_map.qubits:
                    raise NotImplementedError(
                        f"Attempted to measure empty pauli string {pauli_map=}."
                    )
                for q in sorted_complex(pauli_map.qubits):
                    stim_targets.append(stim.target_pauli(self.q2i[q], pauli_map[q]))
                    stim_targets.append(stim.target_combiner())
                stim_targets.pop()

            self.circuit.append(gate, stim_targets, arg)

            for pauli_map in targets:
                if measure_key_func is None:
                    self.tracker.next_measurement_index += 1
                else:
                    self.tracker.record_measurement(measure_key_func(pauli_map))

        elif data.is_two_qubit_gate:
            if not targets:
                return
            for target in targets:
                if not hasattr(target, "__len__") or len(target) != 2:
                    raise ValueError(
                        f"{gate=} is a two-qubit gate, but {target=} isn't a pair of complex numbers."
                    )
                a, b = target
                if a not in self.q2i or b not in self.q2i:
                    if a in self.q2i:
                        raise ValueError(
                            f"Tried to apply {gate=} to {target=}, but {b!r} isn't a known qubit position."
                        )
                    elif b in self.q2i:
                        raise ValueError(
                            f"Tried to apply {gate=} to {target=}, but {a!r} isn't a known qubit position."
                        )
                    else:
                        raise ValueError(
                            f"Tried to apply {gate=} to {target=}, but neither {a!r} nor {b!r} are known qubit positions."
                        )

            # Canonicalize gate and target pairs.
            targets = [tuple(pair) for pair in targets]
            targets = sorted(
                targets, key=lambda pair: (self.q2i[pair[0]], self.q2i[pair[1]])
            )

            if data.is_symmetric_gate:
                targets = [sorted(pair, key=self.q2i.__getitem__) for pair in targets]
            elif gate in _SWAP_CONJUGATED_MAP:
                targets = [pair[::-1] for pair in targets]
                gate = _SWAP_CONJUGATED_MAP[gate]

            self.circuit.append(
                gate, [self.q2i[q] for pair in targets for q in pair], arg
            )

            # Record both orderings.
            if data.produces_measurements:
                for a, b in targets:
                    if measure_key_func is None:
                        self.tracker.next_measurement_index += 1
                    else:
                        k1 = measure_key_func((a, b))
                        k2 = measure_key_func((b, a))
                        self.tracker.record_measurement(key=k1)
                        if k1 != k2:
                            self.tracker.make_measurement_group([k1], key=k2)

        elif data.is_single_qubit_gate:
            if not targets:
                return
            for target in targets:
                if target not in self.q2i:
                    raise ValueError(
                        f"{gate=} is a single-qubit gate, but {target=} isn't in indexed."
                    )
            targets = sorted(targets, key=self.q2i.__getitem__)

            self.circuit.append(gate, [self.q2i[q] for q in targets], arg)
            if data.produces_measurements:
                for target in targets:
                    if measure_key_func is None:
                        self.tracker.next_measurement_index += 1
                    else:
                        self.tracker.record_measurement(key=measure_key_func(target))

        else:
            raise NotImplementedError(f"{gate=}")

    def demolition_measure_with_feedback_passthrough(
        self,
        xs: Iterable[complex] = (),
        ys: Iterable[complex] = (),
        zs: Iterable[complex] = (),
        *,
        measure_key_func: Callable[[complex], Any] = lambda e: e,
    ) -> None:
        """Performs demolition measurements that look like measurements w.r.t. detectors.

        This is done by adding feedback operations that flip the demolished qubits depending
        on the measurement result. This feedback can then later be removed using
        stim.Circuit.with_inlined_feedback. The benefit is that it can be easier to
        programmatically create the detectors using the passthrough measurements, and
        then they can be automatically converted.
        """
        self.append("MX", xs, measure_key_func=measure_key_func)
        self.append("MY", ys, measure_key_func=measure_key_func)
        self.append("MZ", zs, measure_key_func=measure_key_func)
        self.append("TICK")
        self.append("RX", xs)
        self.append("RY", ys)
        self.append("RZ", zs)
        for qs, b in [(xs, "Z"), (ys, "X"), (zs, "X")]:
            for q in qs:
                self.classical_paulis(
                    control_keys=[measure_key_func(q)],
                    targets=[q],
                    basis=b,
                )

    def classical_paulis(
        self, *, control_keys: Iterable[Any], targets: Iterable[complex], basis: str
    ) -> None:
        gate = f"C{basis}"
        indices = [self.q2i[q] for q in sorted_complex(targets)]
        for rec in self.tracker.current_measurement_record_targets_for(control_keys):
            for i in indices:
                self.circuit.append(gate, [rec, i])
