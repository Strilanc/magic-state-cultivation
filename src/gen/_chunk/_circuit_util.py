import collections
from typing import Callable, Iterable, TYPE_CHECKING, Union

import stim

from gen._chunk._stabilizer_code import StabilizerCode

if TYPE_CHECKING:
    import gen


def circuit_with_xz_flipped(circuit: stim.Circuit) -> stim.Circuit:
    result = stim.Circuit()
    for inst in circuit:
        if isinstance(inst, stim.CircuitRepeatBlock):
            result.append(
                stim.CircuitRepeatBlock(
                    body=circuit_with_xz_flipped(inst.body_copy()),
                    repeat_count=inst.repeat_count,
                )
            )
        else:
            other = stim.gate_data(inst.name).hadamard_conjugated(unsigned=True).name
            if other is None:
                raise NotImplementedError(f"{inst=}")
            result.append(
                stim.CircuitInstruction(
                    other, inst.targets_copy(), inst.gate_args_copy()
                )
            )
    return result


def circuit_to_dem_target_measurement_records_map(
    circuit: stim.Circuit,
) -> dict[stim.DemTarget, list[int]]:
    result = {}
    for k in range(circuit.num_observables):
        result[stim.target_logical_observable_id(k)] = []
    num_d = 0
    num_m = 0
    for inst in circuit.flattened():
        if inst.name == "DETECTOR":
            result[stim.target_relative_detector_id(num_d)] = [
                num_m + t.value for t in inst.targets_copy()
            ]
            num_d += 1
        elif inst.name == "OBSERVABLE_INCLUDE":
            result[
                stim.target_logical_observable_id(int(inst.gate_args_copy()[0]))
            ].extend(num_m + t.value for t in inst.targets_copy())
        else:
            c = stim.Circuit()
            c.append(inst)
            num_m += c.num_measurements
    return result


def count_measurement_layers(circuit: stim.Circuit) -> int:
    saw_measurement = False
    result = 0
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            result += (
                count_measurement_layers(instruction.body_copy())
                * instruction.repeat_count
            )
        elif isinstance(instruction, stim.CircuitInstruction):
            saw_measurement |= stim.gate_data(instruction.name).produces_measurements
            if instruction.name == "TICK":
                result += saw_measurement
                saw_measurement = False
        else:
            raise NotImplementedError(f"{instruction=}")
    result += saw_measurement
    return result


def gate_counts_for_circuit(circuit: stim.Circuit) -> collections.Counter[str]:
    """Determines gates used by a circuit, disambiguating MPP/feedback cases.

    MPP instructions are expanded into what they actually measure, such as
    "MXX" for MPP X1*X2 and "MXYZ" for MPP X4*Y5*Z7.

    Feedback instructions like `CX rec[-1] 0` become the gate "feedback".

    Sweep instructions like `CX sweep[2] 0` become the gate "sweep".
    """
    ANNOTATION_OPS = {
        "DETECTOR",
        "OBSERVABLE_INCLUDE",
        "QUBIT_COORDS",
        "SHIFT_COORDS",
        "TICK",
        "MPAD",
    }

    out = collections.Counter()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            for k, v in gate_counts_for_circuit(instruction.body_copy()).items():
                out[k] += v * instruction.repeat_count

        elif instruction.name in ["CX", "CY", "CZ", "XCZ", "YCZ"]:
            targets = instruction.targets_copy()
            for k in range(0, len(targets), 2):
                if (
                    targets[k].is_measurement_record_target
                    or targets[k + 1].is_measurement_record_target
                ):
                    out["feedback"] += 1
                elif (
                    targets[k].is_sweep_bit_target or targets[k + 1].is_sweep_bit_target
                ):
                    out["sweep"] += 1
                else:
                    out[instruction.name] += 1

        elif instruction.name == "MPP":
            op = "M"
            targets = instruction.targets_copy()
            is_continuing = True
            for t in targets:
                if t.is_combiner:
                    is_continuing = True
                    continue
                p = (
                    "X"
                    if t.is_x_target
                    else "Y" if t.is_y_target else "Z" if t.is_z_target else "?"
                )
                if is_continuing:
                    op += p
                    is_continuing = False
                else:
                    if op == "MZ":
                        op = "M"
                    out[op] += 1
                    op = "M" + p
            if op:
                if op == "MZ":
                    op = "M"
                out[op] += 1

        elif stim.gate_data(instruction.name).is_two_qubit_gate:
            out[instruction.name] += len(instruction.targets_copy()) // 2
        elif (
            instruction.name in ANNOTATION_OPS
            or instruction.name == "E"
            or instruction.name == "ELSE_CORRELATED_ERROR"
        ):
            out[instruction.name] += 1
        else:
            out[instruction.name] += len(instruction.targets_copy())

    return out


def gates_used_by_circuit(circuit: stim.Circuit) -> set[str]:
    """Determines gates used by a circuit, disambiguating MPP/feedback cases.

    MPP instructions are expanded into what they actually measure, such as
    "MXX" for MPP X1*X2 and "MXYZ" for MPP X4*Y5*Z7.

    Feedback instructions like `CX rec[-1] 0` become the gate "feedback".

    Sweep instructions like `CX sweep[2] 0` become the gate "sweep".
    """
    out = set()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            out |= gates_used_by_circuit(instruction.body_copy())

        elif instruction.name in ["CX", "CY", "CZ", "XCZ", "YCZ"]:
            targets = instruction.targets_copy()
            for k in range(0, len(targets), 2):
                if (
                    targets[k].is_measurement_record_target
                    or targets[k + 1].is_measurement_record_target
                ):
                    out.add("feedback")
                elif (
                    targets[k].is_sweep_bit_target or targets[k + 1].is_sweep_bit_target
                ):
                    out.add("sweep")
                else:
                    out.add(instruction.name)

        elif instruction.name == "MPP":
            op = "M"
            targets = instruction.targets_copy()
            is_continuing = True
            for t in targets:
                if t.is_combiner:
                    is_continuing = True
                    continue
                p = (
                    "X"
                    if t.is_x_target
                    else "Y" if t.is_y_target else "Z" if t.is_z_target else "?"
                )
                if is_continuing:
                    op += p
                    is_continuing = False
                else:
                    if op == "MZ":
                        op = "M"
                    out.add(op)
                    op = "M" + p
            if op:
                if op == "MZ":
                    op = "M"
                out.add(op)

        else:
            out.add(instruction.name)

    return out


def stim_circuit_with_transformed_coords(
    circuit: stim.Circuit, transform: Callable[[complex], complex]
) -> stim.Circuit:
    """Returns an equivalent circuit, but with the qubit and detector position metadata modified.
    The "position" is assumed to be the first two coordinates. These are mapped to the real and
    imaginary values of a complex number which is then transformed.

    Note that `SHIFT_COORDS` instructions that modify the first two coordinates are not supported.
    This is because supporting them requires flattening loops, or promising that the given
    transformation is affine.

    Args:
        circuit: The circuit with qubits to reposition.
        transform: The transformation to apply to the positions. The positions are given one by one
            to this method, as complex numbers. The method returns the new complex number for the
            position.

    Returns:
        The transformed circuit.
    """
    result = stim.Circuit()
    for instruction in circuit:
        if isinstance(instruction, stim.CircuitInstruction):
            if instruction.name == "QUBIT_COORDS" or instruction.name == "DETECTOR":
                args = list(instruction.gate_args_copy())
                while len(args) < 2:
                    args.append(0)
                c = transform(args[0] + args[1] * 1j)
                args[0] = c.real
                args[1] = c.imag
                result.append(instruction.name, instruction.targets_copy(), args)
                continue
            if instruction.name == "SHIFT_COORDS":
                args = instruction.gate_args_copy()
                if any(args[:2]):
                    raise NotImplementedError(
                        f"Shifting first two coords: {instruction=}"
                    )

        if isinstance(instruction, stim.CircuitRepeatBlock):
            result.append(
                stim.CircuitRepeatBlock(
                    repeat_count=instruction.repeat_count,
                    body=stim_circuit_with_transformed_coords(
                        instruction.body_copy(), transform
                    ),
                )
            )
            continue

        result.append(instruction)
    return result


def stim_circuit_with_transformed_moments(
    circuit: stim.Circuit, *, moment_func: Callable[[stim.Circuit], stim.Circuit]
) -> stim.Circuit:
    """Applies a transformation to regions of a circuit separated by TICKs and blocks.

    For example, in this circuit:

        H 0
        X 0
        TICK

        H 1
        X 1
        REPEAT 100 {
            H 2
            X 2
        }
        H 3
        X 3

        TICK
        H 4
        X 4

    `moment_func` would be called five times, each time with one of the H and X instruction pairs.
    The result from the method would then be substituted into the circuit, replacing each of the H
    and X instruction pairs.

    Args:
        circuit: The circuit to return a transformed result of.
        moment_func: The transformation to apply to regions of the circuit. Returns a new circuit
            for the result.

    Returns:
        A transformed circuit.
    """

    result = stim.Circuit()
    current_moment = stim.Circuit()

    for instruction in circuit:
        if isinstance(instruction, stim.CircuitRepeatBlock):
            # Implicit tick at transition into REPEAT?
            if current_moment:
                result += moment_func(current_moment)
                current_moment.clear()

            transformed_body = stim_circuit_with_transformed_moments(
                instruction.body_copy(), moment_func=moment_func
            )
            result.append(
                stim.CircuitRepeatBlock(
                    repeat_count=instruction.repeat_count, body=transformed_body
                )
            )
        elif (
            isinstance(instruction, stim.CircuitInstruction)
            and instruction.name == "TICK"
        ):
            # Explicit tick. Process even if empty.
            result += moment_func(current_moment)
            result.append("TICK")
            current_moment.clear()
        else:
            current_moment.append(instruction)

    # Implicit tick at end of circuit?
    if current_moment:
        result += moment_func(current_moment)

    return result


def circuit_to_cycle_code_slices(
    circuit: stim.Circuit,
) -> dict[int, "gen.StabilizerCode"]:
    from gen._chunk._stabilizer_code import StabilizerCode
    from gen._chunk._pauli_map import PauliMap
    from gen._chunk._patch import Patch

    t = 0
    ticks = set()
    for inst in circuit.flattened():
        if inst.name == "TICK":
            t += 1
        elif inst.name in ["R", "RX"]:
            if t - 1 not in ticks and t - 2 not in ticks:
                ticks.add(max(t - 1, 0))
        elif inst.name in ["M", "MX"]:
            ticks.add(t)

    regions = circuit.detecting_regions(ticks=ticks)
    layers: dict[int, list[tuple[stim.DemTarget, stim.PauliString]]] = (
        collections.defaultdict(list)
    )
    for dem_target, tick2paulis in regions.items():
        for tick, pauli_string in tick2paulis.items():
            layers[tick].append((dem_target, pauli_string))

    i2q = {k: r + i * 1j for k, (r, i) in circuit.get_final_qubit_coordinates().items()}

    codes = {}
    for tick, layer in sorted(layers.items()):
        obs = []
        tiles = []
        for dem_target, pauli_string in layer:
            pauli_map = PauliMap(
                {i2q[q]: "_XYZ"[pauli_string[q]] for q in pauli_string.pauli_indices()}
            )
            if dem_target.is_relative_detector_id():
                tiles.append(
                    pauli_map.to_tile().with_edits(flags={str(dem_target.val)})
                )
            else:
                obs.append(pauli_map)
        codes[tick] = StabilizerCode(stabilizers=Patch(tiles), logicals=obs)

    return codes


def find_d1_error(
    obj: Union[stim.Circuit, stim.DetectorErrorModel]
) -> stim.ExplainedError | stim.DemInstruction | None:
    circuit: stim.Circuit | None
    dem: stim.DetectorErrorModel
    if isinstance(obj, stim.Circuit):
        circuit = obj
        dem = circuit.detector_error_model()
    elif isinstance(obj, stim.DetectorErrorModel):
        circuit = None
        dem = obj
    else:
        raise NotImplementedError(f"{obj=}")

    for inst in dem:
        if inst.type == "error":
            dets = set()
            obs = set()
            for target in inst.targets_copy():
                if target.is_relative_detector_id():
                    dets ^= {target.val}
                elif target.is_logical_observable_id():
                    obs ^= {target.val}
            dets = frozenset(dets)
            obs = frozenset(obs)
            if obs and not dets:
                if circuit is None:
                    return inst
                filter_det = stim.DetectorErrorModel()
                filter_det.append(inst)
                return circuit.explain_detector_error_model_errors(
                    dem_filter=filter_det, reduce_to_one_representative_error=True
                )[0]

    return None


def find_d2_error(
    obj: Union[stim.Circuit, stim.DetectorErrorModel]
) -> list[stim.ExplainedError] | stim.DetectorErrorModel | None:
    d1 = find_d1_error(obj)
    if d1 is not None:
        if isinstance(d1, stim.DemInstruction):
            result = stim.DetectorErrorModel()
            result.append(d1)
            return result
        return [d1]

    if isinstance(obj, stim.Circuit):
        circuit = obj
        dem = circuit.detector_error_model()
    elif isinstance(obj, stim.DetectorErrorModel):
        circuit = None
        dem = obj
    else:
        raise NotImplementedError(f"{obj=}")

    seen = {}
    for inst in dem.flattened():
        if inst.type == "error":
            dets = set()
            obs = set()
            for target in inst.targets_copy():
                if target.is_relative_detector_id():
                    dets ^= {target.val}
                elif target.is_logical_observable_id():
                    obs ^= {target.val}
            dets = frozenset(dets)
            obs = frozenset(obs)
            if dets not in seen:
                seen[dets] = (obs, inst)
            elif seen[dets][0] != obs:
                filter_det = stim.DetectorErrorModel()
                filter_det.append(inst)
                filter_det.append(seen[dets][1])
                if circuit is None:
                    return filter_det
                return circuit.explain_detector_error_model_errors(
                    dem_filter=filter_det,
                    reduce_to_one_representative_error=True,
                )


def verify_distance_is_at_least_2(
    obj: Union[stim.Circuit, stim.DetectorErrorModel, "gen.StabilizerCode"]
):
    __tracebackhide__ = True
    if isinstance(obj, StabilizerCode):
        obj.verify_distance_is_at_least_2()
        return
    err = find_d1_error(obj)
    if err is not None:
        raise ValueError(f"Found a distance 1 error: {err}")


def verify_distance_is_at_least_3(
    obj: Union[stim.Circuit, stim.DetectorErrorModel, "gen.StabilizerCode"]
):
    __tracebackhide__ = True
    err = find_d2_error(obj)
    if err is not None:
        raise ValueError(f"Found a distance {len(err)} error: {err}")
