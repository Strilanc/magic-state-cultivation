import stim

import gen


def make_chunk_d3_init_degenerate_teleport() -> gen.Chunk:
    chunk = gen.Chunk.from_circuit_with_mpp_boundaries(stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(1, 1) 2
        QUBIT_COORDS(1, 2) 3
        QUBIT_COORDS(2, 0) 4
        QUBIT_COORDS(2, 1) 5
        QUBIT_COORDS(2, 2) 6
        QUBIT_COORDS(2, 3) 7
        QUBIT_COORDS(3, 0) 8
        QUBIT_COORDS(3, 1) 9
        QUBIT_COORDS(3, 2) 10
        QUBIT_COORDS(3, 3) 11
        QUBIT_COORDS(3, 4) 12
        QUBIT_COORDS(4, 0) 13
        QUBIT_COORDS(4, 1) 14
        QUBIT_COORDS(4, 2) 15
        QUBIT_COORDS(4, 3) 16
        #!pragma POLYGON(0,0,1,0.25) 10 15 8 5
        #!pragma POLYGON(0,1,0,0.25) 7 10 5 2
        #!pragma POLYGON(1,0,0,0.25) 7 12 15 10
        #!pragma POLYGON(1,0,0,0.25) 2 5 8 0
        TICK
        R 0 8 15 10 5 2 12 7 6 4 16 14
        RX 3 1 9 11
        TICK
        CX 3 6 1 4 11 16 9 14
        TICK
        CX 1 2 4 5 9 10 14 15 6 7 11 12
        TICK
        CX 1 0 9 5 11 7 6 10 4 8
        TICK
        CX 3 2 9 8 11 10 6 5 16 15
        TICK
        CX 3 6 1 4 9 14 11 16
        TICK
        M 6 4 16 14
        MX 3 1 9 11
        DETECTOR(2, 2, 0) rec[-8]
        DETECTOR(2, 0, 0) rec[-7]
        DETECTOR(4, 3, 0) rec[-6]
        DETECTOR(4, 1, 0) rec[-5]
        TICK
        R 6 4 16
        RX 3 1 9 11 14 13
        TICK
        CX 3 6 1 4 11 16 9 5 14 15
        TICK
        CX 3 2 9 8 11 10 6 5 16 15 13 14
        TICK
        CX 1 2 4 5 9 10 6 7 11 12 15 14
        TICK
        CX 1 0 11 7 6 10 4 8 9 14
        TICK
        CX 3 6 1 4 11 16 14 13
        TICK
        S_DAG 12
        TICK
        M 16 4 6
        MX 14 9 15 12 1 3 11
        DETECTOR(4, 3, 1) rec[-10]
        DETECTOR(2, 0, 1) rec[-9]
        DETECTOR(2, 2, 1) rec[-8]
        DETECTOR(4, 1, 1) rec[-7]
        DETECTOR(4, 2, 1) rec[-5]
        DETECTOR(3, 1, 1) rec[-5] rec[-6] rec[-12]
        DETECTOR(1, 0, 1) rec[-3] rec[-13]
        DETECTOR(1, 2, 1) rec[-2] rec[-14]
        DETECTOR(3, 3, 1) rec[-1] rec[-11]
        TICK
        #!pragma POLYGON(0,0,1,0.25) 13 8 5 10
        #!pragma POLYGON(0,1,0,0.25) 7 10 5 2
        #!pragma POLYGON(1,0,0,0.25) 2 5 8 0
        TICK
        MPP X7*X10*X5*X2
        DETECTOR(1, 2, 2) rec[-1] rec[-3]
        TICK
        MPP X13*X8*X5*X10
        DETECTOR(3, 1, 3) rec[-1] rec[-8] rec[-9]
        TICK
        MPP X8*X0*X2*X5
        DETECTOR(1, 0, 4) rec[-1] rec[-6]
        TICK
        MPP Z7*Z10*Z5*Z2
        DETECTOR(2, 3, 5) rec[-1]
        TICK
        MPP Z8*Z0*Z2*Z5
        DETECTOR(3, 0, 6) rec[-1]
        TICK
        MPP Z13*Z8*Z5*Z10
        DETECTOR(4, 0, 7) rec[-1]
        TICK
        MPP Y0*Y8*Y13*Y10*Y7*Y2*Y5
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-11] rec[-12] rec[-14] rec[-15] rec[-17] rec[-18] rec[-20] rec[-22] rec[-23]
    """))
    return chunk.with_flag_added_to_all_flows('stage=inject')


def make_chunk_d3_init_bell_pair_growth() -> gen.Chunk:
    chunk = gen.Chunk.from_circuit_with_mpp_boundaries(stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(1, 1) 2
        QUBIT_COORDS(1, 2) 3
        QUBIT_COORDS(2, 0) 4
        QUBIT_COORDS(2, 1) 5
        QUBIT_COORDS(2, 2) 6
        QUBIT_COORDS(2, 3) 7
        QUBIT_COORDS(3, 0) 8
        QUBIT_COORDS(3, 1) 9
        QUBIT_COORDS(3, 2) 10
        QUBIT_COORDS(4, 0) 11
        #!pragma POLYGON(0,0,1,0.25) 8 11 10 5
        #!pragma POLYGON(0,1,0,0.25) 7 10 5 2
        #!pragma POLYGON(1,0,0,0.5) 2 5 8 0
        #!pragma POLYGON(1,0,1,0.25) 2 7
        #!pragma POLYGON(1,0,1,0.25) 5 10
        #!pragma POLYGON(1,0,1,0.25) 11 8
        TICK
        R 6 2 7 5 10 11 4
        RX 0 3 9 8 1
        TICK
        CX 3 6 9 5 8 11 1 4
        S_DAG 0
        TICK
        CX 6 7 3 2 9 10 8 4 0 1
        TICK
        CX 7 6 10 9 2 1 5 4
        TICK
        CX 2 3 4 8 1 0
        TICK
        CX 1 2 4 5
        TICK
        CX 1 4
        TICK
        M 3 6 9 4
        MX 1
        CX rec[-2] 0 rec[-2] 7
        DETECTOR(1, 2, 0) rec[-5]
        DETECTOR(2, 2, 0) rec[-4]
        DETECTOR(3, 1, 0) rec[-3]
        TICK
        MPP X0*X8*X5*X2
        DETECTOR(1, 0, 1) rec[-1] rec[-2]
        TICK
        MPP X2*X5*X10*X7
        DETECTOR(1, 1, 2) rec[-1]
        TICK
        MPP X8*X11*X10*X5
        DETECTOR(3, 0, 3) rec[-1]
        TICK
        MPP Z0*Z8*Z5*Z2
        DETECTOR(2, 0, 4) rec[-1]
        TICK
        MPP Z2*Z5*Z10*Z7
        DETECTOR(2, 0, 5) rec[-1]
        TICK
        MPP Z8*Z11*Z10*Z5
        DETECTOR(3, 0, 6) rec[-1]
        TICK
        MPP Y0*Y8*Y11*Y10*Y7*Y5*Y2
        OBSERVABLE_INCLUDE(0) rec[-1]
    """))
    return chunk.with_flag_added_to_all_flows('stage=inject')


def make_chunk_d3_init_unitary() -> gen.Chunk:
    chunk = gen.Chunk.from_circuit_with_mpp_boundaries(stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(1, 1) 2
        QUBIT_COORDS(1, 2) 3
        QUBIT_COORDS(1, 3) 4
        QUBIT_COORDS(2, 0) 5
        QUBIT_COORDS(2, 1) 6
        QUBIT_COORDS(2, 2) 7
        QUBIT_COORDS(2, 3) 8
        QUBIT_COORDS(3, 0) 9
        QUBIT_COORDS(3, 1) 10
        QUBIT_COORDS(3, 2) 11
        QUBIT_COORDS(4, 0) 12
        QUBIT_COORDS(4, 1) 13
        #!pragma POLYGON(0,0,1,0.25) 9 13 11 6
        #!pragma POLYGON(0,1,0,0.25) 2 6 11 3
        #!pragma POLYGON(1,0,0,0.25) 1 9 6 2
        TICK
        R 10 7 5
        RX 6 9 11 2 1 3 13 0 4 12
        TICK
        CX 6 10 9 5 11 7
        TICK
        TICK
        CX 6 7 9 10
        TICK
        CX 6 5 11 10
        TICK
        CX 1 5 3 7 13 10
        TICK
        CX 5 1 7 3 10 13
        TICK
        CX 2 3 12 13
        TICK
        CX 2 1 13 12
        TICK
        CX 3 2
        TICK
        CX 1 2
        TICK
        S_DAG 2
        TICK
        CX 1 2
        TICK
        CX 3 2
        TICK
        CX 0 1 4 3
        TICK
        CX 1 0 3 4
        TICK
        #!pragma POLYGON(0,0,1,0.25) 12 9 6 11
        #!pragma POLYGON(0,1,0,0.25) 8 11 6 2
        #!pragma POLYGON(0,1,0,0.25) 2 6 11 4
        #!pragma POLYGON(1,0,0,0.25) 2 6 9 0
        TICK
        R 5 13 7 8
        RX 1 10 3
        TICK
        CX 3 7 1 5 10 13 4 8
        TICK
        CX 10 6 7 11 1 0 5 9 8 4
        TICK
        CX 1 2 5 6 10 11 7 8
        TICK
        CX 10 9 13 12 3 2 7 6
        TICK
        CX 9 10 12 13 2 3 6 7
        TICK
        CX 6 10 11 7 0 1 9 5
        TICK
        CX 2 1 6 5 11 10 8 7
        TICK
        CX 3 7 1 5 10 13
        TICK
        M 5 13 7 4
        MX 1 10 3
        DETECTOR(2, 0, 0) rec[-7]
        DETECTOR(4, 1, 0) rec[-6]
        DETECTOR(2, 2, 0) rec[-5]
        DETECTOR(1, 3, 0) rec[-4]
        DETECTOR(1, 0, 0) rec[-3]
        DETECTOR(3, 1, 0) rec[-2]
        DETECTOR(1, 2, 0) rec[-1]
        TICK
        #!pragma POLYGON(0,0,1,0.25) 12 9 6 11
        #!pragma POLYGON(0,1,0,0.25) 8 11 6 2
        #!pragma POLYGON(1,0,0,0.25) 2 6 9 0
        TICK
        MPP X0*X9*X6*X2
        DETECTOR(0, 0, 1) rec[-1]
        TICK
        MPP X2*X6*X11*X8
        DETECTOR(1, 1, 2) rec[-1]
        TICK
        MPP X9*X12*X11*X6
        DETECTOR(3, 0, 3) rec[-1]
        TICK
        MPP Z0*Z9*Z6*Z2
        DETECTOR(0, 0, 4) rec[-1]
        TICK
        MPP Z2*Z6*Z11*Z8
        DETECTOR(1, 1, 5) rec[-1]
        TICK
        MPP Z9*Z12*Z11*Z6
        DETECTOR(3, 0, 6) rec[-1]
        TICK
        TICK
        MPP Y0*Y9*Y12*Y11*Y8*Y2*Y6
        OBSERVABLE_INCLUDE(0) rec[-1] rec[-8] rec[-9]
    """))
    return chunk.with_flag_added_to_all_flows('stage=inject')


def injection_circuit_with_rewritten_injection_rotation(
        circuit: stim.Circuit,
        turns: float,
) -> stim.Circuit:
    turns %= 2
    assert turns % 0.5 == 0
    turns *= 2
    turns = int(turns)
    gates = ['I', 'S', 'Z', 'S_DAG']
    forward = gates[turns]
    backward = gates[-turns]
    basis = ['X', 'Y', 'X', 'Y'][turns]
    sign_change = [False, False, True, True][turns]

    new_circuit = stim.Circuit()
    for inst in circuit.flattened():
        if inst.name == 'S':
            new_circuit.append(stim.CircuitInstruction(forward, inst.targets_copy(), inst.gate_args_copy()))
        elif inst.name == 'S_DAG':
            new_circuit.append(stim.CircuitInstruction(backward, inst.targets_copy(), inst.gate_args_copy()))
        elif inst.name == 'MPP':
            args = inst.gate_args_copy()
            for part in inst.target_groups():
                ts = []
                if all(e.is_y_target for e in part):
                    change_remaining = True
                    for e in part:
                        ts.append(stim.target_pauli(e.qubit_value, basis, invert=change_remaining and sign_change))
                        ts.append(stim.target_combiner())
                        change_remaining = False
                else:
                    for e in part:
                        ts.append(e)
                        ts.append(stim.target_combiner())
                ts.pop()
                new_circuit.append(stim.CircuitInstruction("MPP", ts, args))
        else:
            new_circuit.append(inst)

    return new_circuit


def injection_chunk_with_rewritten_injection_rotation(
        chunk: gen.Chunk,
        turns: float,
        *,
        allow_feedback_z_stabilizer: bool = False,
) -> gen.Chunk:
    new_circuit = injection_circuit_with_rewritten_injection_rotation(chunk.circuit, turns)

    turns %= 2
    assert turns % 0.5 == 0
    turns *= 2
    turns = int(turns)
    basis = ['X', 'Y', 'X', 'Y'][turns]
    sign_change = [False, False, True, True][turns]

    is_init_circuit = all(not flow.start for flow in chunk.flows)
    new_flows = []
    for flow in chunk.flows:
        has_x_start = set(flow.start.qubits.values()) == {'X'}
        has_x_end = set(flow.end.qubits.values()) == {'X'}
        has_y_start = set(flow.start.qubits.values()) == {'Y'}
        has_y_end = set(flow.end.qubits.values()) == {'Y'}
        has_z_end = set(flow.end.qubits.values()) == {'Z'}
        new_end = flow.end
        if (has_y_start and has_y_end) or (has_x_start and has_x_end):
            # Stabilizers need to be insensitive to rotation in the XY basis.
            for b in ['X', 'Y']:
                new_flows.append(flow.with_edits(
                    start=flow.start.with_basis(b),
                    end=flow.start.with_basis(b),
                    sign=False,
                ))
        elif has_y_start or has_y_end:
            # Observables need to adapt sign and basis to the rotation, but not change measurement feedback.
            sign = False
            if set(flow.end.qubits.values()) == {'Y'}:
                new_end = flow.end.with_basis(basis)
                sign ^= sign_change
            new_start = flow.start
            if set(flow.start.qubits.values()) == {'Y'}:
                new_start = flow.start.with_basis(basis)
                sign ^= sign_change
            new_flows.append(flow.with_edits(
                start=new_start,
                end=new_end,
                sign=sign,
            ))
        elif has_z_end:
            new_flows.append(flow)
            if flow.measurement_indices and not allow_feedback_z_stabilizer:
                if is_init_circuit:
                    new_flows.append(flow.with_edits(measurement_indices=[]))
                else:
                    # Z basis should pass through without feedback.
                    new_flows.append(flow.with_edits(start=flow.end, measurement_indices=[]))
        else:
            new_flows.append(flow)

    return chunk.with_edits(circuit=new_circuit, flows=new_flows)
