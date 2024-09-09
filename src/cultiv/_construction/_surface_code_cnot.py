from typing import Literal, Callable

import sinter

import gen
from cultiv._construction._surface_code import make_surface_code


def auto_obs_reflow(
        code1: gen.StabilizerCode,
        code2: gen.StabilizerCode,
) -> gen.ChunkReflow:
    assert code1.stabilizers == code2.stabilizers

    vals = []
    out2in = {}
    for tile in code1.tiles:
        ps = tile.to_data_pauli_string()
        out2in[ps] = [ps]
        vals.append(ps)
    for k, logical in enumerate(code1.logicals):
        assert isinstance(logical, (gen.PauliMap, gen.KeyedPauliMap))
        vals.append(logical.keyed(k))
    for k, obs in enumerate(code2.logicals):
        out2in[obs.keyed(k)] = 'auto'

    return gen.ChunkReflow.from_auto_rewrite(
        inputs=vals,
        out2in=out2in,
    )


def _make_idle_chunk(code: gen.StabilizerCode) -> gen.Chunk:
    builder = gen.Builder.for_qubits(code.used_set)

    mxs = code.stabilizers.with_only_x_tiles().measure_set
    mzs = code.stabilizers.with_only_z_tiles().measure_set
    builder.append('RX', mxs)
    builder.append('RZ', mzs)
    builder.append("TICK")

    for layer in range(4):
        offset = [1j, 1, -1, -1j][layer]
        cxs = []
        for tile in code.tiles:
            m = tile.measure_qubit
            s = -1 if tile.basis == 'Z' else +1
            d = m + offset * (s if 1 <= layer <= 2 else 1)
            if d in code.data_set:
                cxs.append((m, d)[::s])
        builder.append('CX', cxs)
        builder.append("TICK")
    builder.append('MX', mxs)
    builder.append('MZ', mzs)

    flows = []
    for tile in code.tiles:
        rec = builder.lookup_recs([tile.measure_qubit])
        flows.append(tile.to_prepare_flow(rec))
        flows.append(tile.to_measure_flow(rec))
    for k, obs in enumerate(code.logicals):
        flows.append(gen.Flow(
            start=obs,
            end=obs,
            obs_key=k,
        ))

    return gen.Chunk(builder.circuit, flows=flows)


def _make_data_reset_layer(
        *,
        prev_code: gen.StabilizerCode,
        next_code: gen.StabilizerCode,
        data_basis_func: Callable[[complex], Literal['X', 'Y', 'Z'] | str],
) -> gen.Chunk:
    builder = gen.Builder.for_qubits(prev_code.used_set | next_code.used_set)

    assert next_code.data_set >= prev_code.data_set
    gained_qubits = next_code.data_set - prev_code.data_set
    for b, qs in sorted(sinter.group_by(gained_qubits, key=data_basis_func).items()):
        builder.append(f'R{b}', qs)

    gained_bases = {q: data_basis_func(q) for q in gained_qubits}
    flows = []
    discards_in = []
    discards_out = []
    for tile in next_code.tiles:
        ps = tile.to_data_pauli_string()
        if any(gained_bases.get(q, b) != b for q, b in ps.items()):
            discards_out.append(ps)
        else:
            flows.append(gen.Flow(
                start=gen.PauliMap({q: b for q, b in ps.items() if q not in gained_bases}),
                end=ps,
                center=tile.measure_qubit,
            ))
    for tile in prev_code.tiles:
        ps = tile.to_data_pauli_string()
        if any(gained_bases.get(q, b) != b for q, b in ps.items()):
            discards_in.append(ps)
    for k, obs in enumerate(next_code.logicals):
        flows.append(gen.Flow(
            start=gen.PauliMap({q: b for q, b in obs.items() if q not in gained_bases}),
            end=obs,
            obs_key=k,
        ))

    return gen.Chunk(
        builder.circuit,
        flows=flows,
        discarded_inputs=discards_in,
        discarded_outputs=discards_out,
    )


def _make_data_measure_layer(
        *,
        prev_code: gen.StabilizerCode,
        next_code: gen.StabilizerCode,
        data_basis_func: Callable[[complex], Literal['X', 'Y', 'Z'] | str],
) -> gen.Chunk:
    builder = gen.Builder.for_qubits(prev_code.used_set | next_code.used_set)

    assert next_code.data_set <= prev_code.data_set
    lost_qubits = prev_code.data_set - next_code.data_set
    for b, qs in sorted(sinter.group_by(lost_qubits, key=data_basis_func).items()):
        builder.append(f'M{b}', qs)

    lost_bases = {q: data_basis_func(q) for q in lost_qubits}
    flows = []
    discards_in = []
    discards_out = []
    for tile in prev_code.tiles:
        ps = tile.to_data_pauli_string()
        if any(lost_bases.get(q, b) != b for q, b in ps.items()):
            discards_in.append(ps)
        else:
            flows.append(gen.Flow(
                start=ps,
                end=gen.PauliMap({q: b for q, b in ps.items() if q not in lost_qubits}),
                center=tile.measure_qubit,
                measurement_indices=builder.lookup_recs(ps.keys() & lost_qubits),
            ))
    for tile in next_code.tiles:
        ps = tile.to_data_pauli_string()
        if any(lost_bases.get(q, b) != b for q, b in ps.items()):
            discards_out.append(ps)
    for k, obs in enumerate(prev_code.logicals):
        flows.append(gen.Flow(
            start=obs,
            end=gen.PauliMap({q: b for q, b in obs.items() if q not in lost_qubits}),
            obs_key=k,
            measurement_indices=builder.lookup_recs(obs.keys() & lost_qubits),
        ))

    return gen.Chunk(
        builder.circuit,
        flows=flows,
        discarded_inputs=discards_in,
        discarded_outputs=discards_out,
    )


def make_surface_code_cnot(
        *,
        distance: int,
        basis: Literal['X', 'Z'],
):
    d = distance
    anc = make_surface_code(d, d)
    sep = d + 2 - (d % 2)
    q1 = anc.with_transformed_coords(lambda e: e + sep * (1 - 1j))
    q2 = anc.with_transformed_coords(lambda e: e + sep * (1 + 1j))
    anc_q1_zz = make_surface_code(d + sep, d)
    anc_q2_xx = make_surface_code(d, d + sep)

    x1 = q1.logicals[0][0]
    z1 = q1.logicals[0][1]
    x2 = q2.logicals[0][0]
    z2 = q2.logicals[0][1]
    xa = anc.logicals[0][0]
    za = anc.logicals[0][1]
    start = gen.StabilizerCode(
        stabilizers=q1.stabilizers + q2.stabilizers,
        logicals=[
            (x1, z1),
            (x2, z2),
        ]
    )
    zz = gen.StabilizerCode(
        stabilizers=anc_q1_zz.stabilizers + q2.stabilizers,
        logicals=[
            (x1, anc_q1_zz.logicals[0][1]),
            (x2, z2),
        ]
    )
    zz2 = gen.StabilizerCode(
        stabilizers=anc_q1_zz.stabilizers + q2.stabilizers,
        logicals=[
            (x1, anc_q1_zz.logicals[0][1]),
            (x2 * x1 * xa, z2),
        ]
    )
    mid = gen.StabilizerCode(
        stabilizers=q1.stabilizers + q2.stabilizers + anc.stabilizers,
        logicals=[
            (x1, z1 * za),
            (x2 * x1 * xa, z2),
        ]
    )
    xx = gen.StabilizerCode(
        stabilizers=anc_q2_xx.stabilizers + q1.stabilizers,
        logicals=[
            (x1, z1 * za),
            (x1 * anc_q2_xx.logicals[0][0], z2),
        ]
    )
    xx2 = gen.StabilizerCode(
        stabilizers=anc_q2_xx.stabilizers + q1.stabilizers,
        logicals=[
            (x1, z1 * z2),
            (x1 * anc_q2_xx.logicals[0][0], z2),
        ]
    )
    end = gen.StabilizerCode(
        stabilizers=q1.stabilizers + q2.stabilizers,
        logicals=[
            (x1, z1 * z2),
            (x1 * x2, z2),
        ]
    )

    start = start.with_observables_from_basis(basis=basis)
    xx = xx.with_observables_from_basis(basis=basis)
    xx2 = xx2.with_observables_from_basis(basis=basis)
    mid = mid.with_observables_from_basis(basis=basis)
    zz = zz.with_observables_from_basis(basis=basis)
    zz2 = zz2.with_observables_from_basis(basis=basis)
    end = end.with_observables_from_basis(basis=basis)

    chunks = [
        _make_idle_chunk(start),
        _make_data_reset_layer(
            prev_code=start,
            next_code=zz,
            data_basis_func=lambda _: 'Z',
        ),
        _make_idle_chunk(zz),
        auto_obs_reflow(zz, zz2),
        _make_idle_chunk(zz2) * (distance - 1),
        _make_data_measure_layer(
            prev_code=zz2,
            next_code=mid,
            data_basis_func=lambda _: 'Z',
        ),
        _make_data_reset_layer(
            prev_code=mid,
            next_code=xx,
            data_basis_func=lambda _: 'X',
        ),
        _make_idle_chunk(xx),
        auto_obs_reflow(xx, xx2),
        _make_idle_chunk(xx2) * (distance - 1),
        _make_data_measure_layer(
            prev_code=xx2,
            next_code=start,
            data_basis_func=lambda _: 'X',
        ),
        _make_idle_chunk(end),
    ]
    for c in chunks:
        c.verify()

    circuit = gen.compile_chunks_into_circuit(
        chunks,
        add_mpp_boundaries=True,
    )
    circuit = gen.LayerCircuit.from_stim_circuit(circuit)
    circuit = circuit.with_ejected_loop_iterations()
    circuit = circuit.with_locally_optimized_layers()
    circuit = circuit.with_locally_merged_measure_layers()
    circuit = circuit.with_locally_optimized_layers()
    circuit = circuit.with_cleaned_up_loop_iterations()
    circuit = circuit.to_stim_circuit()
    circuit = gen.stim_circuit_with_transformed_coords(circuit, lambda e: e * (0.5 + 0.5j))

    return circuit


def main():
    d = 3
    anc = make_surface_code(d, d)
    sep = d + 2 - (d % 2)
    q1 = anc.with_transformed_coords(lambda e: e + sep * (1 - 1j))
    q2 = anc.with_transformed_coords(lambda e: e + sep * (1 + 1j))
    anc_q1_zz = make_surface_code(d + sep, d)
    anc_q2_xx = make_surface_code(d, d + sep)

    x1 = q1.logicals[0][0]
    z1 = q1.logicals[0][1]
    x2 = q2.logicals[0][0]
    z2 = q2.logicals[0][1]
    xa = anc.logicals[0][0]
    za = anc.logicals[0][1]
    start = gen.StabilizerCode(
        stabilizers=q1.stabilizers + q2.stabilizers,
        logicals=[
            (x1, z1),
            (x2, z2),
        ]
    )
    # assert len(gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(gen.compile_chunks_into_circuit([_make_idle_chunk(start.with_observables_from_basis('X')) * 7], add_mpp_boundaries=True)).shortest_graphlike_error()) == 3

    gen.write_file('tmp.html', gen.stim_circuit_html_viewer(make_surface_code_cnot(
        distance=d,
        basis='X',
    ), patch=start.with_transformed_coords(lambda e: e * (0.5 + 0.5j))))


if __name__ == '__main__':
    main()
