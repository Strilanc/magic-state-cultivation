from typing import Literal

import gen


def make_surface_code(width: int, height: int) -> gen.StabilizerCode:
    tiles = []

    for x in range(-1, width):
        for y in range(-1, height):
            m = x + 1j*y + 0.5 + 0.5j
            data = [m + 1j**k * (0.5 + 0.5j) for k in range(4)]
            data = [d for d in data if 0 <= d.real < width]
            data = [d for d in data if 0 <= d.imag < height]
            basis = 'XZ'[(x.real + y.real) % 2 == 0]
            if (m.real < 0 or m.real > width - 1) and basis == 'X':
                continue
            if (m.imag < 0 or m.imag > height - 1) and basis == 'Z':
                continue
            if len(data) not in [2, 4]:
                continue
            tiles.append(gen.Tile(
                measure_qubit=m,
                data_qubits=data,
                bases=basis,
                flags={f'basis={basis}'},
            ))

    patch = gen.Patch(tiles)
    obs_x = gen.PauliMap(xs=[q for q in patch.data_set if q.real == 0])
    obs_z = gen.PauliMap(zs=[q for q in patch.data_set if q.imag == 0])
    return gen.StabilizerCode(stabilizers=patch, logicals=[(obs_x, obs_z)]).with_transformed_coords(lambda e: e * (1 - 1j))


def make_surface_code_idle_chunk(code: gen.StabilizerCode, basis: Literal['X', 'Y', 'Z']) -> gen.Chunk:
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
    flows.append(code.auto_obs_passthrough_flows(obs_basis=basis)[0].with_edits(measurement_indices=[]))

    return gen.Chunk(builder.circuit, flows=flows)
