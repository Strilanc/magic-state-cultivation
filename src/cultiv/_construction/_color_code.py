from typing import Literal

import gen


def tile_rgb_color(tile: gen.Tile) -> str | tuple[float, float, float] | tuple[float, float, float, float]:
    if 'color=r' in tile.flags:
        if tile.basis == 'X':
            return 0.5, 0.1, 0.1
        else:
            return 1, 0.6, 0.6
    if 'color=g' in tile.flags:
        if tile.basis == 'X':
            return 0.1, 0.5, 0.1
        else:
            return 0.6, 1, 0.6
    if 'color=b' in tile.flags:
        if tile.basis == 'X':
            return 0.1, 0.1, 0.5
        else:
            return 0.6, 0.6, 1
    if tile.basis == 'X':
        return 0.25, 0.25, 0.25
    else:
        return 0.6, 0.6, 0.6


def make_color_code(
        base_width: int,
        *,
        obs_location: Literal['all', 'top', 'bottom-left', 'bottom-right', 'x-top-z-bottom-right'] = 'bottom-left',
) -> gen.StabilizerCode:
    tiles = []

    def is_in_color_code(q: complex) -> bool:
        if q.imag < 0:
            return False
        if q.real * 3 < q.imag * 2 - 1:
            return False
        if q.real * 3 > -3 + base_width * 6 - q.imag * 2:
            return False
        return True

    hex_offsets = [
        2,
        -1,
        1j,
        -1j,
        1 + 1j,
        1 - 1j,
    ]

    for x in range(0, base_width * 2):
        for y in range(0, base_width * 2):
            m = x + 1j*y
            if x % 2 != 1 or y % 2 != x // 2 % 2:
                continue
            color = (y // 2 - (x // 2 % 2 == 1)) % 3
            for b in 'XZ':
                tile = gen.Tile(
                    data_qubits=[m + d for d in hex_offsets if is_in_color_code(m + d)],
                    measure_qubit=m + (b == 'Z'),
                    bases=b,
                    flags={
                        f'color={"rgb"[color]}',
                        f'cycle=superdense',
                        f'basis={b}',
                    }
                )
                if len(tile.data_qubits) not in [4, 6]:
                    continue
                tiles.append(tile)

    patch = gen.Patch(tiles)
    bottom_right_obs_qubits = {
        tile.measure_qubit + d
        for tile in tiles
        if 'color=b' in tile.flags
        if tile.basis == 'X'
        if len(tile.data_qubits) == 4
        for d in [1j, 1 - 1j]
    } | {max(patch.data_set, key=lambda e: e.imag)}
    bottom_left_obs_qubits = {
        tile.measure_qubit + d
        for tile in tiles
        if 'color=g' in tile.flags
        if tile.basis == 'X'
        if len(tile.data_qubits) == 4
        for d in [-1j, 1 + 1j]
    } | {0}
    top_obs_qubits = {q for q in patch.data_set if q.imag == 0}

    if obs_location == 'bottom-left':
        obs_x_qubits = bottom_left_obs_qubits
        obs_z_qubits = bottom_left_obs_qubits
    elif obs_location == 'all':
        obs_x_qubits = patch.data_set
        obs_z_qubits = patch.data_set
    elif obs_location == 'top':
        obs_x_qubits = top_obs_qubits
        obs_z_qubits = top_obs_qubits
    elif obs_location == 'bottom-right':
        obs_x_qubits = bottom_right_obs_qubits
        obs_z_qubits = bottom_right_obs_qubits
    elif obs_location == 'x-top-z-bottom-right':
        obs_x_qubits = top_obs_qubits
        obs_z_qubits = bottom_right_obs_qubits
    else:
        raise NotImplementedError(f'{obs_location=}')
    obs_x = gen.PauliMap(xs=obs_x_qubits)
    obs_z = gen.PauliMap(zs=obs_z_qubits)
    return gen.StabilizerCode(stabilizers=patch, logicals=[(obs_x, obs_z)])


def make_chunk_color_code_superdense_cycle(
        code: gen.StabilizerCode,
        *,
        check_x: bool = True,
        check_z: bool = True,
        obs_basis: Literal['X', 'Y', 'Z'],
) -> gen.Chunk:
    builder = gen.Builder.for_qubits(code.used_set)
    bell_pairs = [
        (tile.measure_qubit, tile.measure_qubit + 1)
        for tile in code.stabilizers.tiles
        if tile.basis == 'X'
    ]

    cx1 = list(bell_pairs)
    cx2 = []
    cx3 = []
    cx4 = []
    cx5 = []
    cx6 = []
    cx7 = []
    cx8 = list(bell_pairs)

    for tile in code.stabilizers.tiles:
        def tile_cx(out: list, a: complex, b: complex, *, rev: bool = False):
            if rev:
                a, b = b, a
            if a in tile.data_set or b in tile.data_set:
                out.append((a, b))

        m = tile.measure_qubit
        if tile.basis == 'X':
            if check_x:
                tile_cx(cx2, m, m - 1)
                tile_cx(cx2, m + 1, m + 2)
                tile_cx(cx3, m, m + 1j)
                tile_cx(cx3, m + 1, m + 1 + 1j)
                tile_cx(cx4, m, m - 1j)
                tile_cx(cx4, m + 1, m + 1 - 1j)
        elif tile.basis == 'Z':
            m -= 1
            if check_z:
                tile_cx(cx5, m, m - 1, rev=True)
                tile_cx(cx5, m + 1, m + 2, rev=True)
                tile_cx(cx6, m, m + 1j, rev=True)
                tile_cx(cx6, m + 1, m + 1 + 1j, rev=True)
                tile_cx(cx7, m, m - 1j, rev=True)
                tile_cx(cx7, m + 1, m + 1 - 1j, rev=True)
        else:
            raise NotImplementedError(f'{tile=}')

    builder.append('RX', code.stabilizers.with_only_x_tiles().measure_set)
    builder.append('RZ', code.stabilizers.with_only_z_tiles().measure_set)
    builder.append("TICK")

    for cx in [cx1, cx2, cx3, cx4, cx5, cx6, cx7, cx8]:
        if cx:
            builder.append('CX', cx)
            builder.append("TICK")

    builder.append('MX', code.stabilizers.with_only_x_tiles().measure_set)
    builder.append('MZ', code.stabilizers.with_only_z_tiles().measure_set)
    flows = []
    for tile in code.tiles:
        checked = check_x if tile.basis == 'X' else check_z
        if checked:
            flows.append(tile.to_prepare_flow('auto'))
            flows.append(tile.to_measure_flow('auto'))
        else:
            ps = tile.to_data_pauli_string()
            flows.append(gen.Flow(
                measurement_indices=builder.lookup_recs([tile.measure_qubit]),
                flags=tile.flags,
                center=tile.measure_qubit,
            ))
            flows.append(gen.Flow(
                start=ps,
                end=ps,
                flags=tile.flags,
                center=tile.measure_qubit,
            ))
    flows.extend(code.auto_obs_passthrough_flows(obs_basis=obs_basis))

    return gen.Chunk(
        circuit=builder.circuit,
        q2i=builder.q2i,
        flows=flows,
    )


def make_growing_color_code_bell_pair_patch(start_base_width: int, end_base_width: int) -> gen.Patch:
    code1 = make_color_code(start_base_width)
    code2 = make_color_code(end_base_width)
    tiles = [*make_color_code(start_base_width).stabilizers.tiles]
    for tile in code2.stabilizers.tiles:
        m = tile.measure_qubit
        if m in code1.stabilizers.measure_set:
            continue
        if 'color=r' in tile.flags and tile.basis == 'X':
            if m.imag > 1:
                tiles.append(gen.Tile(
                    data_qubits=[m + 1 - 1j, m + 2 - 2j],
                    bases='Y',
                    measure_qubit=m + 1 - 2j,
                    flags={'bell-common-neighbor'},
                ))
            tiles.append(gen.Tile(
                data_qubits=[m + 2, m + 3],
                bases='Y',
                measure_qubit=m + 2,
                flags={'bell-adjacent'},
            ))
            tiles.append(gen.Tile(
                data_qubits=[m + 1 + 1j, m + 2 + 2j],
                bases='Y',
                measure_qubit=m + 2 + 1j,
                flags={'bell-common-neighbor'},
            ))
        if 'color=g' in tile.flags and tile.basis == 'X' and len(tile.data_set) == 4:
            tiles.append(gen.Tile(
                data_qubits=[m - 1j, m + 1 + 1j],
                bases='Y',
                measure_qubit=m,
                flags={'bell-two-step-at-boundary'},
            ))
    return gen.Patch(tiles)


def make_color_code_grow_chunk(start_base_width: int, end_base_width: int, basis: Literal['X', 'Y', 'Z']) -> gen.Chunk:
    grow_patch = make_growing_color_code_bell_pair_patch(start_base_width, end_base_width)
    old_small_code = make_color_code(start_base_width)
    new_big_code = make_color_code(end_base_width)
    builder = gen.Builder.for_qubits(new_big_code.used_set)

    rx = set()
    cx1 = []
    cx2 = []
    cx3 = []
    for tile in grow_patch.tiles:
        m = tile.measure_qubit
        if 'bell-common-neighbor' in tile.flags:
            a, b = tile.data_qubits
            rx.add(m)
            cx1.append((m, a))
            cx2.append((m, b))
            cx3.append((tile.data_qubits[0], m))
        elif 'bell-two-step-at-boundary' in tile.flags:
            a, b = tile.data_qubits
            rx.add(m)
            cx1.append((m, m + 1))
            cx2.append((m, a))
            cx2.append((m + 1, b))
            cx3.append((a, m))
            cx3.append((b, m + 1))
        elif 'bell-adjacent' in tile.flags:
            a, b = tile.data_qubits
            rx.add(a)
            cx1.append((a, b))
    if start_base_width != end_base_width:
        builder.append('RX', rx)
        builder.append('RZ', new_big_code.used_set - old_small_code.used_set - rx)
        builder.append("TICK")
        builder.append('CX', cx1)
        builder.append("TICK")
        builder.append('CX', cx2)
        builder.append("TICK")
        builder.append('CX', cx3)

    flows = []
    discards = []
    for tile in new_big_code.stabilizers.tiles:
        prev = old_small_code.stabilizers.m2tile.get(tile.measure_qubit, None)
        if prev is not None:
            flows.append(gen.Flow(
                start=prev.to_data_pauli_string(),
                end=tile.to_data_pauli_string(),
                center=tile.measure_qubit,
                flags=tile.flags,
            ))
        elif 'color=r' in tile.flags:
            discards.append(tile.to_data_pauli_string())
        else:
            flows.append(gen.Flow(
                end=tile.to_data_pauli_string(),
                center=tile.measure_qubit,
                flags=tile.flags,
            ))
    obs_x_flow = gen.Flow(
        start=old_small_code.get_observable_by_basis(0, 'X'),
        end=new_big_code.get_observable_by_basis(0, 'X'),
        obs_key=0,
        center=-1,
    )
    obs_z_flow = gen.Flow(
        start=old_small_code.get_observable_by_basis(0, 'Z'),
        end=new_big_code.get_observable_by_basis(0, 'Z'),
        obs_key=0,
        center=-1,
    )
    if basis == 'X':
        flows.append(obs_x_flow)
    elif basis == 'Y':
        flows.append(obs_x_flow * obs_z_flow)
    elif basis == 'Z':
        flows.append(obs_z_flow)
    else:
        raise NotImplementedError(f'{basis=}')

    return gen.Chunk(
        circuit=builder.circuit,
        q2i=builder.q2i,
        flows=flows,
        discarded_outputs=discards,
    )
