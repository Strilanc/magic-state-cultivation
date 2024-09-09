from typing import Literal

import sinter

import gen
from ._color_code import make_color_code
from ._surface_code import make_surface_code


def make_hybrid_color_surface_code(
        *,
        dcolor: int,
        dsurface: int,
        obs_location: Literal['left', 'right', 'transition']
) -> gen.StabilizerCode:
    if dsurface < dcolor * 2:
        raise ValueError(f'{dsurface=} < {dcolor=} * 2')

    color_code = make_color_code(dcolor)
    surface_code = make_surface_code(dsurface, dsurface)
    bottom_left_obs_x_qubits = set()

    tiles = []
    for tile in color_code.tiles:
        if tile.basis == 'X' and len(tile.data_qubits) == 4 and 'color=b' in tile.flags:
            m = tile.measure_qubit

            # Extended blue hexagon along the bottom-right boundary of the color code.
            tiles.append(tile.with_edits(data_qubits=[
                m + d
                for d in [-1, 1j, 1 + 1j, 2, -1j, 1 - 1j]
            ]))

            # Trapezoids below the bottom-right boundary of the color code.
            m2 = m
            for k in range(int(m.real) - dcolor + 1):
                m2 += 2j
                tiles.append(gen.Tile(
                    measure_qubit=m2 + 1,
                    data_qubits=[m2 + d for d in [-1, -1j, 1j, 1 - 1j]],
                    bases='Z',
                    flags={'basis=Z', 'cycle=superdense'} | ({'color=r'} if k == 0 else set()),
                ))
                tiles.append(gen.Tile(
                    measure_qubit=m2,
                    data_qubits=[m2 + d for d in [1 - 1j, 1 + 1j, 1j, 2]],
                    bases='X',
                    flags={'basis=X', 'cycle=superdense'},
                ))

            # Boundary stabiblizers along the bottom left of the patch.
            m2 += 2j
            tiles.append(gen.Tile(
                measure_qubit=m2 + 1,
                data_qubits=[m2 + d for d in [-1j, 1 - 1j]],
                bases='Z',
                flags={'basis=Z', 'cycle=superdense'},
            ))
            tiles.append(gen.Tile(
                measure_qubit=m2,
                data_qubits=[None],
                bases='X',
                flags={'basis=X', 'cycle=superdense'},
            ))
            bottom_left_obs_x_qubits.add(m2 - 1j)
            bottom_left_obs_x_qubits.add(m2 - 1j + 1)
        elif tile.basis == 'Z' and len(tile.data_qubits) == 4 and 'color=r' in tile.flags:
            m = tile.measure_qubit - 1

            # Extended and deformed red hexagon along the top boundary of the color code.
            tiles.append(tile.with_edits(data_qubits=[
                m + d
                for d in [-1, 1j, 1 + 1j, 2, 2-1j, - 1j]
            ]))

            # Green pentagon along the top boundary of the color code.
            m2 = m + 3 - 1j
            tiles.append(gen.Tile(
                measure_qubit=m2,
                data_qubits=[m2 + d for d in [-1, 1j, 1j - 1, -1j, 1]],
                bases='X',
                flags={'basis=X', 'color=g', 'cycle=pentagon-above-color-code'},
            ))

            # Uncolored triangle along thBe top boundary of the color code.
            m3 = m2 - 2
            tiles.append(gen.Tile(
                measure_qubit=m3,
                data_qubits=[m3 + d for d in [1, -1, -1j]],
                bases='X',
                flags={'basis=X', 'cycle=triangle-above-color-code'},
            ))
        else:
            tiles.append(tile)

    used_qubits = gen.Patch(tiles).used_set
    bottom_left_obs_x_qubits |= color_code.get_observable_by_basis(0, 'X').keys()
    bottom_left_obs_x_qubits |= surface_code.get_observable_by_basis(0, 'X').keys() - used_qubits
    for tile in surface_code.tiles:
        if tile.measure_qubit not in used_qubits:
            tiles.append(tile.with_edits(flags={'cycle=shingle' if len(tile.data_set) == 4 else 'cycle=shingle-boundary', f'basis={tile.basis}'}))

    patch = gen.Patch(tiles)
    if obs_location == 'right':
        right_qubits = [
            max(group, key=lambda e: e.real)
            for group in sinter.group_by(patch.data_set, key=lambda e: e.imag).values()
        ]
        obs_x_qubits = [q for q in right_qubits if q.imag <= 0]
        obs_z_qubits = [q for q in right_qubits if q.imag >= 0]
    elif obs_location == 'left':
        obs_x_qubits = bottom_left_obs_x_qubits
        obs_z_qubits = color_code.get_observable_by_basis(0, 'Z').keys() | surface_code.get_observable_by_basis(0, 'Z').keys()
    elif obs_location == 'transition':
        prev_code = make_color_code(base_width=dcolor, obs_location='x-top-z-bottom-right')
        obs_x_qubits = prev_code.get_observable_by_basis(0, 'X').keys() | {q for q in patch.data_set if q.imag == 0 and q.real > dcolor * 2 - 2}
        obs_z_qubits = prev_code.get_observable_by_basis(0, 'Z').keys() | {q for q in patch.data_set if q.imag == -1 and q.real > dcolor * 2 - 2}
    else:
        raise NotImplementedError(f'{obs_location=}')

    obs_x = gen.PauliMap(xs=obs_x_qubits)
    obs_z = gen.PauliMap(zs=obs_z_qubits)

    return gen.StabilizerCode(
        stabilizers=patch,
        logicals=[(obs_x, obs_z)],
    )


def make_color_code_grown_into_surface_code_then_ablated_into_matchable_code_simple(
        *,
        dcolor: int,
        dsurface: int,
        desaturate: bool = True,
) -> gen.StabilizerCode:
    grown_code = make_hybrid_color_surface_code(dcolor=dcolor, dsurface=dsurface, obs_location='left')

    tiles = [
        tile
        for tile in grown_code.tiles
        if not ('color=r' in tile.flags and 'basis=X' in tile.flags)
        if not ('color=g' in tile.flags and 'basis=Z' in tile.flags)
    ]

    if desaturate:
        tiles = [
            tile.with_edits(flags=tile.flags - {'color=r', 'color=b', 'color=g'})
            for tile in tiles
        ]

    return grown_code.with_edits(stabilizers=gen.Patch(tiles))


def make_color_code_all_to_transition_reflow(code1: gen.StabilizerCode, code2: gen.StabilizerCode) -> gen.ChunkReflow:
    code1 = code1.with_observables_from_basis('Y')
    code2 = code2.with_observables_from_basis('Y')

    out2in = {}
    for tile in code1.tiles:
        ps = tile.to_data_pauli_string()
        out2in[ps] = [ps]

    start = code1.logicals[0].keyed(0)
    end = code2.logicals[0].keyed(0)
    out2in[end] = [start, *[
        tile.to_data_pauli_string()
        for tile in code1.tiles
        if ('color=g' in tile.flags and 'basis=X' in tile.flags) or ('color=r' in tile.flags and 'basis=Z' in tile.flags)
    ]]

    return gen.ChunkReflow(out2in=out2in)


def make_color_code_to_big_matchable_code_escape_chunks(
        *,
        dcolor: int,
        dsurface: int,
        basis: Literal['X', 'Y', 'Z'],
        r_growing: int,
        r_end: int,
) -> list[gen.Chunk | gen.ChunkReflow | gen.ChunkLoop]:
    assert r_growing > 0
    assert dcolor % 2 == 1
    assert dsurface >= dcolor * 2
    code1a = make_color_code(dcolor, obs_location='all')
    code1b = make_color_code(dcolor, obs_location='x-top-z-bottom-right')
    code2a = make_hybrid_color_surface_code(dcolor=dcolor, dsurface=dsurface, obs_location='transition')
    code2b = make_hybrid_color_surface_code(dcolor=dcolor, dsurface=dsurface, obs_location='right')
    code3 = make_post_escape_matchable_code(dcolor=dcolor, dsurface=dsurface)

    return [
        make_color_code_all_to_transition_reflow(code1a, code1b),
        make_color_code_to_growing_code_chunk(start_code=code1b, end_code=code2a, obs_basis=basis),
        make_hybrid_code_round_chunk(prev_code=code2a, code=code2b, obs_basis=basis),
        make_hybrid_code_round_chunk(code=code2b, obs_basis=basis) * (r_growing - 1),
        make_hybrid_code_round_chunk(prev_code=code2b, code=code3, obs_basis=basis),
        make_hybrid_code_round_chunk(code=code3, obs_basis=basis) * r_end,
    ]


def make_color_code_grown_into_surface_code_then_ablated_into_matchable_code_full_edges(
        *,
        dcolor: int,
        dsurface: int,
        desaturate: bool = True,
) -> gen.StabilizerCode:
    grown_code = make_hybrid_color_surface_code(dcolor=dcolor, dsurface=dsurface, obs_location='left')

    tiles = [
        tile
        for tile in grown_code.tiles
        if not ('color=r' in tile.flags and 'basis=X' in tile.flags)
        if not ('color=g' in tile.flags and 'basis=Z' in tile.flags)
        if not ('color=b' in tile.flags and 'basis=Z' in tile.flags)
    ]

    remove_z = set()
    for tile in grown_code.tiles:
        if 'color=g' in tile.flags and 'basis=X' in tile.flags and len(tile.data_set) in [4, 6]:
            m = tile.measure_qubit
            edges = [(m - 1, m - 1j), (m + 2, m + 1 - 1j)]
            if len(tile.data_set) == 4:
                edges.append((m - 1j, m + 1 + 1j))
            for a, b in edges:
                if a in grown_code.data_set and b in grown_code.data_set:
                    remove_z.add(a)
                    remove_z.add(b)
                    tiles.append(gen.Tile(
                        data_qubits=[a, b],
                        bases='Z',
                        measure_qubit=a,
                        flags={'basis=Z', 'color=r'},
                    ))
        if 'color=b' in tile.flags and 'basis=X' in tile.flags and len(tile.data_set) in [4, 6]:
            m = tile.measure_qubit
            edges = [(m - 1j, m + 1 - 1j)]
            for a, b in edges:
                if a in grown_code.data_set and b in grown_code.data_set:
                    remove_z.add(a)
                    remove_z.add(b)
                    tiles.append(gen.Tile(
                        data_qubits=[a, b],
                        bases='Z',
                        measure_qubit=a,
                        flags={'basis=Z', 'color=r'},
                    ))

    if desaturate:
        tiles = [
            tile.with_edits(flags=tile.flags - {'color=r', 'color=b', 'color=g'})
            for tile in tiles
        ]

    remove_z.clear()
    return grown_code.with_edits(
        stabilizers=gen.Patch(tiles),
        logicals=[
            (grown_code.logicals[0][0], gen.PauliMap(zs=grown_code.logicals[0][1].keys() - remove_z)),
        ]
    )


def make_post_escape_matchable_code(
        *,
        dcolor: int,
        dsurface: int,
        desaturate: bool = True,
) -> gen.StabilizerCode:
    grown_code = make_hybrid_color_surface_code(
        dcolor=dcolor,
        dsurface=dsurface,
        obs_location='right',
    )

    tiles = []

    left_boundary_greens = {
        tile.measure_qubit
        for tile in grown_code.tiles
        if tile.basis == 'Z'
        if 'color=g' in tile.flags
        if len(tile.data_qubits) == 4
    }

    near_left_boundary_blues = {
        tile.measure_qubit
        for tile in grown_code.tiles
        if tile.basis == 'Z'
        if 'color=b' in tile.flags
        if tile.measure_qubit - 2 + 1j in left_boundary_greens
    }

    bottom_blues = {
        tile.measure_qubit
        for tile in grown_code.tiles
        if tile.basis == 'Z'
        if 'color=b' in tile.flags
        if len(tile.data_qubits) == 4
    }

    near_bottom_greens = {
        tile.measure_qubit
        for tile in grown_code.tiles
        if tile.basis == 'Z'
        if 'color=g' in tile.flags
        if tile.measure_qubit - 1j + 2 in bottom_blues
    }

    near_left_boundary_greens = {
        tile.measure_qubit
        for tile in grown_code.tiles
        if tile.basis == 'Z'
        if 'color=g' in tile.flags
        if len(tile.data_qubits) == 6
        if tile.measure_qubit + 2j in near_left_boundary_blues
        if tile.measure_qubit not in near_bottom_greens
    }

    for tile in grown_code.tiles:
        m = tile.measure_qubit
        if m in near_left_boundary_greens:
            tiles.append(tile.with_edits(
                data_qubits=[m + 1j, m - 1 + 1j],
                flags=tile.flags ^ {'lost-to-transition'},
            ))
        elif m in left_boundary_greens:
            # Split into two-body operators.
            extra_flags = {'cycle=superdense-boundary-edge', 'cycle=superdense'}
            tiles.append(tile.with_edits(
                measure_qubit=m - 1 + 1j,
                data_qubits=[m + 1j, m - 1 - 1j],
                flags=tile.flags ^ extra_flags
            ))
            tiles.append(tile.with_edits(
                data_qubits=[m + 1, m - 1j],
            ))
        elif m in near_left_boundary_blues or m in bottom_blues:
            dropped = {m - 2, m - 1 + 1j}
            if m - 2j in near_left_boundary_greens:
                dropped |= {m - 1j, m - 1 - 1j}
            tiles.append(tile.with_edits(
                data_qubits=set(tile.data_qubits) ^ dropped,
                flags=tile.flags,
            ))
        elif m in near_bottom_greens:
            extra_flags = {'lost-to-transition'}
            if m - 4 in left_boundary_greens and len(grown_code.stabilizers.m2tile[m - 3j - 2].data_set) != 5:
                extra_flags.clear()
            tiles.append(tile.with_edits(
                data_qubits=[m + 1, m - 1j],
                flags=tile.flags ^ extra_flags,
            ))
        elif 'color=r' in tile.flags:
            # Drop the red X stabilizers.
            if tile.basis == 'X':
                tile = tile.with_edits(data_qubits=[None])
            tiles.append(tile.with_edits(flags=tile.flags ^ {'cycle=bell-flagged-Z', 'cycle=superdense'}))
        elif 'color=g' in tile.flags and len(tile.data_qubits) != 5:
            # Drop the green Z stabilizers.
            if tile.basis == 'Z':
                tile = tile.with_edits(data_qubits=[None])
            tiles.append(tile.with_edits(flags=tile.flags ^ {'cycle=bell-flagged-X', 'cycle=superdense'}))
        else:
            tiles.append(tile)

    if desaturate:
        tiles = [
            tile.with_edits(flags=tile.flags - {'color=r', 'color=b', 'color=g'})
            for tile in tiles
        ]

    return gen.StabilizerCode(
        stabilizers=gen.Patch(tiles),
        logicals=grown_code.logicals,
    )


def make_color_code_to_growing_code_chunk(
    *,
    start_code: gen.StabilizerCode,
    end_code: gen.StabilizerCode,
    obs_basis: Literal['X', 'Y', 'Z'],
) -> gen.Chunk:
    builder = gen.Builder.for_qubits(end_code.used_set | start_code.used_set)
    new_qubits = end_code.stabilizers.data_set - start_code.stabilizers.data_set
    init_basis = {q: 'XZ'[q.imag < 0] for q in new_qubits}

    builder.append('RX', {q for q, p in init_basis.items() if p == 'X'})
    builder.append('RZ', {q for q, p in init_basis.items() if p == 'Z'})

    discarded = []
    flows = []
    for end_tile in end_code.stabilizers.tiles:
        if not end_tile.data_set:
            continue
        m = end_tile.measure_qubit
        b = end_tile.basis
        start_tile = start_code.stabilizers.m2tile.get(m)
        if start_tile is None:
            start_tile = end_tile.with_edits(data_qubits=[None], bases=end_tile.basis)

        if any(init_basis.get(q, b) != b for q in end_tile.data_set):
            discarded.append(end_tile)
        else:
            flows.append(gen.Flow(
                start=start_tile,
                end=end_tile,
                center=m,
                flags=end_tile.flags,
            ))

    flows.extend(start_code.auto_obs_passthrough_flows(obs_basis=obs_basis, next_code=end_code))

    chunk = gen.Chunk(
        discarded_outputs=discarded,
        circuit=builder.circuit,
        q2i=builder.q2i,
        flows=flows,
    )

    return chunk


def make_hybrid_code_round_chunk(
        *,
        code: gen.StabilizerCode,
        prev_code: gen.StabilizerCode | None = None,
        obs_basis: Literal['X', 'Y', 'Z'],
) -> gen.Chunk:
    builder = gen.Builder.for_qubits(code.used_set)
    bell_pairs = [
        (tile.measure_qubit, tile.measure_qubit + 1)
        for tile in code.stabilizers.tiles
        if tile.basis == 'X'
        if 'cycle=superdense' in tile.flags or 'cycle=bell-flagged-X' in tile.flags or 'cycle=bell-flagged-Z' in tile.flags
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
        if tile.basis == 'X' and ('cycle=superdense' in tile.flags or 'cycle=bell-flagged-X' in tile.flags):
            tile_cx(cx2, m, m - 1)
            tile_cx(cx2, m + 1, m + 2)
            tile_cx(cx3, m, m + 1j)
            tile_cx(cx3, m + 1, m + 1 + 1j)
            tile_cx(cx4, m, m - 1j)
            tile_cx(cx4, m + 1, m + 1 - 1j)
        elif tile.basis == 'Z' and 'cycle=shingle' in tile.flags:
            tile_cx(cx4, m - 1j, m)
            tile_cx(cx5, m + 1, m)
            tile_cx(cx6, m - 1, m)
            tile_cx(cx8, m + 1j, m)
        elif tile.basis == 'X' and 'cycle=shingle' in tile.flags:
            tile_cx(cx4, m, m - 1j)
            tile_cx(cx5, m, m - 1)
            tile_cx(cx6, m, m + 1)
            tile_cx(cx8, m, m + 1j)
        elif 'cycle=shingle-boundary' in tile.flags:
            tile_cx(cx1, m, m - 1j if m - 1j in tile.data_set else m + 1j, rev=tile.basis == 'Z')
            tile_cx(cx2, m, m - 1 if m - 1 in tile.data_set else m + 1, rev=tile.basis == 'Z')
        elif 'cycle=pentagon-above-color-code' in tile.flags:
            cx1.append((m, m + 1))
            cx3.append((m - 1, m - 1 + 1j))
            cx3.append((m, m + 1j))
            cx4.append((m, m - 1j))
            cx5.append((m, m - 1))
            cx6.append((m - 1, m - 1 + 1j))
        elif 'cycle=triangle-above-color-code' in tile.flags:
            cx1.append((m, m + 1))
            cx3.append((m, m - 1j))
            cx4.append((m, m - 1))
        elif tile.basis == 'Z' and ('cycle=superdense' in tile.flags or 'cycle=bell-flagged-Z' in tile.flags):
            m -= 1
            tile_cx(cx5, m, m - 1, rev=True)
            tile_cx(cx5, m + 1, m + 2, rev=True)
            tile_cx(cx6, m, m + 1j, rev=True)
            tile_cx(cx6, m + 1, m + 1 + 1j, rev=True)
            tile_cx(cx7, m, m - 1j, rev=True)
            tile_cx(cx7, m + 1, m + 1 - 1j, rev=True)
        elif tile.basis == 'X' and 'cycle=bell-flagged-Z' in tile.flags:
            # Handled by Z tile.
            pass
        elif tile.basis == 'Z' and 'cycle=bell-flagged-X' in tile.flags:
            # Handled by X tile.
            pass
        elif tile.basis == 'Z' and 'cycle=superdense-boundary-edge' in tile.flags:
            cx3.append((m - 1j, m))
            cx4.append((m + 1, m))
            cx5.append((m - 2j, m - 1j))
            cx6.append((m - 1j, m))
            cx7.append((m - 2j, m - 1j))
        else:
            raise NotImplementedError(f'{tile=}')

    builder.append('RX', code.stabilizers.with_only_x_tiles().measure_set)
    builder.append('RZ', code.stabilizers.with_only_z_tiles().measure_set)
    builder.append("TICK")
    builder.append('CX', cx1)
    builder.append("TICK")

    builder.append('CX', cx2)
    builder.append("TICK")
    builder.append('CX', cx3)
    builder.append("TICK")
    builder.append('CX', cx4)
    builder.append("TICK")
    builder.append('CX', cx5)
    builder.append("TICK")
    builder.append('CX', cx6)
    builder.append("TICK")
    builder.append('CX', cx7)
    builder.append("TICK")

    builder.append('CX', cx8)
    builder.append("TICK")
    builder.append('MX', code.stabilizers.with_only_x_tiles().measure_set)
    builder.append('MZ', code.stabilizers.with_only_z_tiles().measure_set)
    if prev_code is None:
        prev_code = code

    flows = []
    discards = []
    for tile in code.stabilizers.tiles:
        if tile.data_set:
            flows.append(tile.to_prepare_flow('auto'))
        else:
            flows.append(gen.Flow(
                measurement_indices=builder.lookup_recs([tile.measure_qubit]),
                center=tile.measure_qubit,
                flags=tile.flags,
            ))
    for tile in prev_code.stabilizers.tiles:
        if tile.data_set:
            should_discard = False
            if code is not prev_code:
                new_tile = code.stabilizers.m2tile[tile.measure_qubit]
                if not new_tile.data_set or 'lost-to-transition' in new_tile.flags:
                    should_discard = True
            if should_discard:
                discards.append(tile.to_data_pauli_string())
            else:
                flows.append(tile.to_measure_flow('auto'))
    flows.extend(prev_code.auto_obs_passthrough_flows(next_code=code, obs_basis=obs_basis))

    return gen.Chunk(
        circuit=builder.circuit,
        q2i=builder.q2i,
        flows=flows,
        discarded_inputs=discards,
    )
