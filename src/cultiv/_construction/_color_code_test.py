from typing import Any

import pytest

import gen
from ._color_code import tile_rgb_color, make_color_code, \
    make_chunk_color_code_superdense_cycle, \
    make_growing_color_code_bell_pair_patch, make_color_code_grow_chunk


def test_tile_rgb_color_finishes():
    tile_rgb_color(gen.Tile(data_qubits=[0], measure_qubit=0, bases='X'))


@pytest.mark.parametrize('d', [3, 5, 7])
@pytest.mark.parametrize('obs_location', ['all', 'top', 'bottom-right', 'bottom-left', 'x-top-z-bottom-right'])
def test_make_color_code(d: int, obs_location: Any):
    code = make_color_code(d, obs_location=obs_location)
    code.verify()
    assert code.find_distance(max_search_weight=3) == d
    for tile in code.tiles:
        if tile.basis == 'Z':
            assert 'basis=Z' in tile.flags
            assert code.stabilizers.m2tile[tile.measure_qubit - 1] == tile.with_edits(bases='X', flags=tile.flags ^ {'basis=X', 'basis=Z'}, measure_qubit=tile.measure_qubit - 1)
        else:
            assert tile.basis == 'X'
            assert 'basis=X' in tile.flags
    if obs_location != 'x-top-z-bottom-right':
        assert code.logicals[0][0].qubits.keys() == code.logicals[0][1].qubits.keys()


@pytest.mark.parametrize('d', [3, 5])
@pytest.mark.parametrize('basis', ['X', 'Y', 'Z'])
@pytest.mark.parametrize('check_x', [False, True])
@pytest.mark.parametrize('check_z', [False, True])
def test_make_color_code_superdense_cycle(d: int, basis: Any, check_x: bool, check_z: bool):
    code = make_color_code(d)
    chunk = make_chunk_color_code_superdense_cycle(code, obs_basis=basis, check_x=check_x, check_z=check_z)
    chunk.verify()


def test_make_growing_color_code_patch():
    grow = make_growing_color_code_bell_pair_patch(5, 11)
    end = make_color_code(11)
    assert grow.used_set <= end.used_set
    assert grow.data_set == end.stabilizers.data_set


def test_make_grow_chunk():
    chunk = make_color_code_grow_chunk(5, 11, basis='X')
    chunk.verify(
        expected_in=make_color_code(5).with_observables_from_basis('X'),
        expected_out=make_color_code(11).with_observables_from_basis('X'),
    )


def test_x():
    pairs = gen.Patch([
        tile
        for tile in make_growing_color_code_bell_pair_patch(5, 9)
        if len(tile.data_set) == 2
    ])
    before = make_color_code(5).stabilizers
    after = make_color_code(9).stabilizers
    patch = gen.Patch([
        tile.with_edits(measure_qubit=None)
        for tile in pairs
        if len(tile.data_set) == 2
    ] + [
        tile
        for tile in after.tiles
        if tile.basis == 'Z'
        if tile.measure_qubit in before.measure_set or 'color=b' in tile.flags or 'color=g' in tile.flags
    ])
    patch.write_svg(
        'tmp.svg',
        tile_color_func=lambda e: (0.5, 1, 1) if len(e.data_set) == 2 else tile_rgb_color(e),
        show_coords=False,
    )
