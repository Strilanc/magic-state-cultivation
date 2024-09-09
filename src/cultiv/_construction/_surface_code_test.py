from typing import Any

import pytest

from ._surface_code import make_surface_code, make_surface_code_idle_chunk


@pytest.mark.parametrize('w', [3, 4, 5])
@pytest.mark.parametrize('h', [3, 4, 5])
def test_make_surface_code(w: int, h: int):
    code = make_surface_code(w, h)
    code.verify()
    for tile in code.tiles:
        if tile.basis == 'Z':
            assert 'basis=Z' in tile.flags
        else:
            assert tile.basis == 'X'
            assert 'basis=X' in tile.flags


@pytest.mark.parametrize('w', [3, 4, 5])
@pytest.mark.parametrize('h', [3, 4, 5])
@pytest.mark.parametrize('b', ['X', 'Y', 'Z'])
def test_make_surface_code_idle_chunk(w: int, h: int, b: Any):
    code = make_surface_code(w, h)
    chunk = make_surface_code_idle_chunk(code, basis=b)
    chunk.verify()
    ds = []
    if b != 'X':
        ds.append(h)
    if b != 'Z':
        ds.append(w)
    expected_d = min(ds)
    actual_d = chunk.find_distance(max_search_weight=2)
    assert actual_d == expected_d
