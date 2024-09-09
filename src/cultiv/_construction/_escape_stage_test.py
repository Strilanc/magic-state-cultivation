from typing import Any

import pytest

import gen
from ._color_code import make_color_code
from ._escape_stage import \
    make_hybrid_color_surface_code, \
    make_post_escape_matchable_code, \
    make_hybrid_code_round_chunk, \
    make_color_code_to_growing_code_chunk, \
    make_color_code_grown_into_surface_code_then_ablated_into_matchable_code_simple, \
    make_color_code_grown_into_surface_code_then_ablated_into_matchable_code_full_edges, \
    make_color_code_to_big_matchable_code_escape_chunks, make_color_code_all_to_transition_reflow


def test_make_color_code_all_to_transition_reflow():
    code1 = make_color_code(base_width=5, obs_location='all')
    code2 = make_color_code(base_width=5, obs_location='x-top-z-bottom-right')
    make_color_code_all_to_transition_reflow(code1, code2).verify()


def test_make_color_code_to_big_matchable_code_escape_chunks():
    chunks = make_color_code_to_big_matchable_code_escape_chunks(dcolor=5, dsurface=11, basis='Y', r_growing=3, r_end=2)
    for chunk in chunks:
        chunk.verify()
    gen.compile_chunks_into_circuit(chunks, add_mpp_boundaries=True)


@pytest.mark.parametrize('dcolor,dsurface', [(3, 6), (3, 7), (5, 10), (7, 14)])
@pytest.mark.parametrize('obs_location', ['left', 'right', 'transition'])
def test_make_hybrid_color_surface_code(dcolor: int, dsurface: int, obs_location: Any):
    code = make_hybrid_color_surface_code(dcolor=dcolor, dsurface=dsurface, obs_location=obs_location)
    code.verify()
    for tile in code.tiles:
        if tile.basis == 'Z':
            assert 'basis=Z' in tile.flags
        else:
            assert tile.basis == 'X'
            assert 'basis=X' in tile.flags
        if tile.basis == 'Z' and 'cycle=superdense' in tile.flags:
            assert code.stabilizers.m2tile[tile.measure_qubit - 1].basis == 'X'
        if tile.basis == 'X' and 'cycle=superdense' in tile.flags:
            assert code.stabilizers.m2tile[tile.measure_qubit + 1].basis == 'Z'
    assert code.find_distance(max_search_weight=3) == dsurface


@pytest.mark.parametrize('dcolor,dsurface', [(3, 6), (3, 7), (5, 10), (7, 14)])
def test_make_post_escape_matchable_code(dcolor: int, dsurface: int):
    code = make_post_escape_matchable_code(dcolor=dcolor, dsurface=dsurface)
    code.verify()
    for tile in code.stabilizers.tiles:
        if tile.basis == 'Z':
            assert 'basis=Z' in tile.flags
        else:
            assert tile.basis == 'X'
            assert 'basis=X' in tile.flags
    assert code.find_distance(max_search_weight=3) == dsurface


@pytest.mark.parametrize('dcolor,dsurface', [(3, 6), (5, 10), (3, 7)])
@pytest.mark.parametrize('basis', ['X', 'Z'])
def test_make_color_code_to_growing_code_chunk(dcolor: int, dsurface: int, basis: Any):
    code1 = make_color_code(dcolor, obs_location='x-top-z-bottom-right')
    code2 = make_hybrid_color_surface_code(dcolor=dcolor, dsurface=dsurface, obs_location='transition')
    code3 = make_hybrid_color_surface_code(dcolor=dcolor, dsurface=dsurface, obs_location='right')
    chunk1 = make_color_code_to_growing_code_chunk(
        start_code=code1,
        end_code=code2,
        obs_basis=basis,
    )
    chunk1.verify()
    chunk2 = make_hybrid_code_round_chunk(
        prev_code=code2,
        code=code3,
        obs_basis=basis,
    )
    chunk2.verify()


@pytest.mark.parametrize('dcolor,dsurface', [(3, 6), (5, 10), (3, 7)])
@pytest.mark.parametrize('basis', ['X', 'Z'])
@pytest.mark.parametrize('obs_location', ['left', 'right', 'transition'])
def test_make_hybrid_code_round_chunk(dcolor: int, dsurface: int, basis: Any, obs_location: Any):
    code1 = make_hybrid_color_surface_code(dcolor=dcolor, dsurface=dsurface, obs_location=obs_location)
    chunk1 = make_hybrid_code_round_chunk(
        code=code1,
        obs_basis=basis,
    )
    chunk1.verify(
        expected_in=code1.with_observables_from_basis(basis).as_interface(),
        expected_out=code1.with_observables_from_basis(basis).as_interface(),
    )
    assert chunk1.find_distance(max_search_weight=2) == dsurface

    code2 = make_post_escape_matchable_code(dcolor=dcolor, dsurface=dsurface)
    chunk2 = make_hybrid_code_round_chunk(
        code=code2,
        obs_basis=basis,
    )
    chunk2.verify(
        expected_in=code2.with_observables_from_basis(basis).as_interface(),
        expected_out=code2.with_observables_from_basis(basis).as_interface(),
    )
    assert chunk2.find_distance(max_search_weight=2) == dsurface

    chunk12 = make_hybrid_code_round_chunk(
        code=code2,
        prev_code=code1,
        obs_basis=basis,
    )
    chunk12.verify(
        expected_in=code1.with_observables_from_basis(basis).as_interface(),
        expected_out=code2.with_observables_from_basis(basis).as_interface(),
    )
    assert chunk12.find_distance(max_search_weight=2) == dsurface

    if dcolor <= 5:
        expected_distance = dsurface
        if dcolor == 5 and basis == 'Z':
            expected_distance -= 2
        assert chunk12.find_distance(max_search_weight=3) == expected_distance


@pytest.mark.parametrize('dcolor,dsurface', [(3, 6), (3, 7), (5, 11), (7, 14)])
def test_make_color_code_grown_into_surface_code_then_ablated_into_matchable_code_simple(dcolor: int, dsurface: int):
    c = make_color_code_grown_into_surface_code_then_ablated_into_matchable_code_simple(dcolor=dcolor, dsurface=dsurface, desaturate=True)
    assert c.make_code_capacity_circuit(noise=1e-3).detector_error_model(decompose_errors=True) is not None
    d = c.find_distance(max_search_weight=2)
    assert d == dsurface - dcolor + 1


@pytest.mark.parametrize('dcolor,dsurface', [(3, 6), (3, 7), (5, 11), (7, 14)])
def test_make_color_code_grown_into_surface_code_then_ablated_into_matchable_code_full_edges(dcolor: int, dsurface: int):
    c = make_color_code_grown_into_surface_code_then_ablated_into_matchable_code_full_edges(dcolor=dcolor, dsurface=dsurface, desaturate=True)
    assert c.make_code_capacity_circuit(noise=1e-3).detector_error_model(decompose_errors=True) is not None
    d = c.find_distance(max_search_weight=2)
    assert d == dsurface
