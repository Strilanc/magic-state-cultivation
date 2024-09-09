from typing import Literal

import stim

import gen
from ._color_code import make_color_code, make_color_code_grow_chunk, \
    make_chunk_color_code_superdense_cycle
from ._cultivation_stage import make_inject_and_cultivate_chunks_d3, \
    make_inject_and_cultivate_chunks_d5
from ._escape_stage import \
    make_post_escape_matchable_code, \
    make_hybrid_code_round_chunk, make_color_code_to_big_matchable_code_escape_chunks
from ._injection_stage import injection_circuit_with_rewritten_injection_rotation
from ._surface_code import make_surface_code_idle_chunk, make_surface_code


def make_escape_to_big_matchable_code_circuit(
        *,
        dcolor: int,
        dsurface: int,
        basis: Literal['X', 'Y', 'Z'],
        r_growing: int,
        r_end: int,
) -> stim.Circuit:
    chunks = make_color_code_to_big_matchable_code_escape_chunks(
        dcolor=dcolor,
        dsurface=dsurface,
        basis=basis,
        r_growing=r_growing,
        r_end=r_end,
    )
    layer_circuit = gen.LayerCircuit.from_stim_circuit(gen.compile_chunks_into_circuit(
        chunks,
        add_mpp_boundaries=True,
        flow_to_extra_coords_func=flow_to_extra_coords,
    ))
    return layer_circuit.with_locally_optimized_layers().to_stim_circuit()


def make_inject_and_cultivate_circuit(
        *,
        dcolor: int,
        inject_style: Literal['degenerate', 'bell', 'unitary'],
        basis: Literal['X', 'Y'],
) -> stim.Circuit:
    if dcolor == 3:
        inject_chunks = make_inject_and_cultivate_chunks_d3(style=inject_style)
    elif dcolor == 5:
        inject_chunks = make_inject_and_cultivate_chunks_d5(style=inject_style)
    else:
        raise NotImplementedError(f'{dcolor=}')
    layer_circuit = gen.LayerCircuit.from_stim_circuit(gen.compile_chunks_into_circuit(
        inject_chunks,
        add_mpp_boundaries=True,
        flow_to_extra_coords_func=flow_to_extra_coords,
    ))
    result = layer_circuit.with_locally_optimized_layers().to_stim_circuit()
    if basis == 'X':
        result = injection_circuit_with_rewritten_injection_rotation(result, turns=1)
    elif basis == 'Y':
        pass
    else:
        raise NotImplementedError(f'{basis=}')
    return result


def make_end2end_cultivation_circuit(
        *,
        dcolor: int,
        dsurface: int,
        basis: Literal['X', 'Y', 'Z'],
        r_growing: int,
        r_end: int,
        inject_style: Literal['degenerate', 'bell', 'unitary']
) -> stim.Circuit:
    if dcolor == 3:
        inject_chunks = make_inject_and_cultivate_chunks_d3(style=inject_style)
    elif dcolor == 5:
        inject_chunks = make_inject_and_cultivate_chunks_d5(style=inject_style)
    else:
        raise NotImplementedError(f'{dcolor=}')

    chunks = [
        *inject_chunks,
        *make_color_code_to_big_matchable_code_escape_chunks(
            dcolor=dcolor,
            dsurface=dsurface,
            basis=basis,
            r_growing=r_growing,
            r_end=r_end,
        ),
    ]
    c = gen.LayerCircuit.from_stim_circuit(gen.compile_chunks_into_circuit(
        chunks,
        add_mpp_boundaries=True,
        flow_to_extra_coords_func=flow_to_extra_coords,
    ))
    c = c.with_locally_optimized_layers()
    c = c.with_whole_layers_slid_as_to_merge_with_previous_layer_of_same_type(gen.ResetLayer)
    c = c.with_whole_layers_slid_as_early_as_possible_for_merge_with_same_layer(gen.InteractLayer)
    c = c.with_locally_optimized_layers()
    c = c.with_whole_measurement_layers_slid_earlier()
    return c.to_stim_circuit()


def make_idle_matchable_code_circuit(
        *,
        dcolor: int,
        dsurface: int,
        basis: Literal['X', 'Y', 'Z'],
        rounds: int,
) -> stim.Circuit:
    stable_code = make_post_escape_matchable_code(dcolor=dcolor, dsurface=dsurface)
    chunks = [
        make_hybrid_code_round_chunk(code=stable_code, obs_basis=basis) * rounds,
    ]
    layer_circuit = gen.LayerCircuit.from_stim_circuit(gen.compile_chunks_into_circuit(
        chunks,
        add_mpp_boundaries=True,
        flow_to_extra_coords_func=flow_to_extra_coords,
    ))
    return layer_circuit.with_locally_optimized_layers().to_stim_circuit()


BASIS_COLOR_TO_EXTRA_COORDS = {
    'Xr': (0, 0),
    'Xg': (1, 1),
    'Xb': (2, 2),
    'Zr': (3, 3),
    'Zg': (4, 4),
    'Zb': (5, 5),
    'X_': (0, 6),
    'Z_': (3, 7),
}


def flow_to_extra_coords(flow: gen.Flow) -> list[float]:
    colors = set()
    bases = set()
    for flag in flow.flags:
        if flag.startswith('color='):
            colors.add(flag[len('color='):])
        if flag.startswith('basis='):
            bases.add(flag[len('basis='):])

    assert 'postselect' not in flow.flags
    if 'stage=cultivation' in flow.flags or 'stage=injection' in flow.flags:
        return [-1, -9]
    if len(bases) != 1 or len(colors) > 1:
        raise NotImplementedError(f'{flow=}')

    basis, = bases
    color, = colors or '_'
    coords = list(BASIS_COLOR_TO_EXTRA_COORDS[basis + color])
    return coords


def make_escape_to_big_color_code_circuit(*, start_width: int, end_width: int, rounds: int, basis: Literal['X', 'Y', 'Z']) -> stim.Circuit:
    chunks = [
        make_color_code_grow_chunk(start_width, end_width, basis=basis),
        make_chunk_color_code_superdense_cycle(make_color_code(end_width), obs_basis=basis).time_reversed() * rounds,
    ]
    return gen.compile_chunks_into_circuit(
        chunks,
        add_mpp_boundaries=True,
        flow_to_extra_coords_func=flow_to_extra_coords,
    )


def make_surface_code_memory_circuit(*, dsurface: int, rounds: int, basis: Literal['X', 'Y', 'Z']) -> stim.Circuit:
    code = make_surface_code(width=dsurface, height=dsurface)
    init = code.with_observables_from_basis('Y').mpp_init_chunk() if basis == 'Y' else code.transversal_init_chunk(basis=basis)
    chunks = [
        init,
        make_surface_code_idle_chunk(code=code, basis=basis) * rounds,
        init.time_reversed(),
    ]
    circuit = gen.compile_chunks_into_circuit(chunks)
    circuit = gen.LayerCircuit.from_stim_circuit(circuit)
    circuit = circuit.with_locally_optimized_layers()
    return circuit.to_stim_circuit()
