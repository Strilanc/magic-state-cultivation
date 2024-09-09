import pytest
import stim

import gen
from ._color_code import make_color_code, \
    make_chunk_color_code_superdense_cycle
from ._injection_stage import make_chunk_d3_init_unitary, \
    make_chunk_d3_init_degenerate_teleport, make_chunk_d3_init_bell_pair_growth, \
    injection_chunk_with_rewritten_injection_rotation


@pytest.mark.parametrize('turns', [0, 0.5, 1, 1.5])
def test_injection_chunk_with_rewritten_injection_rotation(turns: float):
    chunk = gen.Chunk(
        circuit=stim.Circuit("""
            QUBIT_COORDS(0, 0) 0
            RX 0
            S 0
        """),
        flows=[gen.Flow(
            end=gen.PauliMap(ys=[0]),
            center=0,
        )],
    )
    chunk = injection_chunk_with_rewritten_injection_rotation(chunk, turns)
    chunk.verify(allow_overlapping_flows=True)


@pytest.mark.parametrize('turns', [0, 0.5, 1, 1.5])
def test_make_chunk_d3_init_bell_pair_growth_signs(turns: float):
    chunk = make_chunk_d3_init_bell_pair_growth()
    chunk = injection_chunk_with_rewritten_injection_rotation(chunk, turns)
    chunk.verify(allow_overlapping_flows=True)


@pytest.mark.parametrize('turns', [0, 0.5, 1, 1.5])
def test_make_chunk_color_code_superdense_cycle_signs(turns: float):
    chunk = make_chunk_color_code_superdense_cycle(make_color_code(3), obs_basis='Y')
    chunk = injection_chunk_with_rewritten_injection_rotation(chunk, turns)
    chunk.verify(allow_overlapping_flows=True)


@pytest.mark.parametrize('turns', [0, 0.5, 1, 1.5])
def test_make_chunk_d3_init_degenerate_teleport_signs(turns: float):
    chunk = make_chunk_d3_init_degenerate_teleport()
    chunk = injection_chunk_with_rewritten_injection_rotation(chunk, turns)
    chunk.verify(allow_overlapping_flows=True)


@pytest.mark.parametrize('turns', [0, 0.5, 1, 1.5])
def test_make_chunk_d3_init_unitary_signs(turns: float):
    chunk = make_chunk_d3_init_unitary()
    chunk = injection_chunk_with_rewritten_injection_rotation(chunk, turns)
    chunk.verify(allow_overlapping_flows=True)
