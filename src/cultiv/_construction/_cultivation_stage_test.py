from typing import Any

import pytest
import stim

import gen
from ._color_code import make_color_code
from ._cultivation_stage import make_inject_and_cultivate_chunks_d3, make_inject_and_cultivate_chunks_d5, \
    make_chunk_d3_double_cat_check, make_chunk_d5_double_cat_check, \
    make_chunk_d3_to_d5_color_code
from ._injection_stage import injection_chunk_with_rewritten_injection_rotation


@pytest.mark.parametrize('style', ['degenerate', 'bell', 'unitary'])
def test_make_chunks_d3_inject(style: Any):
    chunks = make_inject_and_cultivate_chunks_d3(style=style)
    circuit = gen.compile_chunks_into_circuit(chunks, add_mpp_boundaries=True, flow_to_extra_coords_func=lambda _: ())
    circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
    err = circuit.search_for_undetectable_logical_errors(
        dont_explore_edges_with_degree_above=3,
        dont_explore_detection_event_sets_with_size_above=3,
        dont_explore_edges_increasing_symptom_degree=False,
        canonicalize_circuit_errors=True,
    )
    assert len(err) == 3


@pytest.mark.parametrize('style', ['degenerate', 'bell', 'unitary'])
def test_make_chunks_d5_inject(style: Any):
    chunks = make_inject_and_cultivate_chunks_d5(style=style)
    gen.compile_chunks_into_circuit(chunks, add_mpp_boundaries=True, flow_to_extra_coords_func=lambda _: ())


def _find_bad_flows_including_sign(chunk: gen.Chunk) -> list[gen.Flow]:
    result = []
    for flow in chunk.flows:
        inp = stim.PauliString(chunk.circuit.num_qubits)
        out = stim.PauliString(chunk.circuit.num_qubits)
        for q, p in flow.start.qubits.items():
            inp[chunk.q2i[q]] = p
        for q, p in flow.end.qubits.items():
            out[chunk.q2i[q]] = p
        if flow.sign:
            out.sign = -1
        stim_flow = stim.Flow(input=inp, output=out, measurements=flow.measurement_indices)
        if not chunk.circuit.has_flow(stim_flow, unsigned=True):
            result.append(('unsigned', flow))
        elif not chunk.circuit.has_flow(stim_flow):
            result.append(flow)
    return result


@pytest.mark.parametrize('turns', [0, 0.5, 1, 1.5])
def test_make_chunk_d3_double_cat_check_signs(turns: float):
    chunk = make_chunk_d3_double_cat_check()
    chunk = injection_chunk_with_rewritten_injection_rotation(chunk, turns)
    chunk.verify(allow_overlapping_flows=True)


@pytest.mark.parametrize('turns', [0, 0.5, 1, 1.5])
def test_make_chunk_d5_double_cat_check_signs(turns: float):
    chunk = make_chunk_d5_double_cat_check()
    chunk = injection_chunk_with_rewritten_injection_rotation(chunk, turns)
    chunk.verify(allow_overlapping_flows=True)


@pytest.mark.parametrize('turns', [0, 0.5, 1, 1.5])
def test_make_chunk_d3_to_d5_color_code_signed(turns: float):
    chunk = make_chunk_d3_to_d5_color_code()
    chunk = injection_chunk_with_rewritten_injection_rotation(chunk, turns)
    chunk.verify(allow_overlapping_flows=True)


@pytest.mark.parametrize('style', ['degenerate', 'bell', 'unitary'])
@pytest.mark.parametrize('turns', [0, 0.5, 1, 1.5])
def test_make_chunks_d3_inject(style: Any, turns: float):
    inject_chunks = make_inject_and_cultivate_chunks_d3(style=style)
    acc = gen.Chunk(stim.Circuit(), flows=[])
    for chunk in inject_chunks:
        acc = acc.then(chunk)
    acc = injection_chunk_with_rewritten_injection_rotation(acc, turns)
    acc.verify(allow_overlapping_flows=True)
    acc.with_edits(flows=[
        gen.Flow(
            end=tile.to_data_pauli_string(),
            center=tile.measure_qubit,
            sign=False,
        )
        for tile in make_color_code(3).tiles
        if (style == 'unitary') or (style == 'degenerate' and tile.basis == 'Z')
    ]).verify()
