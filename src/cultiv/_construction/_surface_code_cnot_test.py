from typing import Any

import pytest

import gen
from cultiv._construction._surface_code_cnot import \
    make_surface_code_cnot


@pytest.mark.parametrize('distance', [3, 4, 5, 7])
@pytest.mark.parametrize('basis', ['X', 'Z'])
def test_make_surface_code_cnot(distance: int, basis: Any):
    circuit = make_surface_code_cnot(distance=distance, basis=basis)
    print(circuit)
    circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
    circuit.detector_error_model(decompose_errors=True)
    assert len(circuit.shortest_graphlike_error()) == distance
    # assert circuit.num_detectors + circuit.num_observables == circuit.count_determined_measurements()
