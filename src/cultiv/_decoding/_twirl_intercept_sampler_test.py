from typing import Any

import pytest
import sinter
import stim

import gen
from cultiv import make_inject_and_cultivate_circuit
from ._twirl_intercept_sampler import TwirlInterceptSampler


@pytest.mark.parametrize('style', ['unitary', 'degenerate', 'bell'])
@pytest.mark.parametrize('turns', [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75])
def test_twirl_sample_on_noiseless_inject_circuit(style: Any, turns: float):
    circuit = make_inject_and_cultivate_circuit(
        dcolor=3,
        inject_style=style,
        basis='Y',
    )
    result = TwirlInterceptSampler(turns=turns).compiled_sampler_for_task(sinter.Task(
        circuit=circuit,
    )).sample(10)
    assert result.discards == 0
    assert result.errors == 0
    assert result.shots == 10


@pytest.mark.parametrize('style', ['unitary', 'degenerate', 'bell'])
@pytest.mark.parametrize('turns', [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75])
def test_twirl_sample_on_noisy_inject_circuit(style: Any, turns: float):
    circuit = make_inject_and_cultivate_circuit(
        dcolor=3,
        inject_style=style,
        basis='Y',
    )
    noisy_circuit = gen.NoiseModel.uniform_depolarizing(1e-3).noisy_circuit_skipping_mpp_boundaries(circuit)
    TwirlInterceptSampler(turns=turns).compiled_sampler_for_task(sinter.Task(
        circuit=noisy_circuit,
    )).sample(10)


@pytest.mark.parametrize('t', [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75])
@pytest.mark.parametrize('error_basis', ['X', 'Y', 'Z'])
def test_twirl_sample_on_simple_circuit(t: float, error_basis: str):
    circuit = stim.Circuit(f"""
        R 0 1 2
        RX 3
        CX 3 0 3 1 3 2
        {error_basis}_ERROR(0.25) 0
        S 0 1 2 3
        {error_basis}_ERROR(0.25) 0
        MPP X0*X1*X2*X3
        MPP Z0*Z1*Z2*Z3
        DETECTOR rec[-1]
        DETECTOR rec[-2]
    """)
    assert circuit.has_flow(stim.Flow("1 -> XXXX"))
    assert circuit.has_flow(stim.Flow("1 -> YYYY"))
    assert circuit.has_flow(stim.Flow("1 -> ZZZZ"))
    sample = TwirlInterceptSampler(turns=t).compiled_sampler_for_task(sinter.Task(
        circuit=circuit,
    )).sample(100_000)
    discard_rate = sample.discards / sample.shots
    if error_basis == 'Z' or t in [0, 1]:
        expected_rate = 0.25 * 0.75 + 0.75 * 0.25
    elif t in [0.25, 0.75, 1.25, 1.75]:
        expected_rate = 0.25 * 0.75 + 0.75 * 0.25 + 0.25 * 0.25 * 0.5
    elif t in [0.5, 1.5]:
        expected_rate = 0.25 * 0.75 + 0.75 * 0.25 + 0.25 * 0.25
    else:
        raise NotImplementedError(f'{t=}')

    assert abs(discard_rate - expected_rate) <= 0.01, (discard_rate, expected_rate)
