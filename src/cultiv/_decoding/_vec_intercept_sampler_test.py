from typing import Any

import pytest
import sinter

from cultiv import make_inject_and_cultivate_circuit
from ._vec_intercept_sampler import VecInterceptSampler


@pytest.mark.parametrize('style', ['unitary', 'bell'])
@pytest.mark.parametrize('turns', [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75])
def test_vec_intercept_sampler(style: Any, turns: float):
    circuit = make_inject_and_cultivate_circuit(
        dcolor=3,
        inject_style=style,
        basis='Y',
    )
    tampler = VecInterceptSampler(turns=turns, sweep_bit_randomization=True)
    compiled = tampler.compiled_sampler_for_task(sinter.Task(
        circuit=circuit,
    ))
    result = compiled.sample(10)
    assert result.discards == 0
    assert result.errors == 0
    assert result.shots == 10
