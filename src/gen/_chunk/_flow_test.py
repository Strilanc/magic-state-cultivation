from ._flow import Flow
from ._pauli_map import PauliMap


def test_with_xz_flipped():
    assert Flow(
        start=PauliMap({1: "X", 2: "Z"}),
        center=0,
    ).with_xz_flipped() == Flow(
        start=PauliMap({1: "Z", 2: "X"}),
        center=0,
    )
