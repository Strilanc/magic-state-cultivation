import stim

import gen


def test_mul():
    a = "IIIIXXXXYYYYZZZZ"
    b = "IXYZ" * 4
    c = "IXYZXIZYYZIXZYXI"
    a = gen.PauliMap({q: p for q, p in enumerate(a) if p != "I"})
    b = gen.PauliMap({q: p for q, p in enumerate(b) if p != "I"})
    c = gen.PauliMap({q: p for q, p in enumerate(c) if p != "I"})
    assert a * b == c


def test_init():
    assert gen.PauliMap(stim.PauliString("_XYZ_XX")) == gen.PauliMap(
        {
            "X": [1, 5, 6],
            "Y": [2],
            "Z": [3],
        }
    )
