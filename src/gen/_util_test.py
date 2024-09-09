from gen._util import xor_sorted


def test_xor_sorted():
    assert xor_sorted([]) == []
    assert xor_sorted([2]) == [2]
    assert xor_sorted([2, 3]) == [2, 3]
    assert xor_sorted([3, 2]) == [2, 3]
    assert xor_sorted([2, 2]) == []
    assert xor_sorted([2, 2, 2]) == [2]
    assert xor_sorted([2, 2, 2, 2]) == []
    assert xor_sorted([2, 2, 3]) == [3]
    assert xor_sorted([3, 2, 2]) == [3]
    assert xor_sorted([2, 3, 2]) == [3]
    assert xor_sorted([2, 3, 3]) == [2]
    assert xor_sorted([2, 3, 5, 7, 11, 13, 5]) == [2, 3, 7, 11, 13]
