import stim

from gen import InteractLayer
from gen._layers._interact_swap_layer import InteractSwapLayer
from gen._layers._rotation_layer import RotationLayer


def test_to_z_basis():
    layer = InteractSwapLayer(
        i_layer=InteractLayer(
            targets1=[0, 2, 4],
            targets2=[1, 3, 5],
            bases1=["X", "Z", "Z"],
            bases2=["Y", "X", "Z"],
        )
    )
    v = layer.to_z_basis()
    assert v == [
        RotationLayer({0: "H", 1: "H_YZ", 3: "H"}),
        InteractSwapLayer(
            i_layer=InteractLayer(
                targets1=[0, 2, 4],
                targets2=[1, 3, 5],
                bases1=["Z", "Z", "Z"],
                bases2=["Z", "Z", "Z"],
            )
        ),
        RotationLayer({0: "H_YZ", 1: "H", 2: "H"}),
    ]


def test_append_into_circuit():
    layer = InteractSwapLayer(
        i_layer=InteractLayer(
            targets1=[0, 2, 4],
            targets2=[1, 3, 5],
            bases1=["X", "Z", "Z"],
            bases2=["Y", "X", "Z"],
        )
    )
    circuit = stim.Circuit()
    layer.append_into_stim_circuit(circuit)
    assert circuit == stim.Circuit(
        """
        CXSWAP 2 3
        CZSWAP 4 5
        XCY 0 1
        TICK
        SWAP 0 1
        """
    )

    layer = InteractSwapLayer(
        i_layer=InteractLayer(
            targets1=[0, 2, 4],
            targets2=[1, 3, 5],
            bases1=["Z", "Z", "Z"],
            bases2=["Z", "X", "Z"],
        )
    )
    circuit = stim.Circuit()
    layer.append_into_stim_circuit(circuit)
    assert circuit == stim.Circuit(
        """
        CXSWAP 2 3
        CZSWAP 0 1 4 5
        """
    )
