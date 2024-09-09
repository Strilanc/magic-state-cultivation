import stim

import gen
from gen._chunk._flow_util import solve_flow_auto_measurements


def test_solve_flow_auto_measurements():
    assert (
        solve_flow_auto_measurements(
            flows=[
                gen.Flow(
                    start=gen.PauliMap({"Z": [0 + 1j, 2 + 1j]}),
                    measurement_indices="auto",
                    center=-1,
                    flags={"X"},
                )
            ],
            circuit=stim.Circuit(
                """
            R 1
            CX 0 1 2 1
            M 1
        """
            ),
            q2i={0 + 1j: 0, 1 + 1j: 1, 2 + 1j: 2},
        )
        == (
            gen.Flow(
                start=gen.PauliMap({"Z": [0 + 1j, 2 + 1j]}),
                measurement_indices=[0],
                center=-1,
                flags={"X"},
            ),
        )
    )
