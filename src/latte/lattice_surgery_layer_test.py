import itertools
from typing import Iterable, Any

import pytest
import stim

import gen
from latte.lattice_surgery_layer import LatticeSurgeryLayer, InjectedError
from latte.lattice_surgery_instruction import LatticeSurgeryInstruction, ErrorSource
from latte.zx_graph import ZXGraph, ZXNode, FeedbackTargets, ExternalStabilizer
from latte.vec_sim import VecSim


def test_y_prep():
    layer = LatticeSurgeryLayer.from_text(r"""
         /
        Y
    """)
    g = ZXGraph.from_nx_graph(layer.to_nx_zx_graph())
    assert g.nodes == (
        ZXNode(key=(0j, 'future_out'), index=0, kind='out', phase=1),
        ZXNode(key=(0j, 'prep'), index=1, kind='Z', phase=1j),
    )
    assert g.external_stabilizers == (
        ExternalStabilizer(
            inp=gen.PauliMap({}),
            interior=gen.PauliMap({0j: 'Y'}),
            is_measurement=False,
            out=gen.PauliMap({0j: 'Y'}),
            paulis=stim.PauliString("+YYYY"),
            port_stabilizer=stim.PauliString("+Y"),
            sign=-1,
            sink_edge_bases=(),
            sink_edge_cols=(),
            sink_edge_keys=(),
        ),
    )
    assert layer.to_sim_instructions(layer_key='X') == (
        LatticeSurgeryInstruction('qalloc_y', target=0),
    )


def test_y_measure():
    assert LatticeSurgeryLayer.from_text(r"""
         Y
        /
    """).to_sim_instructions(layer_key='V') == (
        LatticeSurgeryInstruction('m_discard_y',
                                  target=0j,
                                  measure_key=('MY', 0j, 'V')),
        LatticeSurgeryInstruction('accumulator_bit_clear', target=None, target2=None, measure_key=None),
        LatticeSurgeryInstruction('accumulator_bit_xor',
                                  measure_key=('MY', 0j, 'V')),
        LatticeSurgeryInstruction('accumulator_bit_save',
                                  measure_key=('logical', 0, 'V')),
    )


def test_yy_xx_zz():
    layers = [
        r"""
             /     /
            Y     Y
        """,
        r"""
             /     /
            X--@--X
           /     /
        """,
        r"""

            Z--@*-Z
           /     /
        """,
    ]
    graphs = [
        LatticeSurgeryLayer.from_text(layer)
        for layer in layers
    ]
    for _ in range(10):
        sim = VecSim()
        for k, g in enumerate(graphs):
            sim.do_lattice_surgery_layer(g, layer_key=k)
        m = sim.m_record
        assert m[('MXX', 0.5, 1)] ^ m[('MZZ', 0.5, 2)] == True
        assert m[('MXX', 0.5, 1)] ^ m[('MX', 0, 2)] ^ m[('MX', 1, 2)] == False
        assert m[('MZZ', 0.5, 2)] ^ m[('MX', 0, 2)] ^ m[('MX', 1, 2)] == True


def test_t_inject_fusion():
    layer0 = LatticeSurgeryLayer.from_text(r"""
           /    /
          T    T
    """)
    layer1 = LatticeSurgeryLayer.from_text(r"""
                /
          Z-*--Z
         /    /
    """)
    for tt in range(30):
        sim = VecSim()
        () = sim.do_lattice_surgery_layer(layer0, layer_key=0)
        tzz, = sim.do_lattice_surgery_layer(layer1, layer_key=1)
        if tzz:
            sim.do_s(1)
        sim.do_s(1)
        assert sim.do_mx(1, key=None), tt


def test_s_inject_tasks():
    assert LatticeSurgeryLayer.from_text(r"""
           /
          S
    """).to_sim_instructions(layer_key='A') == (LatticeSurgeryInstruction('qalloc_x',
                                                                          target=0j,
                                                                          target2=None,
                                                                          measure_key=None),
                                                LatticeSurgeryInstruction('s', target=0j, target2=None, measure_key=None),
                                                LatticeSurgeryInstruction('heralded_random_x',
                                                                          target=0j,
                                                                          target2=None,
                                                                          measure_key=('heralded_random_x', 0j, 'A')),
                                                LatticeSurgeryInstruction('heralded_random_z',
                                                                          target=0j,
                                                                          target2=None,
                                                                          measure_key=('heralded_random_z', 0j, 'A')),
                                                LatticeSurgeryInstruction('feedback_m2x',
                                                                          target=0j,
                                                                          target2=None,
                                                                          measure_key=('heralded_random_x', 0j, 'A')),
                                                LatticeSurgeryInstruction('feedback_m2z',
                                                                          target=0j,
                                                                          target2=None,
                                                                          measure_key=('heralded_random_z', 0j, 'A')))


def test_s_inject_fusion():
    layer0 = LatticeSurgeryLayer.from_text(r"""
           /    /
          S    S
    """)
    layer1 = LatticeSurgeryLayer.from_text(r"""
                /
          Z-*--Z
         /    /
    """)
    for tt in range(30):
        sim = VecSim()
        () = sim.do_lattice_surgery_layer(layer0, layer_key=0)
        tzz, = sim.do_lattice_surgery_layer(layer1, layer_key=1)
        if tzz:
            sim.do_z(1)
        assert sim.do_mx(1, key=None), tt


def test_t_inject_2():
    layer0 = LatticeSurgeryLayer.from_text(r"""
                /
          .    Z
    """)
    layer1 = LatticeSurgeryLayer.from_text(r"""
                /
          T--*-Z
              /
    """)
    for _ in range(30):
        sim = VecSim()
        () = sim.do_lattice_surgery_layer(layer0, layer_key=0)
        tzz, = sim.do_lattice_surgery_layer(layer1, layer_key=1)
        if tzz:
            sim.do_s(1)
        sim.do_t_dag(1)
        assert not sim.do_mx(1, key=None)


def test_graph_h_to_tasks():
    assert LatticeSurgeryLayer.from_text(r"""
         /
        H
       /
    """).to_sim_instructions(layer_key='A') == (
        LatticeSurgeryInstruction('h', target=0j),
    )


def test_graph_to_tasks():
    assert LatticeSurgeryLayer.from_text(r"""
         /
        Z
       /
    """).to_sim_instructions(layer_key='A') == ()

    assert LatticeSurgeryLayer.from_text(r"""
              /
        Z----Z
       /
    """).to_sim_instructions(layer_key='A') == (
        LatticeSurgeryInstruction('qalloc_x', target=1),
        LatticeSurgeryInstruction('mzz', target=0, target2=1, measure_key=('MZZ', 0.5, 'A')),
        LatticeSurgeryInstruction('m_discard_x', target=0, measure_key=('MX', 0, 'A')),
        LatticeSurgeryInstruction('feedback_m2x', target=1, measure_key=('MZZ', (0.5 + 0), 'A')),
        LatticeSurgeryInstruction('feedback_m2z', target=1, measure_key=('MX', 0, 'A')),
    )

    assert LatticeSurgeryLayer.from_text(r"""
         /    /
        X--@-X
       /    /
    """).to_sim_instructions(layer_key='A') == (
        LatticeSurgeryInstruction('mxx', target=0, target2=1, measure_key=('MXX', (0.5 + 0), 'A')),
        LatticeSurgeryInstruction('accumulator_bit_clear'),
        LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=('MXX', (0.5 + 0), 'A')),
        LatticeSurgeryInstruction('accumulator_bit_save', measure_key=('logical', 0, 'A'))
    )

    assert LatticeSurgeryLayer.from_text(r"""
         /
        Z-*-T
       /     
    """).to_sim_instructions(layer_key='A') == (
        LatticeSurgeryInstruction('qalloc_x', target=1),
        LatticeSurgeryInstruction('t', target=1),
        LatticeSurgeryInstruction('heralded_random_x', target=1, measure_key=('heralded_random_x', 1, 'A')),
        LatticeSurgeryInstruction('heralded_random_z', target=1, measure_key=('heralded_random_z', 1, 'A')),
        LatticeSurgeryInstruction('mzz', target=0, target2=1, measure_key=('MZZ', (0.5 + 0), 'A')),
        LatticeSurgeryInstruction('m_discard_x', target=1, measure_key=('MX', 1, 'A')),
        LatticeSurgeryInstruction('feedback_m2z', target=0, measure_key=('heralded_random_z', 1, 'A')),
        LatticeSurgeryInstruction('feedback_m2z', target=0, measure_key=('MX', 1, 'A')),
        LatticeSurgeryInstruction('accumulator_bit_clear'),
        LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=('heralded_random_x', 1, 'A')),
        LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=('MZZ', (0.5 + 0), 'A')),
        LatticeSurgeryInstruction('accumulator_bit_save', measure_key=('logical', 0, 'A')),
    )

    assert LatticeSurgeryLayer.from_text(r"""
         /    /
        X--@-X---Z
       /        /
    """).to_sim_instructions(layer_key='A') == (
        LatticeSurgeryInstruction('qalloc_z', target=1),
        LatticeSurgeryInstruction('cx', target=2, target2=1),
        LatticeSurgeryInstruction('m_discard_x', target=2, measure_key=('MX', 2, 'A')),
        LatticeSurgeryInstruction('mxx', target=0, target2=1, measure_key=('MXX', 0.5, 'A')),
        LatticeSurgeryInstruction('feedback_m2z', target=1, measure_key=('MX', (2 + 0j), 'A')),
        LatticeSurgeryInstruction('accumulator_bit_clear'),
        LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=('MX', (2 + 0), 'A')),
        LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=('MXX', (0.5 + 0), 'A')),
        LatticeSurgeryInstruction('accumulator_bit_save', measure_key=('logical', 0, 'A')),
    )

    assert LatticeSurgeryLayer.from_text(r"""
        .    .    .    .

         /    /    /    /
        X----X----X----X



        .    .    .    .


         /    /    /    /
        X----X----X----X

        .    .    .    .
    """).to_sim_instructions(layer_key='B') == (
        LatticeSurgeryInstruction('qalloc_z', 1j),
        LatticeSurgeryInstruction('qalloc_z', 1 + 1j),
        LatticeSurgeryInstruction('mxx', 1j, 1 + 1j, measure_key=('MXX', (0.5 + 1j), 'B')),
        LatticeSurgeryInstruction('qalloc_z', 3j),
        LatticeSurgeryInstruction('qalloc_z', (1 + 3j)),
        LatticeSurgeryInstruction('mxx', 3j, (1 + 3j), measure_key=('MXX', (0.5 + 3j), 'B')),
        LatticeSurgeryInstruction('qalloc_z', 3 + 1j),
        LatticeSurgeryInstruction('qalloc_z', 2 + 1j),
        LatticeSurgeryInstruction('mxx', 2 + 1j, 3 + 1j, measure_key=('MXX', (2.5 + 1j), 'B')),
        LatticeSurgeryInstruction('qalloc_z', (3 + 3j)),
        LatticeSurgeryInstruction('qalloc_z', (2 + 3j)),
        LatticeSurgeryInstruction('mxx', (2 + 3j), (3 + 3j), measure_key=('MXX', (2.5 + 3j), 'B')),
        LatticeSurgeryInstruction('mxx', 1 + 1j, 2 + 1j, measure_key=('MXX', (1.5 + 1j), 'B')),
        LatticeSurgeryInstruction('mxx', (1 + 3j), (2 + 3j), measure_key=('MXX', (1.5 + 3j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', 1 + 1j, measure_key=('MXX', (0.5 + 1j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (2 + 1j), measure_key=('MXX', (0.5 + 1j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (3 + 1j), measure_key=('MXX', (0.5 + 1j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (1 + 3j), measure_key=('MXX', (0.5 + 3j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (2 + 3j), measure_key=('MXX', (0.5 + 3j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (3 + 3j), measure_key=('MXX', (0.5 + 3j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (3 + 1j), measure_key=('MXX', (2.5 + 1j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (3 + 3j), measure_key=('MXX', (2.5 + 3j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (2 + 1j), measure_key=('MXX', (1.5 + 1j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (3 + 1j), measure_key=('MXX', (1.5 + 1j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (2 + 3j), measure_key=('MXX', (1.5 + 3j), 'B')),
        LatticeSurgeryInstruction('feedback_m2z', (3 + 3j), measure_key=('MXX', (1.5 + 3j), 'B')),
    )

    assert LatticeSurgeryLayer.from_text(r"""
        .    .    .    .

         /    /    /    /
        Z    Z    Z    Z
       /    /|   /|   /
             |    |
             |    |
        .    X--*-X    .
             |    |
             |    |
         /   |/   |/    /
        Z    Z    Z    Z
       /    /    /    /

        .    .    .    .
    """).to_sim_instructions(layer_key='C') == (
        LatticeSurgeryInstruction('qalloc_z', (1 + 2j)),
        LatticeSurgeryInstruction('cx', 1 + 1j, (1 + 2j)),
        LatticeSurgeryInstruction('cx', (1 + 3j), (1 + 2j)),
        LatticeSurgeryInstruction('qalloc_z', (2 + 2j)),
        LatticeSurgeryInstruction('mxx', (1 + 2j), (2 + 2j), measure_key=('MXX', (1.5 + 2j), 'C')),
        LatticeSurgeryInstruction('m_discard_z', (1 + 2j), measure_key=('MZ', (1 + 2j), 'C')),
        LatticeSurgeryInstruction('cx', 2 + 1j, (2 + 2j)),
        LatticeSurgeryInstruction('cx', (2 + 3j), (2 + 2j)),
        LatticeSurgeryInstruction('m_discard_z', (2 + 2j), measure_key=('MZ', (2 + 2j), 'C')),
        LatticeSurgeryInstruction('feedback_m2z', target=(2 + 1j), measure_key=('MXX', (1.5 + 2j), 'C')),
        LatticeSurgeryInstruction('feedback_m2z', target=(2 + 3j), measure_key=('MXX', (1.5 + 2j), 'C')),
        LatticeSurgeryInstruction('accumulator_bit_clear'),
        LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=('MZ', (1 + 2j), 'C')),
        LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=('MZ', (2 + 2j), 'C')),
        LatticeSurgeryInstruction('accumulator_bit_save', measure_key=('logical', 0, 'C')),
    )
    assert LatticeSurgeryLayer.from_text(r"""
        .    .    .    .

         /    /    /    /
        Z    Z    Z    Z
       /|   /|   /    /
        |    |
        |    |
        X-*--X    .    .
        |    |
        |    |
        |/   |/    /    /
        Z    Z    Z    Z
       /    /    /    /

        .    .    .    .
    """).to_sim_instructions(layer_key='C') == (
        LatticeSurgeryInstruction('qalloc_z', 2j),
        LatticeSurgeryInstruction('cx', 1j, 2j),
        LatticeSurgeryInstruction('cx', 3j, 2j),
        LatticeSurgeryInstruction('qalloc_z', (1 + 2j)),
        LatticeSurgeryInstruction('mxx', 2j, (1 + 2j), measure_key=('MXX', (0.5 + 2j), 'C')),
        LatticeSurgeryInstruction('m_discard_z', 2j, measure_key=('MZ', 2j, 'C')),
        LatticeSurgeryInstruction('cx', 1 + 1j, (1 + 2j)),
        LatticeSurgeryInstruction('cx', (1 + 3j), (1 + 2j)),
        LatticeSurgeryInstruction('m_discard_z', (1 + 2j), measure_key=('MZ', (1 + 2j), 'C')),
        LatticeSurgeryInstruction('feedback_m2z',
                                  target=(1+1j),
                                  measure_key=('MXX', (0.5+2j), 'C')),
        LatticeSurgeryInstruction('feedback_m2z',
                                  target=(1+3j),
                                  measure_key=('MXX', (0.5+2j), 'C')),
        LatticeSurgeryInstruction('accumulator_bit_clear'),
        LatticeSurgeryInstruction('accumulator_bit_xor',
                                  measure_key=('MZ', 2j, 'C')),
        LatticeSurgeryInstruction('accumulator_bit_xor',
                                  measure_key=('MZ', (1+2j), 'C')),
        LatticeSurgeryInstruction('accumulator_bit_save',
                                  measure_key=('logical', 0, 'C')),
    )


def test_to_nx_zx_graph():
    g = LatticeSurgeryLayer.from_text(r"""
              /    /    /
        .    Z    Z    Z    .
            /|   /    /|
             |         |     /
        T----X----X----X----Z
                           /
    """).to_nx_zx_graph()
    assert dict(g.nodes) == {
        1: {'phase': 1, 'type': 'Z'},
        (2+0): {'phase': 1, 'type': 'Z'},
        (3+0): {'phase': 1, 'type': 'Z'},
        1j: {'phase': (0.7071067811865476+0.7071067811865475j), 'type': 'T_port'},
        (1+1j): {'phase': 1, 'type': 'X'},
        (2+1j): {'phase': 1, 'type': 'X'},
        (3+1j): {'phase': 1, 'type': 'X'},
        (4+1j): {'phase': 1, 'type': 'Z'},
        ((4+1j), 'future_out'): {'phase': 1, 'type': 'out_port'},
        ((4+1j), 'past_in'): {'phase': 1, 'type': 'in_port'},
        ((3+0), 'past_in'): {'phase': 1, 'type': 'in_port'},
        (1, 'future_out'): {'phase': 1, 'type': 'out_port'},
        ((3+0), 'future_out'): {'phase': 1, 'type': 'out_port'},
        ((2+0), 'future_out'): {'phase': 1, 'type': 'out_port'},
        ((2+0), 'past_in'): {'phase': 1, 'type': 'in_port'},
        (1, 'past_in'): {'phase': 1, 'type': 'in_port'},
    }
    assert set(g.edges) == {
        (3, (3, 'future_out')),
        (2, (2, 'future_out')),
        (1+1j, 2+1j),
        (1j, 1+1j),
        (4+1j, (4+1j, 'future_out')),
        (1, 1+1j),
        (1, (1, 'past_in')),
        (3, 3+1j),
        (2, (2, 'past_in')),
        (3+1j, 4+1j),
        (3, (3, 'past_in')),
        (1, (1, 'future_out')),
        (2+1j, 3+1j),
        (4+1j, (4+1j, 'past_in')),
    }


def test_to_stim_circuit():
    assert LatticeSurgeryLayer.from_text(r"""
              /    /    /
        .    Z    Z    Z    .
            /|   /    /|
             |         |     /
        T-*--X----X----X----Z
                           /
    """).to_stim_circuit() == stim.Circuit("""
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 0) 3
        QUBIT_COORDS(4, 0) 4
        QUBIT_COORDS(0, 1) 5
        QUBIT_COORDS(1, 1) 6
        QUBIT_COORDS(2, 1) 7
        QUBIT_COORDS(3, 1) 8
        QUBIT_COORDS(4, 1) 9
        QUBIT_COORDS(733, 57473, 11) 10
        RX 5
        CX 5 10
        MR 10
        Z_ERROR(0) 5
        MPAD 0
        R 6
        CX 5 6
        MRX 5
        R 7
        MXX 6 7
        R 8
        MXX 7 8
        MR 7
        CX 1 6
        MR 6
        CX 3 8 9 8
        MR 8
        CZ rec[-7] 1 rec[-7] 3 rec[-7] 9 rec[-6] 1 rec[-6] 3 rec[-6] 9 rec[-5] 3 rec[-5] 9 rec[-4] 3 rec[-4] 9
        OBSERVABLE_INCLUDE(0) rec[-8] rec[-3] rec[-2] rec[-1]
    """)


def test_zx_multi_target_t():
    nx_graph = LatticeSurgeryLayer.from_text(r"""
              /    /    /
        .    Z    Z    Z    .
            /|   /    /|
             |         |     /
        T-*--X----X----X----Z
                           /
    """).to_nx_zx_graph()
    zx_graph = ZXGraph.from_nx_graph(nx_graph)

    assert zx_graph.internal_edge_set == {
        ((3+1j), (4+1j)),
        ((2+1j), (1+1j)),
        ((3+0), (3+1j)),
        (1, (1+1j)),
        ((3+1j), (3+0)),
        ((2+1j), (3+1j)),
        ((4+1j), (3+1j)),
        ((3+1j), (2+1j)),
        ((1+1j), (2+1j)),
        ((1+1j), 1),
    }
    assert zx_graph.input_edge_set == {
        (1, (1, 'past_in')),
        ((1, 'past_in'), 1),
        ((2+0), ((2+0), 'past_in')),
        (((2+0), 'past_in'), (2+0)),
        ((3+0), ((3+0), 'past_in')),
        (((3+0), 'past_in'), (3+0)),
        (1j, (1+1j)),
        ((1+1j), 1j),
        ((4+1j), ((4+1j), 'past_in')),
        (((4+1j), 'past_in'), (4+1j)),
    }
    assert zx_graph.output_edge_set == {
        ((4+1j), ((4+1j), 'future_out')),
        ((3+0), ((3+0), 'future_out')),
        (((4+1j), 'future_out'), (4+1j)),
        (((3+0), 'future_out'), (3+0)),
        ((2+0), ((2+0), 'future_out')),
        (((2+0), 'future_out'), (2+0)),
        (1, (1, 'future_out')),
        ((1, 'future_out'), 1),
    }
    assert zx_graph.to_stabilizer_flow_table(
        include_edges_not_centers=False) == [
        stim.PauliString("+X__________________X________________________"),
        stim.PauliString("+Z__________________Z________________________"),
        stim.PauliString("+_X_____________________X____________________"),
        stim.PauliString("+_Z_____________________Z____________________"),
        stim.PauliString("+__X__________________________X______________"),
        stim.PauliString("+__Z__________________________Z______________"),
        stim.PauliString("+___X_____________________________________X__"),
        stim.PauliString("+___Z_____________________________________Z__"),
        stim.PauliString("+____X___________________________X___________"),
        stim.PauliString("+____Z___________________________Z___________"),
        stim.PauliString("+_____X_______________X______________________"),
        stim.PauliString("+_____Z_______________Z______________________"),
        stim.PauliString("+______X__________________X__________________"),
        stim.PauliString("+______Z__________________Z__________________"),
        stim.PauliString("+_______X_______________________X____________"),
        stim.PauliString("+_______Z_______________________Z____________"),
        stim.PauliString("+________X__________________________________X"),
        stim.PauliString("+________Z__________________________________Z"),
        stim.PauliString("+________________Z_Z_________________________"),
        stim.PauliString("+__________________Z_Z_______________________"),
        stim.PauliString("+_________X______X_X_X_______________________"),
        stim.PauliString("+______________________Z_Z___________________"),
        stim.PauliString("+__________X___________X_X___________________"),
        stim.PauliString("+__________________________Z_Z_______________"),
        stim.PauliString("+____________________________Z_Z_____________"),
        stim.PauliString("+___________X______________X_X_X_____________"),
        stim.PauliString("+_________________X_______________X__________"),
        stim.PauliString("+_________________________________XX_________"),
        stim.PauliString("+____________Z____Z_______________ZZ_________"),
        stim.PauliString("+___________________________________XX_______"),
        stim.PauliString("+_____________Z_____________________ZZ_______"),
        stim.PauliString("+___________________________X_________X______"),
        stim.PauliString("+_____________________________________XX_____"),
        stim.PauliString("+______________Z____________Z_________ZZ_____"),
        stim.PauliString("+_______________________________________ZZ___"),
        stim.PauliString("+________________________________________Z_Z_"),
        stim.PauliString("+_______________X_______________________XX_X_"),
    ]
    assert zx_graph.to_stabilizer_flow_table_with_edge_differences_eliminated() == [
        stim.PauliString("+___XX___X______X________________XXXXXXXXXXXX"),
        stim.PauliString("+X__X_X__XX_____XXXXXXX____________XXXXXXXXXX"),
        stim.PauliString("+Z____Z____________ZZZZ______________________"),
        stim.PauliString("+_X____X___X___________XXXX__________________"),
        stim.PauliString("+_Z____Z_______________ZZZZ__________________"),
        stim.PauliString("+__XX___XX__X___X__________XXXXXX______XXXXXX"),
        stim.PauliString("+__Z____Z____________________ZZZZ____________"),
        stim.PauliString("+___Z____Z_______________________________ZZZZ"),
        stim.PauliString("+____ZZ_ZZ___ZZZ_ZZ__ZZ____ZZ__ZZZZZZZZZZ__ZZ"),
    ]


def test_zx_graph_identity():
    nx_graph = LatticeSurgeryLayer.from_text(r"""
              /
             Z
            /
    """).to_nx_zx_graph()
    zx_graph = ZXGraph.from_nx_graph(nx_graph)

    assert zx_graph.nn2ii == {
        (0j, (0j, 'future_out')): (3, 4),
        ((0j, 'future_out'), 0j): (4, 3),
        (0j, (0j, 'past_in')): (5, 6),
        ((0j, 'past_in'), 0j): (6, 5),
    }
    assert zx_graph.i2n == {
        0: ZXNode(key=(0, 'future_out'), index=0, kind='out', phase=1),
        1: ZXNode(key=(0, 'past_in'), index=1, kind='in', phase=1),
        2: ZXNode(key=0, index=2, kind='Z', phase=1),
    }
    assert zx_graph.internal_edge_set == frozenset()
    assert zx_graph.input_edge_set == {(0, (0, 'past_in')), ((0, 'past_in'), 0)}
    assert zx_graph.output_edge_set == {(0, (0, 'future_out')), ((0, 'future_out'), 0)}
    assert zx_graph.n2neighbors == {
        0j: [ZXNode(key=(0j, 'future_out'), index=0, kind='out', phase=1),
             ZXNode(key=(0j, 'past_in'), index=1, kind='in', phase=1)],
        (0j, 'future_out'): [ZXNode(key=0j, index=2, kind='Z', phase=1)],
        (0j, 'past_in'): [ZXNode(key=0j, index=2, kind='Z', phase=1)],
    }
    assert zx_graph.to_stabilizer_flow_table(include_edges_not_centers=True) == [
        stim.PauliString("+X___X__"),
        stim.PauliString("+Z___Z__"),
        stim.PauliString("+_X____X"),
        stim.PauliString("+_Z____Z"),
        stim.PauliString("+___Z_Z_"),
        stim.PauliString("+___X_X_"),
        stim.PauliString("+__ZZ___"),
        stim.PauliString("+_____XX"),
        stim.PauliString("+_____ZZ"),
        stim.PauliString("+___XX__"),
        stim.PauliString("+___ZZ__"),
    ]
    assert zx_graph.to_stabilizer_flow_table_with_edge_differences_eliminated() == [
        stim.PauliString("+XXXXXXX"),
        stim.PauliString("+ZZ_ZZZZ"),
    ]
    assert zx_graph.to_lattice_surgery_error_table() == [
        stim.PauliString("+X______"),
        stim.PauliString("+Z______"),
        stim.PauliString("+_X_____"),
        stim.PauliString("+_Z_____"),
        stim.PauliString("+___X_X_"),
        stim.PauliString("+__Z____"),
        stim.PauliString("+___X___"),
        stim.PauliString("+___Z___"),
        stim.PauliString("+____X__"),
        stim.PauliString("+____Z__"),
        stim.PauliString("+_____X_"),
        stim.PauliString("+_____Z_"),
        stim.PauliString("+______X"),
        stim.PauliString("+______Z"),
    ]


def test_zx_graph_junction():
    nx_graph = LatticeSurgeryLayer.from_text(r"""
              /
             Z---Z
            /
    """).to_nx_zx_graph()
    zx_graph = ZXGraph.from_nx_graph(nx_graph)

    assert zx_graph.i2n == {
        0: ZXNode(key=(0, 'future_out'), index=0, kind='out', phase=1),
        1: ZXNode(key=(0, 'past_in'), index=1, kind='in', phase=1),
        2: ZXNode(key=0, index=2, kind='Z', phase=1),
        3: ZXNode(key=1, index=3, kind='Z', phase=1),
    }
    assert zx_graph.to_stabilizer_flow_table(
        include_edges_not_centers=False) == [
        stim.PauliString("+X______X__"),
        stim.PauliString("+Z______Z__"),
        stim.PauliString("+_X_______X"),
        stim.PauliString("+_Z_______Z"),
        stim.PauliString("+____Z_Z___"),
        stim.PauliString("+______Z_Z_"),
        stim.PauliString("+__X_X_X_X_"),
        stim.PauliString("+___X_X____"),
    ]
    assert zx_graph.to_stabilizer_flow_table_with_edge_differences_eliminated() == [
        stim.PauliString("+XXXXXXXXXX"),
        stim.PauliString("+ZZ____ZZZZ"),
    ]
    assert zx_graph.to_lattice_surgery_error_table() == [
        stim.PauliString("+X_________"),
        stim.PauliString("+Z_________"),
        stim.PauliString("+_X________"),
        stim.PauliString("+_Z________"),
        stim.PauliString("+____X_X___"),
        stim.PauliString("+____X___X_"),
        stim.PauliString("+______X_X_"),
        stim.PauliString("+__Z_______"),
        stim.PauliString("+___Z______"),
        stim.PauliString("+____X_____"),
        stim.PauliString("+____Z_____"),
        stim.PauliString("+_____X____"),
        stim.PauliString("+_____Z____"),
        stim.PauliString("+______X___"),
        stim.PauliString("+______Z___"),
        stim.PauliString("+_______X__"),
        stim.PauliString("+_______Z__"),
        stim.PauliString("+________X_"),
        stim.PauliString("+________Z_"),
        stim.PauliString("+_________X"),
        stim.PauliString("+_________Z"),
    ]
    assert zx_graph.external_stabilizers == (ExternalStabilizer(paulis=stim.PauliString("+XXXXXXXXXX"),
                                                                sign=1,
                                                                inp=gen.PauliMap({0j: 'X'}),
                                                                out=gen.PauliMap({0j: 'X'}),
                                                                interior=gen.PauliMap({0j: 'X', (1 + 0j): 'X', (0.5 + 0j): 'X'}),
                                                                is_measurement=False,
                                                                sink_edge_cols=(),
                                                                sink_edge_keys=(),
                                                                sink_edge_bases=(),
                                                                port_stabilizer=stim.PauliString("+XX")),
                                             ExternalStabilizer(paulis=stim.PauliString("+ZZ____ZZZZ"),
                                                                sign=1,
                                                                inp=gen.PauliMap({0j: 'Z'}),
                                                                out=gen.PauliMap({0j: 'Z'}),
                                                                interior=gen.PauliMap({0j: 'Z'}),
                                                                is_measurement=False,
                                                                sink_edge_cols=(),
                                                                sink_edge_keys=(),
                                                                sink_edge_bases=(),
                                                                port_stabilizer=stim.PauliString("+ZZ")))


def test_zx_graph_measurement():
    layer = LatticeSurgeryLayer.from_text(r"""
              /   /
             Z-*-Z
            /   /
    """)
    nx_graph = layer.to_nx_zx_graph()
    zx_graph = ZXGraph.from_nx_graph(nx_graph)

    assert zx_graph.i2n == {
        0: ZXNode(key=(0, 'future_out'), index=0, kind='out', phase=1),
        1: ZXNode(key=(1, 'future_out'), index=1, kind='out', phase=1),
        2: ZXNode(key=(0, 'past_in'), index=2, kind='in', phase=1),
        3: ZXNode(key=(1, 'past_in'), index=3, kind='in', phase=1),
        4: ZXNode(key=0, index=4, kind='Z', phase=1),
        5: ZXNode(key=1, index=5, kind='Z', phase=1),
    }
    assert zx_graph.to_stabilizer_flow_table(
        include_edges_not_centers=False) == [
        stim.PauliString("+X________X______"),
        stim.PauliString("+Z________Z______"),
        stim.PauliString("+_X___________X__"),
        stim.PauliString("+_Z___________Z__"),
        stim.PauliString("+__X________X____"),
        stim.PauliString("+__Z________Z____"),
        stim.PauliString("+___X___________X"),
        stim.PauliString("+___Z___________Z"),
        stim.PauliString("+______Z_Z_______"),
        stim.PauliString("+________Z_Z_____"),
        stim.PauliString("+____X_X_X_X_____"),
        stim.PauliString("+_______Z____Z___"),
        stim.PauliString("+____________Z_Z_"),
        stim.PauliString("+_____X_X____X_X_"),
    ]
    assert zx_graph.to_stabilizer_flow_table_with_edge_differences_eliminated() == [
        stim.PauliString("+XXXXXXXXXXXXXXXX"),
        stim.PauliString("+Z__Z__ZZZZ____ZZ"),
        stim.PauliString("+_Z_Z________ZZZZ"),
        stim.PauliString("+__ZZ__ZZ__ZZ__ZZ"),
    ]
    assert zx_graph.measurement_stabilizers == (
        ExternalStabilizer(
            inp=gen.PauliMap({0j: 'Z', (1 + 0j): 'Z'}),
            interior=gen.PauliMap({0j: 'Z', (1 + 0j): 'Z', (0.5 + 0j): 'Z'}),
            is_measurement=True,
            out=gen.PauliMap({}),
            paulis=stim.PauliString("+__ZZ__ZZ__ZZ__ZZ"),
            port_stabilizer=stim.PauliString("+__ZZ"),
            sign=1,
            sink_edge_bases=('X',),
            sink_edge_cols=(6,),
            sink_edge_keys=((0.5+0j),),
        ),
    )
    assert zx_graph.external_stabilizers == (ExternalStabilizer(paulis=stim.PauliString("+XXXXXXXXXXXXXXXX"),
                                                                sign=1,
                                                                inp=gen.PauliMap({0j: 'X', (1 + 0j): 'X'}),
                                                                out=gen.PauliMap({0j: 'X', (1 + 0j): 'X'}),
                                                                interior=gen.PauliMap({0j: 'X', (1 + 0j): 'X', (0.5 + 0j): 'X'}),
                                                                is_measurement=False,
                                                                sink_edge_cols=(),
                                                                sink_edge_keys=(),
                                                                sink_edge_bases=(),
                                                                port_stabilizer=stim.PauliString("+XXXX")),
                                             ExternalStabilizer(paulis=stim.PauliString("+_Z_Z________ZZZZ"),
                                                                sign=1,
                                                                inp=gen.PauliMap({(1 + 0j): 'Z'}),
                                                                out=gen.PauliMap({(1 + 0j): 'Z'}),
                                                                interior=gen.PauliMap({(1 + 0j): 'Z'}),
                                                                is_measurement=False,
                                                                sink_edge_cols=(),
                                                                sink_edge_keys=(),
                                                                sink_edge_bases=(),
                                                                port_stabilizer=stim.PauliString("+_Z_Z")),
                                             ExternalStabilizer(paulis=stim.PauliString("+Z__Z__ZZZZ____ZZ"),
                                                                sign=1,
                                                                inp=gen.PauliMap({(1 + 0j): 'Z'}),
                                                                out=gen.PauliMap({0j: 'Z'}),
                                                                interior=gen.PauliMap({0j: 'Z', (1 + 0j): 'Z', (0.5 + 0j): 'Z'}),
                                                                is_measurement=False,
                                                                sink_edge_cols=(6,),
                                                                sink_edge_keys=(0.5+0j,),
                                                                sink_edge_bases=('X',),
                                                                port_stabilizer=stim.PauliString("+Z__Z")),
                                             ExternalStabilizer(paulis=stim.PauliString("+__ZZ__ZZ__ZZ__ZZ"),
                                                                sign=1,
                                                                inp=gen.PauliMap({0j: 'Z', (1 + 0j): 'Z'}),
                                                                out=gen.PauliMap({}),
                                                                interior=gen.PauliMap({0j: 'Z', (1 + 0j): 'Z', (0.5 + 0j): 'Z'}),
                                                                is_measurement=True,
                                                                sink_edge_cols=(6,),
                                                                sink_edge_keys=(0.5+0j,),
                                                                sink_edge_bases=('X',),
                                                                port_stabilizer=stim.PauliString("+__ZZ")))

    x0 = FeedbackTargets(xs=frozenset([(0, 'future_out')]))
    x1 = FeedbackTargets(xs=frozenset([(1, 'future_out')]))
    z1 = FeedbackTargets(zs=frozenset([(1, 'future_out')]))
    m = FeedbackTargets(ms=frozenset([0]))
    x0_m = FeedbackTargets(xs=frozenset([(0, 'future_out')]), ms=frozenset([0]))
    x1_m = FeedbackTargets(xs=frozenset([(1, 'future_out')]), ms=frozenset([0]))
    p0 = (0, 'past_in')
    p1 = (1, 'past_in')
    f0 = (0, 'future_out')
    f1 = (1, 'future_out')
    expected = {
        (0, 'Z'): z1,
        (p0, 'Z'): z1,
        (p1, 'Z'): z1,
        (1, 'Z'): z1,
        (f1, 'Z'): z1,
        ((1, f1), 'Z'): z1,
        ((f1, 1), 'Z'): z1,
        ((0, f0), 'Z'): z1,
        ((f0, 0), 'Z'): z1,
        ((p0, 0), 'Z'): z1,
        (f0, 'Z'): z1,
        ((0, p0), 'Z'): z1,
        ((1, 0), 'Z'): z1,
        ((0, 1), 'Z'): z1,
        ((1, p1), 'Z'): z1,
        ((p1, 1), 'Z'): z1,

        (p0, 'X'): x0_m,
        ((p0, 0), 'X'): x0_m,
        ((0, p0), 'X'): x0_m,

        ((1, 0), 'X'): m,
        ((0, 1), 'X'): m,

        (p1, 'X'): x1_m,
        ((1, p1), 'X'): x1_m,
        ((p1, 1), 'X'): x1_m,

        (f0, 'X'): x0,
        ((0, f0), 'X'): x0,
        ((f0, 0), 'X'): x0,

        (f1, 'X'): x1,
        ((f1, 1), 'X'): x1,
        ((1, f1), 'X'): x1,
    }
    assert zx_graph.error_to_feedback_map == expected

    assert layer.to_sim_instructions(layer_key='A') == (
        LatticeSurgeryInstruction(
            action='mzz',
            target=0,
            target2=1,
            measure_key=('MZZ', 0.5, 'A'),
        ),
        LatticeSurgeryInstruction('accumulator_bit_clear'),
        LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=('MZZ', 0.5, 'A')),
        LatticeSurgeryInstruction('accumulator_bit_save', measure_key=('logical', 0, 'A')),
    )


def test_zx_graph_merge():
    with pytest.raises(ValueError):
        LatticeSurgeryLayer.from_text(r"""
                  /
             Z---Z---Z
            /       /
        """).to_sim_instructions(layer_key='A')
    with pytest.raises(ValueError):
        LatticeSurgeryLayer.from_text(r"""
                  /
             Z-@-Z---Z
            /       /
        """).to_sim_instructions(layer_key='A')
    with pytest.raises(ValueError):
        LatticeSurgeryLayer.from_text(r"""
                  /
             Z-@-Z-*-Z
            /       /
        """).to_sim_instructions(layer_key='A')
    with pytest.raises(ValueError):
        LatticeSurgeryLayer.from_text(r"""
                   /
             Z-@*-Z---Z
            /        /
        """).to_sim_instructions(layer_key='A')
    with pytest.raises(ValueError):
        LatticeSurgeryLayer.from_text(r"""
                  /
             Z-*-Z-*-Z
            /       /
        """).to_sim_instructions(layer_key='A')

    assert LatticeSurgeryLayer.from_text(r"""
              /
         Z-*-Z---Z
        /       /
    """).to_sim_instructions(layer_key='A') == (
        LatticeSurgeryInstruction('qalloc_x', target=1),
        LatticeSurgeryInstruction('mzz',
                                  target=0,
                                  target2=1,
                                  measure_key=('MZZ', 0.5, 'A')),
        LatticeSurgeryInstruction('m_discard_x',
                                  target=0,
                                  measure_key=('MX', 0, 'A')),
        LatticeSurgeryInstruction('mzz',
                                  target=1,
                                  target2=2,
                                  measure_key=('MZZ', 1.5, 'A')),
        LatticeSurgeryInstruction('m_discard_x',
                                  target=2,
                                  measure_key=('MX', 2, 'A')),
        LatticeSurgeryInstruction('feedback_m2z',
                                  target=1,
                                  measure_key=('MX', 0, 'A')),
        LatticeSurgeryInstruction('feedback_m2x',
                                  target=1,
                                  measure_key=('MZZ', 1.5, 'A')),
        LatticeSurgeryInstruction('feedback_m2z',
                                  target=1,
                                  measure_key=('MX', 2, 'A')),
        LatticeSurgeryInstruction('accumulator_bit_clear', target=None, target2=None, measure_key=None),
        LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=('MZZ', 0.5, 'A')),
        LatticeSurgeryInstruction('accumulator_bit_xor', measure_key=('MZZ', 1.5, 'A')),
        LatticeSurgeryInstruction('accumulator_bit_save', measure_key=('logical', 0, 'A')),
    )

    assert lines(LatticeSurgeryLayer.from_text(r"""
              /
         Z---Z-*-Z
        /       /
    """).to_sim_instructions(layer_key='A')) == """
        qalloc_x 1
        mzz 0 1 ('MZZ', (0.5+0j), 'A')
        m_discard_x 0 ('MX', 0j, 'A')
        mzz 1 2 ('MZZ', (1.5+0j), 'A')
        m_discard_x 2 ('MX', (2+0j), 'A')
        feedback_m2x 1 ('MZZ', (0.5+0j), 'A')
        feedback_m2z 1 ('MX', 0j, 'A')
        feedback_m2z 1 ('MX', (2+0j), 'A')
        accumulator_bit_clear
        accumulator_bit_xor ('MZZ', (0.5+0j), 'A')
        accumulator_bit_xor ('MZZ', (1.5+0j), 'A')
        accumulator_bit_save ('logical', 0, 'A')
    """


def test_error_mechanisms_during_merge():
    assert LatticeSurgeryLayer.from_text(r"""
                /
          Z-*--Z
         /    /
    """).to_sim_instructions(layer_key='A', include_error_mechanisms=True) == (
        LatticeSurgeryInstruction(action='error_mechanism_x',
                                  target=0j,
                                  target2=None,
                                  measure_key=None,
                                  error_source=ErrorSource(error_type='timelike_edge_error',
                                           error_basis='X',
                                           error_location=0j,
                                           error_initiative='before',
                                           error_layer='A')),
        LatticeSurgeryInstruction(action='error_mechanism_z',
                                  target=0j,
                                  target2=None,
                                  measure_key=None,
                                  error_source=ErrorSource(error_type='timelike_edge_error',
                                           error_basis='Z',
                                           error_location=0j,
                                           error_initiative='before',
                                           error_layer='A')),
        LatticeSurgeryInstruction(action='error_mechanism_x',
                                  target=(1+0j),
                                  target2=None,
                                  measure_key=None,
                                  error_source=ErrorSource(error_type='timelike_edge_error',
                                           error_basis='X',
                                           error_location=(1+0j),
                                           error_initiative='before',
                                           error_layer='A')),
        LatticeSurgeryInstruction(action='error_mechanism_z',
                                  target=(1+0j),
                                  target2=None,
                                  measure_key=None,
                                  error_source=ErrorSource(error_type='timelike_edge_error',
                                           error_basis='Z',
                                           error_location=(1+0j),
                                           error_initiative='before',
                                           error_layer='A')),
        LatticeSurgeryInstruction(action='error_mechanism_m',
                                  target=None,
                                  target2=None,
                                  measure_key=('MZZ', (0.5+0j), 'A'),
                                  error_source=ErrorSource(error_type='spacelike_edge_error',
                                           error_basis='X',
                                           error_location=(0.5+0j),
                                           error_initiative='during',
                                           error_layer='A')),
        LatticeSurgeryInstruction(action='error_mechanism_z',
                                  target=0j,
                                  target2=None,
                                  measure_key=None,
                                  error_source=ErrorSource(error_type='spacelike_edge_error',
                                           error_basis='Z',
                                           error_location=(0.5+0j),
                                           error_initiative='during',
                                           error_layer='A')),
        LatticeSurgeryInstruction(action='mzz',
                                  target=0j,
                                  target2=(1+0j),
                                  measure_key=('MZZ', (0.5+0j), 'A'),
                                  error_source=None),
        LatticeSurgeryInstruction(action='m_discard_x',
                                  target=0j,
                                  target2=None,
                                  measure_key=('MX', 0j, 'A'),
                                  error_source=None),
        LatticeSurgeryInstruction(action='error_mechanism_x',
                                  target=(1+0j),
                                  target2=None,
                                  measure_key=None,
                                  error_source=ErrorSource(error_type='timelike_edge_error',
                                           error_basis='X',
                                           error_location=(1+0j),
                                           error_initiative='after',
                                           error_layer='A')),
        LatticeSurgeryInstruction(action='error_mechanism_z',
                                  target=(1+0j),
                                  target2=None,
                                  measure_key=None,
                                  error_source=ErrorSource(error_type='timelike_edge_error',
                                           error_basis='Z',
                                           error_location=(1+0j),
                                           error_initiative='after',
                                           error_layer='A')),
        LatticeSurgeryInstruction(action='feedback_m2z',
                                  target=(1+0j),
                                  target2=None,
                                  measure_key=('MX', 0j, 'A'),
                                  error_source=None),
        LatticeSurgeryInstruction(action='accumulator_bit_clear',
                                  target=None,
                                  target2=None,
                                  measure_key=None,
                                  error_source=None),
        LatticeSurgeryInstruction(action='accumulator_bit_xor',
                                  target=None,
                                  target2=None,
                                  measure_key=('MZZ', (0.5+0j), 'A'),
                                  error_source=None),
        LatticeSurgeryInstruction(action='accumulator_bit_save',
                                  target=None,
                                  target2=None,
                                  measure_key=('logical', 0, 'A'),
                                  error_source=None),
    )


@pytest.mark.parametrize('node0_basis,node1_basis,data_basis', itertools.product('XZ', repeat=3))
def test_error_mechanisms_during_move_do_right_thing(node0_basis: str, node1_basis: str, data_basis: str):
    diagram = r"""
                /
          A----B
         /    
    """.replace('A', node0_basis).replace('B', node1_basis)
    tasks = LatticeSurgeryLayer.from_text(diagram).to_sim_instructions(layer_key='V', include_error_mechanisms=True)
    mechanisms = []
    for task in tasks:
        if 'error_mechanism' in task.action:
            mechanisms.append(task.error_source)
    for k in range(-1, len(mechanisms)):
        sim = VecSim()
        if data_basis == 'X':
            sim.do_qalloc_x(0)
        else:
            sim.do_qalloc_z(0)
        sim.included_error_mechanisms.add(k)
        sim.do_instructions(tasks)
        if data_basis == 'X':
            flipped = sim.do_mx(1)
        else:
            flipped = sim.do_mz(1)
        should_be_flipped = k != -1 and mechanisms[k].error_basis != data_basis
        assert flipped == should_be_flipped


def lines(x: Iterable[Any]) -> str:
    return '\n' + '\n'.join('        ' + str(e) for e in x) + '\n    '


def test_injected_errors():
    layer = LatticeSurgeryLayer.from_text(r"""
               /   /
          X---Z---Z
         /    
    """)

    assert lines(layer.to_sim_instructions(layer_key='A', injected_errors=frozenset([
        InjectedError(pos=0.5, layer=0, basis='X'),
    ]))) == """
        qalloc_x 1
        cx 1 0
        x 0
        m_discard_z 0 ('MZ', 0j, 'A')
        qalloc_x 2
        mzz 1 2 ('MZZ', (1.5+0j), 'A')
        feedback_m2x 1 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZZ', (1.5+0j), 'A')
    """
    assert lines(layer.to_sim_instructions(layer_key='A', injected_errors=frozenset([
        InjectedError(pos=0.5, layer=0, basis='Y'),
    ]))) == """
        qalloc_x 1
        cx 1 0
        x 0
        z 1
        m_discard_z 0 ('MZ', 0j, 'A')
        qalloc_x 2
        mzz 1 2 ('MZZ', (1.5+0j), 'A')
        feedback_m2x 1 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZZ', (1.5+0j), 'A')
    """
    assert lines(layer.to_sim_instructions(layer_key='A', injected_errors=frozenset([
        InjectedError(pos=0.5, layer=0, basis='Z'),
    ]))) == """
        qalloc_x 1
        cx 1 0
        z 1
        m_discard_z 0 ('MZ', 0j, 'A')
        qalloc_x 2
        mzz 1 2 ('MZZ', (1.5+0j), 'A')
        feedback_m2x 1 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZZ', (1.5+0j), 'A')
    """

    assert lines(layer.to_sim_instructions(layer_key='A', injected_errors=frozenset([
        InjectedError(pos=1.5, layer=0, basis='X'),
    ]))) == """
        qalloc_x 1
        cx 1 0
        m_discard_z 0 ('MZ', 0j, 'A')
        qalloc_x 2
        x 1
        mzz 1 2 ('MZZ', (1.5+0j), 'A')
        x 1
        feedback_m2x 1 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZZ', (1.5+0j), 'A')
    """
    assert lines(layer.to_sim_instructions(layer_key='A', injected_errors=frozenset([
        InjectedError(pos=1.5, layer=0, basis='Y'),
    ]))) == """
        qalloc_x 1
        cx 1 0
        m_discard_z 0 ('MZ', 0j, 'A')
        qalloc_x 2
        y 1
        mzz 1 2 ('MZZ', (1.5+0j), 'A')
        x 1
        feedback_m2x 1 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZZ', (1.5+0j), 'A')
    """
    assert lines(layer.to_sim_instructions(layer_key='A', injected_errors=frozenset([
        InjectedError(pos=1.5, layer=0, basis='Z'),
    ]))) == """
        qalloc_x 1
        cx 1 0
        m_discard_z 0 ('MZ', 0j, 'A')
        qalloc_x 2
        z 1
        mzz 1 2 ('MZZ', (1.5+0j), 'A')
        feedback_m2x 1 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZZ', (1.5+0j), 'A')
    """

    assert lines(layer.to_sim_instructions(layer_key='A', injected_errors=frozenset([
        InjectedError(pos=0, layer=-0.5, basis='Y'),
    ]))) == """
        y 0
        qalloc_x 1
        cx 1 0
        m_discard_z 0 ('MZ', 0j, 'A')
        qalloc_x 2
        mzz 1 2 ('MZZ', (1.5+0j), 'A')
        feedback_m2x 1 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZZ', (1.5+0j), 'A')
    """

    assert lines(layer.to_sim_instructions(layer_key='A', injected_errors=frozenset([
        InjectedError(pos=0.5, layer=0, basis='Y'),
        InjectedError(pos=1.5, layer=0, basis='Y'),
        InjectedError(pos=0, layer=-0.5, basis='Y'),
    ]))) == """
        y 0
        qalloc_x 1
        cx 1 0
        x 0
        z 1
        m_discard_z 0 ('MZ', 0j, 'A')
        qalloc_x 2
        y 1
        mzz 1 2 ('MZZ', (1.5+0j), 'A')
        x 1
        feedback_m2x 1 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZ', 0j, 'A')
        feedback_m2x 2 ('MZZ', (1.5+0j), 'A')
    """
