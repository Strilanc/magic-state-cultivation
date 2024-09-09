import functools
from typing import Iterable, Callable, TYPE_CHECKING, Literal

from gen._chunk._keyed_pauli_map import KeyedPauliMap
from gen._chunk._patch import Patch
from gen._chunk._pauli_map import PauliMap
from gen._chunk._stabilizer_code import StabilizerCode
from gen._chunk._test_util import assert_has_same_set_of_items_as
from gen._chunk._tile import Tile
from gen._chunk._complex_util import sorted_complex

if TYPE_CHECKING:
    from gen._chunk._chunk import Chunk
    from gen._chunk._chunk_interface import ChunkInterface


class ChunkReflow:
    def __init__(
        self,
        out2in: dict[PauliMap | KeyedPauliMap, list[PauliMap | KeyedPauliMap]],
        discard_in: Iterable[PauliMap | KeyedPauliMap] = (),
    ):
        self.out2in = out2in
        self.discard_in = tuple(discard_in)
        assert isinstance(self.out2in, dict)
        for k, vs in self.out2in.items():
            assert isinstance(k, (PauliMap, KeyedPauliMap)), k
            assert isinstance(vs, list)
            for v in vs:
                assert isinstance(v, (PauliMap, KeyedPauliMap))

    @staticmethod
    def from_auto_rewrite(
            *,
            inputs: Iterable[PauliMap | KeyedPauliMap],
            out2in: dict[PauliMap | KeyedPauliMap, list[PauliMap | KeyedPauliMap] | Literal['auto']],
    ):
        new_out2in = {}
        unsolved = []
        for k, v in out2in.items():
            if v == 'auto':
                unsolved.append(k)
            else:
                new_out2in[k] = v
        if not unsolved:
            return ChunkReflow(out2in=new_out2in)

        rows: list[tuple[set[int], PauliMap]] = []
        inputs = list(inputs)
        qs = set()
        for k in range(len(inputs)):
            rows.append(({k}, inputs[k].pauli_string))
            qs |= inputs[k].pauli_string.keys()
        for v in unsolved:
            rows.append((set(), v.pauli_string))
        qs = sorted_complex(qs)
        num_solved = 0
        for q in qs:
            for b in 'ZX':
                for pivot in range(num_solved, len(inputs)):
                    p = rows[pivot][1][q]
                    if p != b and p != 'I':
                        break
                else:
                    continue
                for row in range(len(rows)):
                    p = rows[row][1][q]
                    if row != pivot and p != b and p != 'I':
                        a1, b1 = rows[row]
                        a2, b2 = rows[pivot]
                        rows[row] = (a1 ^ a2, b1 * b2)
                if pivot != num_solved:
                    rows[num_solved], rows[pivot] = rows[pivot], rows[num_solved]
                num_solved += 1
        for k in range(len(unsolved)):
            v = rows[k + len(inputs)]
            if v[1]:
                raise ValueError(f"Failed to solve for {unsolved[k]}.")
            new_out2in[unsolved[k]] = [inputs[v2] for v2 in v[0]]

        return ChunkReflow(out2in=new_out2in)

    def with_obs_flows_as_det_flows(self):
        return ChunkReflow(
            out2in={
                PauliMap(k): [PauliMap(v) for v in vs]
                for k, vs in self.out2in.items()
            },
            discard_in=[PauliMap(k) for k in self.discard_in],
        )

    def with_transformed_coords(
        self, transform: Callable[[complex], complex]
    ) -> "ChunkReflow":
        return ChunkReflow(
            out2in={
                kp.with_transformed_coords(transform): [
                    vp.with_transformed_coords(transform) for vp in vs
                ]
                for kp, vs in self.out2in.items()
            },
            discard_in=[
                kp.with_transformed_coords(transform) for kp in self.discard_in
            ],
        )

    def start_interface(self) -> "ChunkInterface":
        from gen._chunk._chunk_interface import ChunkInterface

        return ChunkInterface(
            ports={v for vs in self.out2in.values() for v in vs},
            discards=self.discard_in,
        )

    def end_interface(self) -> "ChunkInterface":
        from gen._chunk._chunk_interface import ChunkInterface

        return ChunkInterface(
            ports=self.out2in.keys(),
            discards=self.discard_in,
        )

    def start_code(self) -> StabilizerCode:
        tiles = []
        observables = []
        for ps, obs in self.removed_inputs:
            if obs is None:
                tiles.append(
                    Tile(
                        data_qubits=ps.qubits.keys(),
                        bases="".join(ps.qubits.values()),
                        measure_qubit=list(ps.qubits.keys())[0],
                    )
                )
            else:
                observables.append(ps)
        return StabilizerCode(stabilizers=Patch(tiles), logicals=observables)

    def start_patch(self) -> Patch:
        tiles = []
        for ps, obs in self.removed_inputs:
            if obs is None:
                tiles.append(
                    Tile(
                        data_qubits=ps.qubits.keys(),
                        bases="".join(ps.qubits.values()),
                        measure_qubit=list(ps.qubits.keys())[0],
                    )
                )
        return Patch(tiles)

    def end_patch(self) -> Patch:
        tiles = []
        for ps, obs in self.out2in.keys():
            if obs is None:
                tiles.append(
                    Tile(
                        data_qubits=ps.qubits.keys(),
                        bases="".join(ps.qubits.values()),
                        measure_qubit=list(ps.qubits.keys())[0],
                    )
                )
        return Patch(tiles)

    def mpp_init_chunk(self) -> "Chunk":
        return self.start_interface().mpp_init_chunk()

    def mpp_end_chunk(self) -> "Chunk":
        return self.end_interface().mpp_end_chunk()

    @functools.cached_property
    def removed_inputs(self) -> frozenset[PauliMap | KeyedPauliMap]:
        return frozenset(v for vs in self.out2in.values() for v in vs) | frozenset(
            self.discard_in
        )

    def verify(
        self,
        *,
        expected_in: StabilizerCode | None = None,
        expected_out: StabilizerCode | None = None,
    ):
        assert isinstance(self.out2in, dict)
        for k, vs in self.out2in.items():
            assert isinstance(k, (PauliMap, KeyedPauliMap)), k
            assert isinstance(vs, list)
            for v in vs:
                assert isinstance(v, (PauliMap, KeyedPauliMap))

        for k, vs in self.out2in.items():
            acc = PauliMap({})
            for v in vs:
                acc *= PauliMap(v)
            if acc != PauliMap(k):
                lines = [
                    "A reflow output wasn't equal to the product of its inputs.",
                    f"   Output: {k}",
                    f"   Difference: {PauliMap(k) * acc}",
                    f"   Inputs:",
                ]
                for v in vs:
                    lines.append(f"        {v}")
                raise ValueError("\n".join(lines))

        if expected_in is not None:
            if isinstance(expected_in, StabilizerCode):
                expected_in = expected_in.as_interface()
            assert_has_same_set_of_items_as(
                self.start_interface().with_discards_as_ports().ports,
                expected_in.with_discards_as_ports().ports,
                "self.start_interface().with_discards_as_ports().ports",
                "expected_in.with_discards_as_ports().ports",
            )

        if expected_out is not None:
            if isinstance(expected_out, StabilizerCode):
                expected_out = expected_out.as_interface()
            assert_has_same_set_of_items_as(
                self.end_interface().with_discards_as_ports().ports,
                expected_out.with_discards_as_ports().ports,
                "self.end_interface().with_discards_as_ports().ports",
                "expected_out.with_discards_as_ports().ports",
            )

        if len(self.out2in) != len(self.removed_inputs):
            msg = [
                "Number of outputs != number of distinct inputs.",
                "Outputs {",
            ]
            for ps, obs in self.out2in:
                msg.append(f"    {ps}, obs={obs}")
            msg.append("}")
            msg.append("Distinct inputs {")
            for ps, obs in self.removed_inputs:
                msg.append(f"    {ps}, obs={obs}")
            msg.append("}")
            raise ValueError("\n".join(msg))

    def __eq__(self, other) -> bool:
        if isinstance(other, ChunkReflow):
            return self.out2in == other.out2in and self.discard_in == other.discard_in
        return False

    def __ne__(self, other) -> bool:
        return not (self == other)

    def __repr__(self) -> str:
        return f'gen.ChunkReflow(out2in={self.out2in!r}, discard_in={self.discard_in!r})'

    def __str__(self) -> str:
        lines = [
            'Reflow {',
        ]
        for k, v in self.out2in.items():
            if [k] != v:
                lines.append(f'    gather {k} {{')
                for v2 in v:
                    lines.append(f'        {v2}')
                lines.append(f'    }}')
        for k, v in self.out2in.items():
            if [k] == v:
                lines.append(f'    keep {k}')
        for k in self.discard_in:
            lines.append(f'    discard {k}')
        lines.append('}')
        return '\n'.join(lines)
