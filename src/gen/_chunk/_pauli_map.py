from typing import (
    Callable,
    Literal,
    TYPE_CHECKING,
    cast,
    Iterable,
    Any,
    Union,
    AbstractSet,
    Iterator,
)

import stim

from gen._chunk._complex_util import sorted_complex

if TYPE_CHECKING:
    from gen._chunk._keyed_pauli_map import KeyedPauliMap
    from gen._chunk._tile import Tile


_multiplication_table: dict[
    Literal["X", "Y", "Z"] | None,
    dict[Literal["X", "Y", "Z"] | None, Literal["X", "Y", "Z"] | None],
] = {
    None: {None: None, "X": "X", "Y": "Y", "Z": "Z"},
    "X": {None: "X", "X": None, "Y": "Z", "Z": "Y"},
    "Y": {None: "Y", "X": "Z", "Y": None, "Z": "X"},
    "Z": {None: "Z", "X": "Y", "Y": "X", "Z": None},
}


class PauliMap:
    """A qubit-to-pauli mapping."""

    def __init__(
        self,
        mapping: Union[
            dict[complex, Literal["X", "Y", "Z"]],
            dict[Literal["X", "Y", "Z"], complex | Iterable[complex]],
            "PauliMap",
            "KeyedPauliMap",
            stim.PauliString,
            None,
        ] = None,
        *,
        xs: Iterable[complex] = (),
        ys: Iterable[complex] = (),
        zs: Iterable[complex] = (),
    ):
        """Initializes a PauliMap using maps of Paulis to/from qubits."""

        self.qubits: dict[complex, Literal["X", "Y", "Z"]] = {}

        from gen._chunk._keyed_pauli_map import KeyedPauliMap

        if (
            isinstance(mapping, (PauliMap, KeyedPauliMap))
            and not xs
            and not ys
            and not zs
        ):
            self.qubits = dict(mapping.qubits)
            self._hash = (
                getattr(mapping, "_hash")
                if isinstance(mapping, PauliMap)
                else getattr(mapping.pauli_string, "_hash")
            )
            return

        for q in xs:
            self._mul_term(q, "X")
        for q in ys:
            self._mul_term(q, "Y")
        for q in zs:
            self._mul_term(q, "Z")
        if isinstance(mapping, stim.PauliString):
            for q in mapping.pauli_indices():
                self._mul_term(q, cast(Any, "_XYZ"[mapping[q]]))
            mapping = None
        if mapping is not None:
            if isinstance(mapping, (PauliMap, KeyedPauliMap)):
                mapping = mapping.qubits
            for k, v in mapping.items():
                if isinstance(k, str):
                    assert k == "X" or k == "Y" or k == "Z"
                    b = cast(Literal["X", "Y", "Z"], k)
                    if isinstance(v, (int, float, complex)):
                        self._mul_term(v, b)
                    else:
                        for q in v:
                            assert isinstance(q, (int, float, complex))
                            self._mul_term(q, b)
                elif isinstance(v, str):
                    assert v == "X" or v == "Y" or v == "Z"
                    assert isinstance(k, (int, float, complex))
                    b = cast(Literal["X", "Y", "Z"], v)
                    self._mul_term(k, b)

        self.qubits = {
            complex(q): self.qubits[q] for q in sorted_complex(self.qubits.keys())
        }
        self._hash: int = hash(tuple(self.qubits.items()))

    def __contains__(self, item) -> bool:
        return self.qubits.__contains__(item)

    def items(self) -> Iterable[tuple[complex, Literal["X", "Y", "Z"]]]:
        return self.qubits.items()

    def values(self) -> Iterable[tuple[complex, Literal["X", "Y", "Z"]]]:
        return self.qubits.values()

    def keys(self) -> AbstractSet[complex]:
        return self.qubits.keys()

    def __getitem__(self, item) -> Literal["I", "X", "Y", "Z"]:
        return self.qubits.get(item, "I")

    def __len__(self) -> int:
        return len(self.qubits)

    def __iter__(self) -> Iterator[complex]:
        return self.qubits.__iter__()

    @property
    def pauli_string(self) -> "PauliMap":
        """Duck-typing compatibility with KeyedPauliString."""
        return self

    def keyed(self, key: Any) -> "KeyedPauliMap":
        from gen._chunk._keyed_pauli_map import KeyedPauliMap

        return KeyedPauliMap(key=key, pauli_string=self)

    def _mul_term(self, q: complex, b: Literal["X", "Y", "Z"]):
        new_b = _multiplication_table[self.qubits.pop(q, None)][b]
        if new_b is not None:
            self.qubits[q] = new_b

    @staticmethod
    def from_tile_data(tile: "Tile") -> "PauliMap":
        return PauliMap(
            {k: v for k, v in zip(tile.data_qubits, tile.bases) if k is not None}
        )

    def with_basis(self, basis: Literal["X", "Y", "Z"]) -> "PauliMap":
        return PauliMap({q: basis for q in self.qubits.keys()})

    def __bool__(self) -> bool:
        return bool(self.qubits)

    def __mul__(self, other: Union["PauliMap", "KeyedPauliMap", "Tile"]) -> "PauliMap":
        from gen._chunk._tile import Tile

        if isinstance(other, Tile):
            other = other.to_data_pauli_string()

        result: dict[complex, Literal["X", "Y", "Z"]] = {}
        for q in self.qubits.keys() | other.qubits.keys():
            a = self.qubits.get(q, "I")
            b = other.qubits.get(q, "I")
            ax = a in "XY"
            az = a in "YZ"
            bx = b in "XY"
            bz = b in "YZ"
            cx = ax ^ bx
            cz = az ^ bz
            c = "IXZY"[cx + cz * 2]
            if c != "I":
                result[q] = cast(Literal["X", "Y", "Z"], c)
        return PauliMap(result)

    def __repr__(self) -> str:
        s = {q: self.qubits[q] for q in sorted_complex(self.qubits)}
        return f"gen.PauliMap({s!r})"

    def __str__(self) -> str:
        def simplify(c: complex) -> str:
            if c == int(c.real):
                return str(int(c.real))
            if c == c.real:
                return str(c.real)
            return str(c)

        return "*".join(
            f"{self.qubits[q]}{simplify(q)}" for q in sorted_complex(self.qubits.keys())
        )

    def with_xz_flipped(self) -> "PauliMap":
        remap = {"X": "Z", "Y": "Y", "Z": "X"}
        return PauliMap({q: remap[p] for q, p in self.qubits.items()})

    def with_xy_flipped(self) -> "PauliMap":
        remap = {"X": "Y", "Y": "X", "Z": "Z"}
        return PauliMap({q: remap[p] for q, p in self.qubits.items()})

    def commutes(self, other: "PauliMap") -> bool:
        return not self.anticommutes(other)

    def anticommutes(self, other: "PauliMap") -> bool:
        t = 0
        for q in self.qubits.keys() & other.qubits.keys():
            t += self.qubits[q] != other.qubits[q]
        return t % 2 == 1

    def with_transformed_coords(
        self, transform: Callable[[complex], complex]
    ) -> "PauliMap":
        return PauliMap({transform(q): p for q, p in self.qubits.items()})

    def to_tile(self) -> "Tile":
        from gen._chunk._tile import Tile

        qs = list(self.qubits.keys())
        return Tile(
            bases="".join(self.qubits.values()),
            data_qubits=qs,
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other) -> bool:
        if not isinstance(other, PauliMap):
            return NotImplemented
        return self.qubits == other.qubits

    def _sort_key(self) -> Any:
        return tuple((q.real, q.imag, p) for q, p in self.qubits.items())

    def __lt__(self, other) -> bool:
        if not isinstance(other, PauliMap):
            return NotImplemented
        return self._sort_key() < other._sort_key()
