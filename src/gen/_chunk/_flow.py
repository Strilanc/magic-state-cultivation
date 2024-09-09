from typing import Iterable, Any, Callable, Literal, TYPE_CHECKING, Union

from gen._chunk._keyed_pauli_map import KeyedPauliMap
from gen._chunk._pauli_map import PauliMap
from gen._util import xor_sorted

if TYPE_CHECKING:
    import gen


class Flow:
    """A rule for how a stabilizer travels into, through, and/or out of a chunk."""

    def __init__(
        self,
        *,
        start: Union[PauliMap, "gen.Tile", None] = None,
        end: Union[PauliMap, "gen.Tile", None] = None,
        measurement_indices: Iterable[int] | Literal["auto"] = (),
        obs_key: Any = None,
        center: complex | None = None,
        flags: Iterable[str] = frozenset(),
        sign: bool | None = None,
    ):
        if obs_key is None and center is None:
            raise ValueError("Specify obs_key or center.")
        if isinstance(flags, str):
            raise TypeError(f"{flags=} is a str instead of a set")
        if obs_key is None and isinstance(start, KeyedPauliMap):
            obs_key = start.key
        if obs_key is None and isinstance(end, KeyedPauliMap):
            obs_key = end.key
        if isinstance(start, KeyedPauliMap):
            assert obs_key == start.key
            start = start.pauli_string
        if isinstance(end, KeyedPauliMap):
            assert obs_key == end.key
            end = end.pauli_string
        from ._tile import Tile

        if start is not None and not isinstance(start, (PauliMap, Tile)):
            raise ValueError(
                f"{start=} is not None and not isinstance(start, (gen.PauliMap, gen.Tile))"
            )
        if end is not None and not isinstance(end, (PauliMap, Tile)):
            raise ValueError(
                f"{end=} is not None and not isinstance(end, (gen.PauliMap, gen.Tile))"
            )
        if isinstance(start, Tile):
            start = start.to_data_pauli_string()
        elif start is None:
            start = PauliMap()
        if isinstance(end, Tile):
            end = end.to_data_pauli_string()
        elif end is None:
            end = PauliMap()
        self.start: PauliMap = start
        self.end: PauliMap = end
        self.measurement_indices: tuple[int, ...] | Literal["auto"] = (
            measurement_indices
            if measurement_indices == "auto"
            else tuple(xor_sorted(measurement_indices))
        )
        self.flags: frozenset[str] = frozenset(flags)
        self.obs_key: Any = obs_key
        self.center: complex | None = center
        self.sign: bool | None = sign
        if measurement_indices == "auto" and not start and not end:
            raise ValueError("measurement_indices == 'auto' and not start and not end")

    @property
    def key_start(self) -> KeyedPauliMap | PauliMap:
        if self.obs_key is None:
            return self.start
        return self.start.keyed(self.obs_key)

    @property
    def key_end(self) -> KeyedPauliMap | PauliMap:
        if self.obs_key is None:
            return self.end
        return self.end.keyed(self.obs_key)

    def with_edits(
        self,
        *,
        start: PauliMap | None = None,
        end: PauliMap | None = None,
        measurement_indices: Iterable[int] | None = None,
        obs_key: Any = "__not_specified!!",
        center: complex | None = None,
        flags: Iterable[str] | None = None,
        sign: Any = "__not_specified!!",
    ) -> "Flow":
        return Flow(
            start=self.start if start is None else start,
            end=self.end if end is None else end,
            measurement_indices=(
                self.measurement_indices
                if measurement_indices is None
                else measurement_indices
            ),
            obs_key=self.obs_key if obs_key == "__not_specified!!" else obs_key,
            center=self.center if center is None else center,
            flags=self.flags if flags is None else flags,
            sign=self.sign if sign == "__not_specified!!" else sign,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Flow):
            return NotImplemented
        return (
            self.start == other.start
            and self.end == other.end
            and self.measurement_indices == other.measurement_indices
            and self.obs_key == other.obs_key
            and self.flags == other.flags
            and self.center == other.center
            and self.sign == other.sign
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.start,
                self.end,
                self.measurement_indices,
                self.obs_key,
                self.flags,
                self.center,
                self.sign,
            )
        )

    def __str__(self) -> str:
        start_terms = []
        for q, p in self.start.qubits.items():
            start_terms.append(f"{p}[{q}]")
        end_terms = []
        for q, p in self.end.qubits.items():
            q = complex(q)
            if q.real == 0:
                q = "0+" + str(q)
            q = str(q).replace("(", "").replace(")", "")
            end_terms.append(f"{p}[{q}]")
        if self.measurement_indices == "auto":
            end_terms.append("rec[auto]")
        else:
            for m in self.measurement_indices:
                end_terms.append(f"rec[{m}]")
        if not start_terms:
            start_terms.append("1")
        if not end_terms:
            end_terms.append("1")
        key = "" if self.obs_key is None else f" (obs={self.obs_key})"
        result = f'{"*".join(start_terms)} -> {"*".join(end_terms)}{key}'
        if self.sign is None:
            pass
        elif self.sign:
            result = "-" + result
        else:
            result = "+" + result
        return result

    def __repr__(self):
        return (
            f"Flow(start={self.start!r}, "
            f"end={self.end!r}, "
            f"measurement_indices={self.measurement_indices!r}, "
            f"flags={sorted(self.flags)}, "
            f"obs_key={self.obs_key!r}, "
            f"center={self.center!r}, "
            f"sign={self.sign!r}"
        )

    def with_xz_flipped(self) -> "Flow":
        return self.with_edits(
            start=self.start.with_xz_flipped(),
            end=self.end.with_xz_flipped(),
        )

    def with_transformed_coords(
        self, transform: Callable[[complex], complex]
    ) -> "Flow":
        return self.with_edits(
            start=self.start.with_transformed_coords(transform),
            end=self.end.with_transformed_coords(transform),
            center=transform(self.center),
        )

    def fuse_with_next_flow(
        self, next_flow: "Flow", *, next_flow_measure_offset: int
    ) -> "Flow":
        if next_flow.start != self.end:
            raise ValueError("other.start != self.end")
        if next_flow.obs_key != self.obs_key:
            raise ValueError("other.obs_key != self.obs_key")
        if self.center is None:
            new_center = next_flow.center
        elif next_flow.center is None:
            new_center = self.center
        else:
            new_center = (self.center + next_flow.center) / 2
        return Flow(
            start=self.start,
            end=next_flow.end,
            center=new_center,
            measurement_indices=self.measurement_indices
            + tuple(
                m + next_flow_measure_offset for m in next_flow.measurement_indices
            ),
            obs_key=self.obs_key,
            flags=self.flags | next_flow.flags,
            sign=(
                None
                if self.sign is None or next_flow.sign is None
                else self.sign ^ next_flow.sign
            ),
        )

    def __mul__(self, other: "Flow") -> "Flow":
        if other.obs_key != self.obs_key:
            raise ValueError("other.obs_key != self.obs_key")
        return Flow(
            start=self.start * other.start,
            end=self.end * other.end,
            measurement_indices=sorted(
                set(self.measurement_indices) ^ set(other.measurement_indices)
            ),
            obs_key=self.obs_key,
            flags=self.flags | other.flags,
            center=(self.center + other.center) / 2,
            sign=(
                None
                if self.sign is None or other.sign is None
                else self.sign ^ other.sign
            ),
        )
