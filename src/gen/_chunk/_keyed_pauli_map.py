import dataclasses
from typing import Literal, Callable, Any, Iterable, AbstractSet, Iterator

from gen._chunk._pauli_map import PauliMap


@dataclasses.dataclass(frozen=True)
class KeyedPauliMap:
    key: Any
    pauli_string: PauliMap

    @property
    def qubits(self) -> dict[complex, Literal["X", "Y", "Z"]]:
        return self.pauli_string.qubits

    def __lt__(self, other) -> bool:
        if isinstance(other, PauliMap):
            return True
        if isinstance(other, KeyedPauliMap):
            return (self.key, self.pauli_string) < (other.key, other.pauli_string)
        return NotImplemented

    def __gt__(self, other) -> bool:
        if isinstance(other, PauliMap):
            return False
        if isinstance(other, KeyedPauliMap):
            return (self.key, self.pauli_string) > (other.key, other.pauli_string)
        return NotImplemented

    def with_transformed_coords(
        self, transform: Callable[[complex], complex]
    ) -> "KeyedPauliMap":
        return KeyedPauliMap(
            key=self.key,
            pauli_string=self.pauli_string.with_transformed_coords(transform),
        )

    def with_xz_flipped(self) -> "KeyedPauliMap":
        return self.pauli_string.with_xz_flipped().keyed(self.key)

    def __str__(self):
        return f"(key={self.key}) {self.pauli_string}"

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
