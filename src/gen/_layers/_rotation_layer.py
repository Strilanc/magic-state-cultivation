import dataclasses

import sinter
import stim

from gen._layers._data import (
    single_qubit_clifford_inverse_table,
    single_qubit_clifford_multiplication_table,
)
from gen._layers._layer import Layer


@dataclasses.dataclass
class RotationLayer(Layer):
    named_rotations: dict[int, str] = dataclasses.field(default_factory=dict)

    def touched(self) -> set[int]:
        return {k for k, v in self.named_rotations.items() if v != "I"}

    def copy(self) -> "RotationLayer":
        return RotationLayer(dict(self.named_rotations))

    def inverse(self) -> "RotationLayer":
        t = single_qubit_clifford_inverse_table()
        return RotationLayer({q: t[r] for q, r in self.named_rotations.items()})

    def append_into_stim_circuit(self, out: stim.Circuit) -> None:
        v = sinter.group_by(self.named_rotations.items(), key=lambda e: e[1])
        for gate, items in sorted(v.items()):
            qs = sorted(q for q, _ in items)
            if gate != "I":
                if "*" in gate:
                    after, before = gate.split("*")
                    out.append(before, qs)
                    out.append(after, qs)
                else:
                    out.append(gate, qs)

    def prepend_named_rotation(self, name: str, target: int):
        m = single_qubit_clifford_multiplication_table()
        cur = self.named_rotations.get(target, "I")
        new_val = m[(cur, name)]
        if new_val == "I":
            self.named_rotations.pop(target, None)
        else:
            self.named_rotations[target] = new_val

    def append_named_rotation(self, name: str, target: int):
        m = single_qubit_clifford_multiplication_table()
        cur = self.named_rotations.get(target, "I")
        new_val = m[(name, cur)]
        if new_val == "I":
            self.named_rotations.pop(target, None)
        else:
            self.named_rotations[target] = new_val

    def is_vacuous(self) -> bool:
        return not any(self.named_rotations.values())

    def locally_optimized(self, next_layer: Layer | None) -> list[Layer | None]:
        from gen._layers._det_obs_annotation_layer import DetObsAnnotationLayer
        from gen._layers._feedback_layer import FeedbackLayer
        from gen._layers._reset_layer import ResetLayer
        from gen._layers._shift_coord_annotation_layer import ShiftCoordAnnotationLayer

        if isinstance(next_layer, (DetObsAnnotationLayer, ShiftCoordAnnotationLayer)):
            return [next_layer, self]
        if isinstance(next_layer, FeedbackLayer):
            return [next_layer.before(self), self]
        if isinstance(next_layer, ResetLayer):
            trimmed = self.copy()
            for t in next_layer.targets.keys():
                trimmed.named_rotations.pop(t, None)
            if trimmed.named_rotations:
                return [trimmed, next_layer]
            else:
                return [next_layer]
        if isinstance(next_layer, RotationLayer):
            result = self.copy()
            for q, r in next_layer.named_rotations.items():
                result.append_named_rotation(r, q)
            return [result]
        return [self, next_layer]
