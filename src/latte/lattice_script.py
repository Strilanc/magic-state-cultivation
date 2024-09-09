import random
from typing import Iterable, TYPE_CHECKING

import gen
from latte.lattice_surgery_layer import LatticeSurgeryLayer, InjectedError
from latte.lattice_surgery_layer_with_feedback import \
    LatticeSurgeryLayerWithFeedback
from latte.vec_sim import VecSim

if TYPE_CHECKING:
    import pygltflib


class LatticeScript:
    def __init__(self, *, layers_with_feedback: Iterable[LatticeSurgeryLayerWithFeedback]):
        self.layers_with_feedback: tuple[LatticeSurgeryLayerWithFeedback, ...] = tuple(layers_with_feedback)

    @staticmethod
    def from_str(content: str) -> 'LatticeScript':
        layers_with_feedback: list[LatticeSurgeryLayerWithFeedback] = []
        cur_lines = []
        seen_keys = set()
        for line in (content + '\n=====').splitlines():
            if '#' in line:
                line = line.split('#')[0]
            if '=====' in line:
                if any(e.strip() for e in cur_lines):
                    new_layer = LatticeSurgeryLayerWithFeedback.from_content('\n'.join(cur_lines))
                    for k in [e.name for e in new_layer.measure_actions] + [e.name for e in new_layer.let_actions]:
                        if k in seen_keys:
                            raise ValueError(f"Value redefined: {k!r}")
                        seen_keys.add(k)
                    layers_with_feedback.append(new_layer)
                cur_lines.clear()
            else:
                cur_lines.append(line)

        return LatticeScript(layers_with_feedback=layers_with_feedback)

    def list_edge_errors(self) -> list[InjectedError]:
        result = []
        seen = set()

        ms = {}
        t = 0
        for feedback_layer in self.layers_with_feedback:
            layer = feedback_layer.make_layer(ms)
            for m in feedback_layer.measure_actions:
                ms[m.name] = False if m.ground else bool(random.randrange(2))
            for m in feedback_layer.let_actions:
                m.try_assign(ms)
            for err in layer.list_edge_errors():
                err = err.time_shifted_by(t)
                if err not in seen:
                    result.append(err)
                    seen.add(err)
            t += 1

        return result

    def simulate(self, *, injected_errors: Iterable[InjectedError] = ()) -> tuple[str, dict[str, bool]]:
        injected_errors = frozenset(gen.xor_sorted(injected_errors))
        sim = VecSim()
        state: dict[str, bool] = {}
        t = 0
        for layer_with_feedback in self.layers_with_feedback:
            r = layer_with_feedback.run(
                sim,
                state,
                injected_errors=frozenset(
                    err.time_shifted_by(-t)
                    for err in injected_errors
                    if err.layer == t - 0.5 or err.layer == t
                ))
            if r != 'pass':
                return r, state
            t += 1

        return 'correct', state

    def to_3d_gltf_model(
            self,
            *,
            spacing: float = 2,
            ignore_contradictions: bool = False,
            wireframe: bool = False,
            injected_errors: Iterable[InjectedError] = (),
            randomly_resolve_nodes: bool = False,
    ) -> 'pygltflib.GLTF2':
        ms = {}
        layers = []
        for feedback_layer in self.layers_with_feedback:
            layer = feedback_layer.make_layer(ms, skip_resolving=not randomly_resolve_nodes)
            for m in feedback_layer.measure_actions:
                ms[m.name] = False if m.ground else bool(random.randrange(2))
            for m in feedback_layer.let_actions:
                m.try_assign(ms)
            layers.append(layer)
        return LatticeSurgeryLayer.combined_3d_model_gltf(
            layers,
            spacing=spacing,
            ignore_contradictions=ignore_contradictions,
            wireframe=wireframe,
            injected_errors=injected_errors,
        )
