import collections
import functools
import pathlib
from typing import (
    Iterable,
    Literal,
    Any,
    Callable,
    Sequence,
    Union,
    TYPE_CHECKING,
    Optional,
)

import stim

from gen._chunk._builder import Builder
from gen._chunk._complex_util import sorted_complex, min_max_complex
from gen._chunk._noise import NoiseRule
from gen._chunk._patch import Patch
from gen._chunk._tile import Tile
from gen._chunk._pauli_map import PauliMap
from gen._chunk._flow import Flow
from gen._util import write_file

if TYPE_CHECKING:
    import gen


class StabilizerCode:
    """This class stores the stabilizers and logicals of a stabilizer code.

    The exact semantics of the class are somewhat loose. For example, by default
    this class doesn't verify that its fields actually form a valid stabilizer
    code. This is so that the class can be used as a sort of useful data dumping
    ground even in cases where what is being built isn't a stabilizer code. For
    example, you can store a gauge code in the fields... it's just that methods
    like 'make_code_capacity_circuit' will no longer work.

    The stabilizers are stored as a `gen.Patch`. A patch is like a list of `gen.PauliMap`,
    except it actually stores `gen.Tile` instances so additional annotations can be added
    and additional utility methods are easily available.
    """

    def __init__(
        self,
        *,
        stabilizers: Iterable[Tile | PauliMap] | Patch | None = None,
        logicals: Iterable[PauliMap | tuple[PauliMap, PauliMap]] = (),
    ):
        """
        Args:
            stabilizers: The stabilizers of the code, specified as a Patch
            logicals: The logical qubits and/or observables of the code. Each entry should be
                either a pair of anti-commuting gen.PauliMap (e.g. the X and Z observables of the
                logical qubit) or a single gen.PauliMap (e.g. just the X observable).
        """
        packed_obs = []
        for obs in logicals:
            if isinstance(obs, PauliMap):
                packed_obs.append(obs)
            elif (
                len(obs) == 2
                and isinstance(obs[0], PauliMap)
                and isinstance(obs[1], PauliMap)
            ):
                packed_obs.append(tuple(obs))
            else:
                raise NotImplementedError(
                    f"{obs=} isn't a pauli product or anti-commuting pair of Pauli products."
                )
        if stabilizers is None:
            stabilizers = Patch([])
        elif not isinstance(stabilizers, Patch):
            stabilizers = Patch(stabilizers)

        self.stabilizers: Patch = stabilizers
        self.logicals: tuple[Union[PauliMap, tuple[PauliMap, PauliMap]], ...] = tuple(
            packed_obs
        )

    def with_integer_coordinates(self) -> "StabilizerCode":
        r2r = {v: i for i, v in enumerate(sorted({e.real for e in self.used_set}))}
        i2i = {v: i for i, v in enumerate(sorted({e.imag for e in self.used_set}))}
        return self.with_transformed_coords(lambda e: r2r[e.real] + i2i[e.imag] * 1j)

    def physical_to_logical(self, ps: stim.PauliString) -> "PauliMap":
        result = PauliMap()
        for q in ps.pauli_indices():
            if q >= len(self.logicals):
                raise ValueError("More qubits than logicals.")
            obs = self.logicals[q]
            if isinstance(obs, PauliMap):
                raise ValueError(
                    "Need logicals to be pairs of observables to map physical to logical."
                )
            p = ps[q]
            if p == 1:
                result *= obs[0]
            elif p == 2:
                result *= obs[0]
                result *= obs[1]
            elif p == 3:
                result *= obs[1]
            else:
                assert False
        return result

    def concat_over(
        self, under: "StabilizerCode", *, skip_inner_stabilizers: bool = False
    ) -> "StabilizerCode":
        over = self.with_integer_coordinates()
        c_min, c_max = min_max_complex(under.data_set)
        pitch = c_max - c_min + 1 + 1j

        def concatenated_obs(over_obs: PauliMap, under_index: int) -> PauliMap:
            total = PauliMap()
            for q, p in over_obs.items():
                x, z = under.logicals[under_index]
                if p == "X":
                    obs = x
                elif p == "Y":
                    obs = x * z
                elif p == "Z":
                    obs = z
                else:
                    raise NotImplementedError(f"{q=}, {p=}")
                total *= obs.with_transformed_coords(
                    lambda e: q.real * pitch.real + q.imag * pitch.imag * 1j + e
                )
            return total

        new_stabilizers = []
        for stabilizer in over.stabilizers:
            ps = stabilizer.to_data_pauli_string()
            for k in range(len(under.logicals)):
                new_stabilizers.append(
                    concatenated_obs(ps, k).to_tile().with_edits(flags=stabilizer.flags)
                )
        if not skip_inner_stabilizers:
            for stabilizer in under.stabilizers:
                for q in over.data_set:
                    new_stabilizers.append(
                        stabilizer.with_transformed_coords(
                            lambda e: q.real * pitch.real + q.imag * pitch.imag * 1j + e
                        )
                    )

        new_logicals = []
        for logical in over.logicals:
            for k in range(len(under.logicals)):
                if isinstance(logical, PauliMap):
                    new_logicals.append(concatenated_obs(logical, k))
                else:
                    x, z = logical
                    new_logicals.append(
                        (concatenated_obs(x, k), concatenated_obs(z, k))
                    )

        return StabilizerCode(
            stabilizers=new_stabilizers,
            logicals=new_logicals,
        )

    def get_observable_by_basis(
        self,
        index: int,
        basis: Literal["X", "Y", "Z"],
        *,
        default: Any = "__!not_specified",
    ) -> PauliMap:
        obs = self.logicals[index]
        if isinstance(obs, PauliMap) and set(obs.values()) == {basis}:
            return obs
        elif isinstance(obs, tuple):
            a1, a2 = obs
            b1 = frozenset(a1.values())
            b2 = frozenset(a2.values())
            if b1 == {basis}:
                return a1
            if b2 == {basis}:
                return a2
            if len(b1) == 1 and len(b2) == 1:
                # For example, we have X and Z specified and the user asked for Y.
                # Note that this works even if the X doesn't exactly overlap the Z.
                return a1 * a2
        if default != "__!not_specified":
            return default
        raise ValueError(f"Couldn't return a basis {basis} observable from {obs=}.")

    def auto_obs_passthrough_flows(
        self,
        *,
        obs_basis: Literal["X", "Y", "Z"],
        next_code: Optional["StabilizerCode"] = None,
    ) -> list["Flow"]:
        if next_code is None:
            next_code = self

        flows = []
        assert len(self.logicals) == len(next_code.logicals)
        obs_chosen = [
            (
                self.get_observable_by_basis(k, obs_basis),
                next_code.get_observable_by_basis(k, obs_basis),
            )
            for k in range(len(self.logicals))
        ]
        for k, (obs1, obs2) in enumerate(obs_chosen):
            flows.append(
                Flow(
                    start=obs1,
                    end=obs2,
                    measurement_indices="auto",
                    obs_key=k,
                    center=0,
                    flags={"obs"},
                )
            )
        return flows

    def round_auto_flows(
        self,
        *,
        obs_basis: Literal["X", "Y", "Z"],
        next_code: Optional["StabilizerCode"] = None,
    ) -> list["Flow"]:
        """Prepare and measure every stabilizer. Conserve observables."""
        if next_code is None:
            next_code = self

        flows = []
        for tile in self.stabilizers.tiles:
            if tile.data_set:
                flows.append(tile.to_measure_flow("auto"))
        for tile in next_code.tiles:
            if tile.data_set:
                flows.append(tile.to_prepare_flow("auto"))

        flows.extend(
            self.auto_obs_passthrough_flows(obs_basis=obs_basis, next_code=next_code)
        )
        return flows

    def list_pure_basis_observables(
        self, basis: Literal["X", "Y", "Z"]
    ) -> list[PauliMap]:
        result = []
        for k in range(len(self.logicals)):
            obs = self.get_observable_by_basis(k, basis, default=None)
            if obs is not None:
                result.append(obs)
        return result

    def x_basis_subset(self) -> "StabilizerCode":
        return StabilizerCode(
            stabilizers=self.stabilizers.with_only_x_tiles(),
            logicals=self.list_pure_basis_observables("X"),
        )

    def z_basis_subset(self) -> "StabilizerCode":
        return StabilizerCode(
            stabilizers=self.stabilizers.with_only_x_tiles(),
            logicals=self.list_pure_basis_observables("Z"),
        )

    @property
    def tiles(self) -> tuple["gen.Tile", ...]:
        return self.stabilizers.tiles

    def verify_distance_is_at_least_2(self):
        __tracebackhide__ = True
        self.verify()

        b: Any
        for b in "XZ":
            circuit = self.with_observables_from_basis(b).make_code_capacity_circuit(
                noise=1e-3
            )
            for inst in circuit.detector_error_model():
                if inst.type == "error":
                    dets = set()
                    obs = set()
                    for target in inst.targets_copy():
                        if target.is_relative_detector_id():
                            dets ^= {target.val}
                        elif target.is_logical_observable_id():
                            obs ^= {target.val}
                    dets = frozenset(dets)
                    obs = frozenset(obs)
                    if obs and not dets:
                        filter_det = stim.DetectorErrorModel()
                        filter_det.append(inst)
                        err = circuit.explain_detector_error_model_errors(
                            dem_filter=filter_det
                        )
                        loc = err[0].circuit_error_locations[0].flipped_pauli_product[0]
                        raise ValueError(
                            f"Code has a distance 1 error:"
                            f"\n    {loc.gate_target.pauli_type} at {loc.coords}"
                        )

    def verify_distance_is_at_least_3(self):
        __tracebackhide__ = True
        self.verify_distance_is_at_least_2()
        b: Any
        for b in "XZ":
            seen = {}
            circuit = self.with_observables_from_basis(b).make_code_capacity_circuit(
                noise=1e-3
            )
            for inst in circuit.detector_error_model().flattened():
                if inst.type == "error":
                    dets = set()
                    obs = set()
                    for target in inst.targets_copy():
                        if target.is_relative_detector_id():
                            dets ^= {target.val}
                        elif target.is_logical_observable_id():
                            obs ^= {target.val}
                    dets = frozenset(dets)
                    obs = frozenset(obs)
                    if dets not in seen:
                        seen[dets] = (obs, inst)
                    elif seen[dets][0] != obs:
                        filter_det = stim.DetectorErrorModel()
                        filter_det.append(inst)
                        filter_det.append(seen[dets][1])
                        err = circuit.explain_detector_error_model_errors(
                            dem_filter=filter_det,
                            reduce_to_one_representative_error=True,
                        )
                        loc1 = (
                            err[0].circuit_error_locations[0].flipped_pauli_product[0]
                        )
                        loc2 = (
                            err[1].circuit_error_locations[0].flipped_pauli_product[0]
                        )
                        raise ValueError(
                            f"Code has a distance 2 error:"
                            f"\n    {loc1.gate_target.pauli_type} at {loc1.coords}"
                            f"\n    {loc2.gate_target.pauli_type} at {loc2.coords}"
                        )

    def find_distance(self, *, max_search_weight: int) -> int:
        return len(self.find_logical_error(max_search_weight=max_search_weight))

    def find_logical_error(
        self, *, max_search_weight: int
    ) -> list[stim.ExplainedError]:
        circuit = self.make_code_capacity_circuit(noise=1e-3)
        if max_search_weight == 2:
            return circuit.shortest_graphlike_error(canonicalize_circuit_errors=True)
        return circuit.search_for_undetectable_logical_errors(
            dont_explore_edges_with_degree_above=max_search_weight,
            dont_explore_detection_event_sets_with_size_above=max_search_weight,
            dont_explore_edges_increasing_symptom_degree=False,
            canonicalize_circuit_errors=True,
        )

    def with_observables_from_basis(
        self, basis: Literal["X", "Y", "Z"]
    ) -> "StabilizerCode":
        if basis == "X":
            return StabilizerCode(
                stabilizers=self.stabilizers,
                logicals=self.list_pure_basis_observables("X"),
            )
        elif basis == "Y":
            return StabilizerCode(
                stabilizers=self.stabilizers,
                logicals=self.list_pure_basis_observables("Y"),
            )
        elif basis == "Z":
            return StabilizerCode(
                stabilizers=self.stabilizers,
                logicals=self.list_pure_basis_observables("Z"),
            )
        else:
            raise NotImplementedError(f"{basis=}")

    def mpp_init_chunk(self) -> "gen.Chunk":
        return self.mpp_chunk(flow_style="init")

    def mpp_end_chunk(self) -> "gen.Chunk":
        return self.mpp_chunk(flow_style="end")

    def mpp_chunk(
        self,
        *,
        noise: float | NoiseRule | None = None,
        flow_style: Literal["passthrough", "end", "init"] = "passthrough",
        resolve_anticommutations: bool = False,
    ) -> "gen.Chunk":
        assert flow_style in ["init", "end", "passthrough"]
        if resolve_anticommutations:
            observables, immune = self.entangled_observables(
                ancilla_qubits_for_xz_pairs=None
            )
            immune = set(immune)
        elif all(isinstance(obs, PauliMap) for obs in self.logicals):
            observables = self.logicals
            immune = set()
        else:
            raise NotImplementedError(
                f"{resolve_anticommutations=} but {self.logicals=}"
            )

        from gen._chunk import Flow, Chunk

        builder = Builder.for_qubits(self.data_set | immune)
        flows = []
        discards = []

        if noise is None or noise == 0:
            noise = NoiseRule()
        elif isinstance(noise, (float, int)):
            noise = NoiseRule(before={"DEPOLARIZE1": noise}, flip_result=noise)

        for k, obs in enumerate(observables):
            if flow_style != "passthrough":
                builder.append("MPP", [obs], measure_key_func=lambda _: f"obs{k}")
            flows.append(
                Flow(
                    center=-1,
                    start=None if flow_style == "init" else obs,
                    end=None if flow_style == "end" else obs,
                    measurement_indices=(
                        builder.lookup_recs([f"obs{k}"])
                        if flow_style != "passthrough"
                        else ()
                    ),
                    obs_key=k,
                )
            )

        for gate, strength in noise.before.items():
            builder.append(gate, self.data_set, arg=strength)
        for m, tile in enumerate(self.stabilizers.tiles):
            if tile.data_set:
                ps = tile.to_data_pauli_string()
                builder.append(
                    "MPP",
                    [ps],
                    measure_key_func=lambda _: f"det{m}",
                    arg=noise.flip_result,
                )
                if flow_style != "init":
                    flows.append(
                        tile.to_measure_flow(
                            measurement_indices=builder.lookup_recs([f"det{m}"])
                        )
                    )
                if flow_style != "end":
                    flows.append(
                        tile.to_prepare_flow(
                            measurement_indices=builder.lookup_recs([f"det{m}"])
                        )
                    )
        for gate, strength in noise.after.items():
            builder.append(gate, self.data_set, arg=strength)

        return Chunk(
            circuit=builder.circuit,
            q2i=builder.q2i,
            flows=flows,
            discarded_inputs=discards,
        )

    def as_interface(self) -> "gen.ChunkInterface":
        from gen._chunk._chunk_interface import ChunkInterface

        ports = []
        for tile in self.stabilizers.tiles:
            if tile.data_set:
                ports.append(tile.to_data_pauli_string())
        for k, obs in enumerate(self.logicals):
            if isinstance(obs, PauliMap):
                ports.append(obs.keyed(k))
            else:
                raise NotImplementedError(f"{obs=}")
        return ChunkInterface(ports=ports, discards=[])

    def with_edits(
        self,
        *,
        stabilizers: Iterable[Tile | PauliMap] | Patch | None = None,
        logicals: Iterable[PauliMap | tuple[PauliMap, PauliMap]] | None = None,
    ) -> "StabilizerCode":
        return StabilizerCode(
            stabilizers=self.stabilizers if stabilizers is None else stabilizers,
            logicals=self.logicals if logicals is None else logicals,
        )

    @functools.cached_property
    def data_set(self) -> frozenset[complex]:
        result = set(self.stabilizers.data_set)
        for obs in self.logicals:
            if isinstance(obs, PauliMap):
                result |= obs.keys()
            else:
                a, b = obs
                result |= a.keys()
                result |= b.keys()
        return frozenset(result)

    @functools.cached_property
    def measure_set(self) -> frozenset[complex]:
        return self.stabilizers.measure_set

    @functools.cached_property
    def used_set(self) -> frozenset[complex]:
        result = set(self.stabilizers.used_set)
        for obs in self.logicals:
            if isinstance(obs, PauliMap):
                result |= obs.keys()
            else:
                a, b = obs
                result |= a.keys()
                result |= b.keys()
        return frozenset(result)

    @staticmethod
    def from_patch_with_inferred_observables(patch: Patch) -> "StabilizerCode":
        q2i = {q: i for i, q in enumerate(sorted_complex(patch.data_set))}
        i2q = {i: q for q, i in q2i.items()}

        stabilizers: list[stim.PauliString] = []
        for tile in patch.tiles:
            stabilizer = stim.PauliString(len(q2i))
            for p, q in zip(tile.bases, tile.data_qubits):
                if q is not None:
                    stabilizer[q2i[q]] = p
            stabilizers.append(stabilizer)

        stabilizer_set: set[str] = set(str(e) for e in stabilizers)
        solved_tableau = stim.Tableau.from_stabilizers(
            stabilizers,
            allow_redundant=True,
            allow_underconstrained=True,
        )

        observables = []

        k: int = len(solved_tableau)
        while k > 0 and str(solved_tableau.z_output(k - 1)) not in stabilizer_set:
            k -= 1
            x = PauliMap(solved_tableau.x_output(k)).with_transformed_coords(
                i2q.__getitem__
            )
            z = PauliMap(solved_tableau.z_output(k)).with_transformed_coords(
                i2q.__getitem__
            )
            observables.append((x, z))

        return StabilizerCode(stabilizers=patch, logicals=observables)

    def with_epr_observables(self) -> "StabilizerCode":
        return StabilizerCode(
            stabilizers=self.stabilizers, logicals=self.entangled_observables()[0]
        )

    def entangled_observables(
        self,
        *,
        ancilla_qubits_for_xz_pairs: Sequence[complex] | None = None,
    ) -> tuple[list[PauliMap], list[complex]]:
        """Makes XZ observables commute by entangling them with ancilla qubits.

        This is useful when attempting to test all observables simultaneously.
        As long as noise is not applied to the ancilla qubits, the observables
        returned by this method cover the same noise as the original observables
        but the returned observables can be simultaneously measured.
        """
        num_pairs = sum(isinstance(e, tuple) for e in self.logicals)
        if ancilla_qubits_for_xz_pairs is None:
            a = (
                min(q.real for q in self.stabilizers.data_set)
                + min(q.imag for q in self.stabilizers.data_set) * 1j
                - 1j
            )
            ancilla_qubits_for_xz_pairs = [a + k for k in range(num_pairs)]
        else:
            assert len(ancilla_qubits_for_xz_pairs) == num_pairs
        next_ancilla_index = 0
        observables = []
        for obs in self.logicals:
            if isinstance(obs, tuple):
                ancilla = ancilla_qubits_for_xz_pairs[next_ancilla_index]
                next_ancilla_index += 1
                x, z = obs
                observables.append(x * PauliMap({ancilla: "X"}))
                observables.append(z * PauliMap({ancilla: "Z"}))
            else:
                observables.append(obs)
        return observables, list(ancilla_qubits_for_xz_pairs)

    def verify(self) -> None:
        """Verifies observables and stabilizers relate as a stabilizer code.

        All stabilizers should commute with each other.
        All stabilizers should commute with all observables.
        Same-index X and Z observables should anti-commute.
        All other observable pairs should commute.
        """
        __tracebackhide__ = True

        q2tiles = collections.defaultdict(list)
        for tile in self.stabilizers.tiles:
            for q in tile.data_set:
                q2tiles[q].append(tile)
        for tile1 in self.stabilizers.tiles:
            overlapping = {tile2 for q in tile1.data_set for tile2 in q2tiles[q]}
            for tile2 in overlapping:
                t1 = tile1.to_data_pauli_string()
                t2 = tile2.to_data_pauli_string()
                if not t1.commutes(t2):
                    raise ValueError(
                        f"Tile stabilizer {t1=} anticommutes with tile stabilizer {t2=}."
                    )

        flat_obs = []
        packed_obs = []
        for entry in self.logicals:
            if isinstance(entry, PauliMap):
                flat_obs.append(entry)
                packed_obs.append([entry])
            else:
                a, b = entry
                flat_obs.append(a)
                flat_obs.append(b)
                packed_obs.append(entry)

        for tile in self.stabilizers.tiles:
            ps = tile.to_data_pauli_string()
            for obs in flat_obs:
                if not ps.commutes(obs):
                    raise ValueError(
                        f"Tile stabilizer {tile=} anticommutes with {obs=}."
                    )

        for entry in self.logicals:
            if not isinstance(entry, PauliMap):
                a, b = entry
                if a.commutes(b):
                    raise ValueError(
                        f"The observable pair {a} vs {b} didn't anticommute."
                    )

        for k1 in range(len(self.logicals)):
            for k2 in range(k1 + 1, len(self.logicals)):
                for obs1 in packed_obs[k1]:
                    for obs2 in packed_obs[k2]:
                        if not obs1.commutes(obs2):
                            raise ValueError(
                                f"Unpaired observables didn't commute: {obs1=}, {obs2=}."
                            )

    def with_xz_flipped(self) -> "StabilizerCode":
        new_observables = []
        for entry in self.logicals:
            if isinstance(entry, PauliMap):
                new_observables.append(entry.with_xz_flipped())
            else:
                a, b = entry
                new_observables.append((a.with_xz_flipped(), b.with_xz_flipped()))
        return StabilizerCode(
            stabilizers=self.stabilizers.with_xz_flipped(),
            logicals=new_observables,
        )

    def write_svg(
        self,
        path: str | pathlib.Path,
        *,
        title: str | list[str] | None = None,
        canvas_height: int | None = None,
        show_order: bool | Literal["undirected", "3couplerspecial"] = False,
        show_measure_qubits: bool = False,
        show_data_qubits: bool = True,
        system_qubits: Iterable[complex] = (),
        opacity: float = 1,
        show_coords: bool = True,
        show_obs: bool = True,
        other: Union[
            None, "StabilizerCode", "Patch", Iterable[Union["StabilizerCode", "Patch"]]
        ] = None,
        tile_color_func: (
            Callable[
                ["gen.Tile"],
                str
                | tuple[float, float, float]
                | tuple[float, float, float, float]
                | None,
            ]
            | None
        ) = None,
        rows: int | None = None,
        cols: int | None = None,
        find_logical_err_max_weight: int | None = None,
        stabilizer_style: Literal["polygon", "circles"] = "polygon",
    ) -> None:
        flat = [self] if self is not None else []
        if isinstance(other, (StabilizerCode, Patch)):
            flat.append(other)
        elif other is not None:
            flat.extend(other)

        from gen._viz_patch_svg import patch_svg_viewer

        viewer = patch_svg_viewer(
            patches=flat,
            title=title,
            show_obs=show_obs,
            canvas_height=canvas_height,
            show_measure_qubits=show_measure_qubits,
            show_data_qubits=show_data_qubits,
            show_order=show_order,
            find_logical_err_max_weight=find_logical_err_max_weight,
            system_qubits=system_qubits,
            opacity=opacity,
            show_coords=show_coords,
            tile_color_func=tile_color_func,
            cols=cols,
            rows=rows,
            stabilizer_style=stabilizer_style,
        )
        write_file(path, viewer)

    def with_transformed_coords(
        self, coord_transform: Callable[[complex], complex]
    ) -> "StabilizerCode":
        new_observables = []
        for entry in self.logicals:
            if isinstance(entry, PauliMap):
                new_observables.append(entry.with_transformed_coords(coord_transform))
            else:
                a, b = entry
                new_observables.append(
                    (
                        a.with_transformed_coords(coord_transform),
                        b.with_transformed_coords(coord_transform),
                    )
                )
        return StabilizerCode(
            stabilizers=self.stabilizers.with_transformed_coords(coord_transform),
            logicals=new_observables,
        )

    def make_code_capacity_circuit(
        self,
        *,
        noise: float | NoiseRule,
        extra_coords_func: Callable[["Flow"], Iterable[float]] = lambda _: (),
    ) -> stim.Circuit:
        if isinstance(noise, (int, float)):
            noise = NoiseRule(after={"DEPOLARIZE1": noise})
        if noise.flip_result:
            raise ValueError(f"{noise=} includes measurement noise.")
        chunk1 = self.mpp_chunk(
            noise=NoiseRule(after=noise.after),
            flow_style="init",
            resolve_anticommutations=True,
        )
        chunk3 = self.mpp_chunk(
            noise=NoiseRule(before=noise.before),
            flow_style="end",
            resolve_anticommutations=True,
        )
        from gen._chunk._chunk_compiler import compile_chunks_into_circuit

        return compile_chunks_into_circuit(
            [chunk1, chunk3], flow_to_extra_coords_func=extra_coords_func
        )

    def make_phenom_circuit(
        self,
        *,
        noise: float | NoiseRule,
        rounds: int,
        extra_coords_func: Callable[["gen.Flow"], Iterable[float]] = lambda _: (),
    ) -> stim.Circuit:
        if isinstance(noise, (int, float)):
            noise = NoiseRule(after={"DEPOLARIZE1": noise}, flip_result=noise)
        chunk1 = self.mpp_chunk(
            noise=NoiseRule(after=noise.after),
            flow_style="init",
            resolve_anticommutations=True,
        )
        chunk2 = self.mpp_chunk(noise=noise, resolve_anticommutations=True)
        chunk3 = self.mpp_chunk(
            noise=NoiseRule(before=noise.before),
            flow_style="end",
            resolve_anticommutations=True,
        )
        from gen._chunk._chunk_compiler import compile_chunks_into_circuit

        return compile_chunks_into_circuit(
            [chunk1, chunk2 * rounds, chunk3],
            flow_to_extra_coords_func=extra_coords_func,
        )

    def __repr__(self) -> str:
        def indented(x: str) -> str:
            return x.replace("\n", "\n    ")

        def indented_repr(x: Any) -> str:
            if isinstance(x, tuple):
                return indented(
                    indented("[\n" + ",\n".join(indented_repr(e) for e in x)) + ",\n]"
                )
            return indented(repr(x))

        return f"""gen.StabilizerCode(
    patch={indented_repr(self.stabilizers)},
    observables={indented_repr(self.logicals)},
)"""

    def __eq__(self, other) -> bool:
        if not isinstance(other, StabilizerCode):
            return NotImplemented
        return self.stabilizers == other.stabilizers and self.logicals == other.logicals

    def __ne__(self, other) -> bool:
        return not (self == other)

    @functools.lru_cache(maxsize=1)
    def __hash__(self) -> int:
        return hash((StabilizerCode, self.stabilizers, self.logicals))

    def transversal_init_chunk(self, *, basis: Literal["X", "Y", "Z"]) -> "gen.Chunk":
        builder = Builder.for_qubits(self.data_set)
        builder.append(f"R{basis}", self.data_set)
        flows = []
        discards = []
        for tile in self.tiles:
            if tile.basis == basis:
                flows.append(tile.to_prepare_flow([]))
            else:
                discards.append(tile.to_data_pauli_string())
        kept_observables = self.list_pure_basis_observables(basis)
        for k, obs in enumerate(kept_observables):
            flows.append(
                Flow(
                    end=obs,
                    obs_key=k,
                    center=0,
                )
            )
        from gen._chunk._chunk import Chunk

        return Chunk(builder.circuit, flows=flows, discarded_outputs=discards)

    def transversal_measure_chunk(
        self, *, basis: Literal["X", "Y", "Z"]
    ) -> "gen.Chunk":
        return self.transversal_init_chunk(basis=basis).time_reversed()
