import functools
import pathlib
from typing import Iterable, Callable, Literal, Union, TYPE_CHECKING, Iterator, overload

from gen._chunk._complex_util import min_max_complex
from gen._chunk._tile import Tile
from gen._chunk._pauli_map import PauliMap
from gen._util import write_file

if TYPE_CHECKING:
    from gen._chunk._stabilizer_code import StabilizerCode


class Patch:
    """A collection of annotated stabilizers."""

    def __init__(self, tiles: Iterable[Tile | PauliMap], *, do_not_sort: bool = False):
        kept_tiles = []
        for tile in tiles:
            if isinstance(tile, Tile):
                kept_tiles.append(tile)
            elif isinstance(tile, PauliMap):
                kept_tiles.append(tile.to_tile())
            else:
                raise ValueError(
                    f"Don't know how to interpret this as a gen.Tile: {tile=}"
                )
        if not do_not_sort:
            kept_tiles = sorted(kept_tiles)

        self.tiles: tuple[Tile, ...] = tuple(kept_tiles)

    def __len__(self) -> int:
        return len(self.tiles)

    @overload
    def __getitem__(self, item: int) -> Tile:
        pass

    @overload
    def __getitem__(self, item: slice) -> "Patch":
        pass

    def __getitem__(self, item: Union[int, slice]) -> Union["Patch", Tile]:
        if isinstance(item, slice):
            return Patch(self.tiles[item])
        if isinstance(item, int):
            return self.tiles[item]
        raise NotImplementedError(f"{item=}")

    def __iter__(self) -> Iterator[Tile]:
        return self.tiles.__iter__()

    def with_edits(
        self,
        *,
        tiles: Iterable[Tile] | None = None,
    ) -> "Patch":
        return Patch(
            tiles=self.tiles if tiles is None else tiles,
        )

    def with_transformed_coords(
        self, coord_transform: Callable[[complex], complex]
    ) -> "Patch":
        return Patch(
            [e.with_transformed_coords(coord_transform) for e in self.tiles],
        )

    def with_transformed_bases(
        self,
        basis_transform: Callable[[Literal["X", "Y", "Z"]], Literal["X", "Y", "Z"]],
    ) -> "Patch":
        return Patch(
            [e.with_transformed_bases(basis_transform) for e in self.tiles],
        )

    def with_only_x_tiles(self) -> "Patch":
        return Patch([tile for tile in self.tiles if tile.basis == "X"])

    def with_only_y_tiles(self) -> "Patch":
        return Patch([tile for tile in self.tiles if tile.basis == "Y"])

    def with_only_z_tiles(self) -> "Patch":
        return Patch([tile for tile in self.tiles if tile.basis == "Z"])

    def without_wraparound_tiles(self) -> "Patch":
        p_min, p_max = min_max_complex(self.data_set, default=0)
        w = p_max.real - p_min.real
        h = p_max.imag - p_min.imag
        left = p_min.real + w * 0.1
        right = p_min.real + w * 0.9
        top = p_min.imag + h * 0.1
        bot = p_min.imag + h * 0.9

        def keep_tile(tile: Tile) -> bool:
            t_min, t_max = min_max_complex(tile.data_set, default=0)
            if t_min.real < left and t_max.real > right:
                return False
            if t_min.imag < top and t_max.imag > bot:
                return False
            return True

        return Patch([t for t in self.tiles if keep_tile(t)])

    @functools.cached_property
    def m2tile(self) -> dict[complex, Tile]:
        return {e.measure_qubit: e for e in self.tiles}

    def write_svg(
        self,
        path: str | pathlib.Path,
        *,
        title: str | list[str] | None = None,
        other: Union[
            "Patch", "StabilizerCode", Iterable[Union["Patch", "StabilizerCode"]]
        ] = (),
        show_order: bool | Literal["undirected", "3couplerspecial"] = False,
        show_measure_qubits: bool = False,
        show_data_qubits: bool = True,
        system_qubits: Iterable[complex] = (),
        show_coords: bool = True,
        opacity: float = 1,
        show_obs: bool = False,
        rows: int | None = None,
        cols: int | None = None,
        tile_color_func: Callable[[Tile], str] | None = None,
    ) -> None:
        from gen._viz_patch_svg import patch_svg_viewer

        from gen._chunk._stabilizer_code import StabilizerCode

        patches = [self] + (
            [other] if isinstance(other, (Patch, StabilizerCode)) else list(other)
        )

        viewer = patch_svg_viewer(
            patches=patches,
            show_measure_qubits=show_measure_qubits,
            show_data_qubits=show_data_qubits,
            show_order=show_order,
            system_qubits=system_qubits,
            opacity=opacity,
            show_coords=show_coords,
            show_obs=show_obs,
            rows=rows,
            cols=cols,
            tile_color_func=tile_color_func,
            title=title,
        )
        write_file(path, viewer)

    def with_xz_flipped(self) -> "Patch":
        trans: dict[Literal["X", "Y", "Z"], Literal["X", "Y", "Z"]] = {
            "X": "Z",
            "Y": "Y",
            "Z": "X",
        }
        return self.with_transformed_bases(trans.__getitem__)

    @functools.cached_property
    def used_set(self) -> frozenset[complex]:
        result = set()
        for e in self.tiles:
            result |= e.used_set
        return frozenset(result)

    @functools.cached_property
    def data_set(self) -> frozenset[complex]:
        result = set()
        for e in self.tiles:
            for q in e.data_qubits:
                if q is not None:
                    result.add(q)
        return frozenset(result)

    def __eq__(self, other):
        if not isinstance(other, Patch):
            return NotImplemented
        return self.tiles == other.tiles

    def __ne__(self, other):
        return not (self == other)

    @functools.cached_property
    def measure_set(self) -> frozenset[complex]:
        return frozenset(e.measure_qubit for e in self.tiles)

    def __add__(self, other: 'Patch') -> 'Patch':
        if not isinstance(other, Patch):
            return NotImplemented
        return Patch([*self, *other])

    def __repr__(self):
        return "\n".join(
            [
                "gen.Patch(tiles=[",
                *[f"    {e!r},".replace("\n", "\n    ") for e in self.tiles],
                "])",
            ]
        )
