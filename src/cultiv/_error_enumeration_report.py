import dataclasses
import pathlib

import stim

import gen
from ._error_set import DemErrorSet, DemCombinedError


@dataclasses.dataclass(frozen=True)
class ErrorEnumerationReport:
    heralded_error_rate: float
    keep_rate: float
    distance_to_involved_physical_errors: dict[int, tuple[int, ...]]
    distance_to_heralded_error_rate: dict[int, float]
    error_set: DemErrorSet
    logical_errs: list[DemCombinedError]

    @property
    def discard_rate(self) -> float:
        return 1 - self.keep_rate

    @property
    def retry_gain_factor(self) -> float:
        if self.keep_rate == 0:
            return float('inf')
        return 1 / self.keep_rate

    @staticmethod
    def read_cache_file(cache_file: str | pathlib.Path) -> dict[str, list[tuple[int, ...]]]:
        cache = {}
        with open(cache_file, 'r') as f:
            entries = f.read().split('ENTRY ')
            for entry in entries:
                if not entry.strip():
                    continue
                strong_id, *terms = entry.split('\n')
                strong_id = strong_id.strip()
                errs = []
                for term in terms:
                    term = term.strip()
                    if term:
                        errs.append(tuple(int(e) for e in term.split(',')))
                cache[strong_id] = errs
        return cache

    @staticmethod
    def from_circuit(
            circuit: stim.Circuit,
            *,
            max_weight: int, noise: None | float | gen.NoiseModel = None,
            cache: dict[str, list[tuple[int, ...]]],
    ) -> 'ErrorEnumerationReport':
        if isinstance(noise, float):
            noise = gen.NoiseModel.uniform_depolarizing(noise)
        if noise is not None:
            circuit = noise.noisy_circuit_skipping_mpp_boundaries(circuit)
        dem = circuit.detector_error_model()
        if dem.num_errors == 0:
            raise ValueError("dem.num_errors == 0")
        if dem.num_observables == 0:
            raise ValueError("dem.num_observables == 0")
        return ErrorEnumerationReport.from_dem(dem, max_weight=max_weight, cache=cache)

    @staticmethod
    def from_dem(
            dem: stim.DetectorErrorModel,
            *,
            max_weight: int,
            cache: dict[str, list[tuple[int, ...]]],
    ) -> 'ErrorEnumerationReport':
        err_set = DemErrorSet.from_dem(dem)
        keep_rate = 1
        for err in err_set.errors:
            keep_rate *= 1 - err.p

        key = err_set.strong_id(max_weight=max_weight)
        if key not in cache:
            print("    cache miss", key)
            cache[key] = err_set.find_logical_errors(max_distance=max_weight)
        logical_errs = err_set.expand_logical_errors(cache[key])

        distance_to_involved_physical_errors = {
            d: {
                e
                for err in logical_errs
                if len(err.src_errors) == d
                for e in err.src_errors
            }
            for d in range(max_weight + 1)
        }

        distance_to_heralded_error_rate = {
            d: sum(
                err.p
                for err in logical_errs
                if len(err.src_errors) == d
            ) / keep_rate
            for d in range(max_weight + 1)
        }

        heralded_error_rate = sum(err.p for err in logical_errs) / keep_rate

        return ErrorEnumerationReport(
            heralded_error_rate=heralded_error_rate,
            keep_rate=keep_rate,
            distance_to_involved_physical_errors=distance_to_involved_physical_errors,
            distance_to_heralded_error_rate=distance_to_heralded_error_rate,
            error_set=err_set,
            logical_errs=logical_errs,
        )
