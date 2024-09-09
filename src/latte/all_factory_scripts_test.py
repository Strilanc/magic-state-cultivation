import pathlib
import sys

import pytest

from latte.factory_script import FactoryScript


def read_factories() -> list[str]:
    factories_path = pathlib.Path(__file__).parent.parent.parent / 'testdata' / 'factory_scripts'
    return [str(e.absolute()) for e in factories_path.iterdir()]


@pytest.mark.parametrize('factory_path', read_factories())
def test_verify_factories(factory_path: str):
    factory = FactoryScript.read_from_path(factory_path)
    q = factory._recompute_max_storage()
    d = factory.distance or 0
    if q >= 16:
        print(f"Skipping {pathlib.Path(factory_path).name} due to qubit count", file=sys.stderr)
        return
    skip_distance_check = False
    if d > 4:
        skip_distance_check = True
    elif d > 3 and q >= 10:
        skip_distance_check = True
    if skip_distance_check:
        print(f"Skipping distance check  in {pathlib.Path(factory_path).name} due to cost ({d=}, {q=})", file=sys.stderr)

    try:
        factory.verify(
            count_uncaught_benign_as_error=factory.num_t_outputs == 0,
            require_distance_exact=False,
            skip_distance_check=skip_distance_check,
        )
    except Exception:
        print(file=sys.stderr)
        print(factory.name, file=sys.stderr)
        print(factory.to_quirk_url(), file=sys.stderr)
        print('file://' + str(pathlib.Path(factory_path).absolute()), file=sys.stderr)
        raise
