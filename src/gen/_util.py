import io
import pathlib
from typing import Callable, Any, TypeVar, Iterable


def write_file(path: str | pathlib.Path | io.IOBase, content: Any):
    if isinstance(path, io.IOBase):
        path.write(content)
        return
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    if isinstance(content, bytes):
        with open(path, "wb") as f:
            print(content, file=f)
    else:
        with open(path, "w") as f:
            print(content, file=f)
    print(f"wrote file://{pathlib.Path(path).absolute()}")


TItem = TypeVar("TItem")


def xor_sorted(
    vals: Iterable[TItem], *, key: Callable[[TItem], Any] = lambda e: e
) -> list[TItem]:
    """Sorts items and then cancels pairs of equal items.

    An item will be in the result once if it appeared an odd number of times.
    An item won't be in the result if it appeared an even number of times.
    """
    result = sorted(vals, key=key)
    n = len(result)
    skipped = 0
    k = 0
    while k + 1 < n:
        if result[k] == result[k + 1]:
            skipped += 2
            k += 2
        else:
            result[k - skipped] = result[k]
            k += 1
    if k < n:
        result[k - skipped] = result[k]
    while skipped:
        result.pop()
        skipped -= 1
    return result
