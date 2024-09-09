#!/usr/bin/env python3

import pathlib
import sys

src_path = pathlib.Path(__file__).parent.parent / 'src'
assert src_path.exists()
sys.path.append(str(src_path))


def main():
    from sinter._command._main import main
    main(command_line_args=['plot', *sys.argv[1:]])
    for k in range(len(sys.argv) - 1):
        if sys.argv[k] == '--out':
            print(f'wrote file://{pathlib.Path(sys.argv[k + 1]).absolute()}')
            return


if __name__ == '__main__':
    main()
