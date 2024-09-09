#!/usr/bin/env python3

import argparse
import multiprocessing
import pathlib
import sys
from typing import Any

import sinter

src_path = pathlib.Path(__file__).parent.parent / 'src'
assert src_path.exists()
sys.path.append(str(src_path))

import gen
import cultiv


def report_on_circuit(arg):
    print(f"Enumerating d={arg['d']} c={arg['style']} p={arg['p']}...", file=sys.stderr)
    p = arg['p']
    cache = arg['cache']
    report = cultiv.ErrorEnumerationReport.from_circuit(arg['circuit'], noise=p, max_weight=5, cache=cache)
    return {
        **{
            k: v
            for k, v in arg.items()
            if k != 'cache'
        },
        'report': report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache_file', required=True, type=str)
    parser.add_argument('--p', default=None, type=float, nargs='+')
    parser.add_argument('--c', default=None, type=str, nargs='+')
    args = parser.parse_args()

    print(sinter.CSV_HEADER)
    ps = args.p if args.p is not None else [1e-4, 2e-4, 3e-4, 5e-4, 1e-3, 2e-3]
    styles = args.c if args.c is not None else ['degenerate', 'bell', 'unitary']
    cache = cultiv.ErrorEnumerationReport.read_cache_file(args.cache_file)

    style: Any
    inputs = []
    for style in styles:
        for d in [3, 5]:
            if d == 5 and style != 'unitary':
                continue
            circuit = cultiv.make_inject_and_cultivate_circuit(dcolor=d, inject_style=style, basis='Y')
            for p in ps:
                inputs.append({
                    'p': p,
                    'circuit': circuit,
                    'd': d,
                    'style': style,
                    'cache': cache,
                })
    all_results = []
    for k, r in enumerate(multiprocessing.Pool().imap_unordered(report_on_circuit, inputs)):
        report: cultiv.ErrorEnumerationReport = r['report']
        all_results.append({**r, 'r': gen.count_measurement_layers(r['circuit']), 'q': r['circuit'].num_qubits})
        print(sinter.TaskStats(
            strong_id=f'refref{k}',
            decoder='enumeration',
            json_metadata={
                k: v for k, v in r.items() if k != 'circuit' and k != 'cache' and k != 'report'
            },
            shots=10**20,
            errors=round(10**20 * (1 - report.discard_rate) * report.heralded_error_rate),
            discards=round(10**20 * report.discard_rate),
        ), flush=True)


if __name__ == '__main__':
    main()
