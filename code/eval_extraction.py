"""Measure how accurately the LLM turns written requests into scenario inputs.

Needs OPENAI_API_KEY. Prints field-level accuracy over evals/request_extraction_cases.json
and the time each extraction took, so the "faster scenario preparation" claim is measured.

    python code/eval_extraction.py
"""
import json
import os
import time

import fab_data as fd
import planning_copilot as rag
import request_parser as P

FIELDS = ['site', 'program', 'quarter', 'requested_wspw', 'protected']


def main():
    key = os.getenv('OPENAI_API_KEY')
    if not key:
        raise SystemExit('Set OPENAI_API_KEY to run the extraction evaluation.')
    model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')
    cases = json.loads((fd.ROOT / 'evals' / 'request_extraction_cases.json').read_text())
    data, index = fd.load(), rag.DocumentIndex(fd.DOCS)
    hits = {f: 0 for f in FIELDS}
    for case in cases:
        start = time.perf_counter()
        try:
            got = P.extract(case['text'], data, index, key, model)
            inputs, warnings = got['inputs'], got['warnings']
        except Exception as exc:  # report and keep scoring
            inputs, warnings = {}, [str(exc)]
        seconds = time.perf_counter() - start
        row = []
        for f in FIELDS:
            want, have = case['expect'][f], inputs.get(f)
            ok = sorted(want) == sorted(have or []) if f == 'protected' else want == have
            hits[f] += ok
            row.append(f"{f}={'ok' if ok else f'{have!r}≠{want!r}'}")
        print(f"{case['id']} {seconds:4.1f}s  " + '  '.join(row) + (f"  warnings={warnings}" if warnings else ''))
    print('\nField accuracy:', {f: f'{hits[f]}/{len(cases)}' for f in FIELDS})


if __name__ == '__main__':
    main()
