"""Shared helpers for the experiment scripts.

Every script in this folder is run from the project root, e.g.
    python experiments/bench_engine.py
and writes its outputs to results/.
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.rcParams['figure.figsize'] = (7, 4)
matplotlib.rcParams['savefig.dpi'] = 150
matplotlib.rcParams['savefig.bbox'] = 'tight'


def out_dir(*parts):
    d = RESULTS.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=_jsonable))
    print('wrote', path)


def _jsonable(x):
    try:
        import numpy as np
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
    except ImportError:
        pass
    raise TypeError(type(x))


def save_csv(rows, path, columns=None):
    """rows: list of dicts with the same keys."""
    import csv
    columns = columns or list(rows[0])
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    print('wrote', path)


def savefig(fig, path):
    fig.savefig(path)
    fig.savefig(Path(path).with_suffix('.pdf'))
    print('wrote', path)


def run_grid(fn, configs, processes=None):
    """Run fn(config) for every config in parallel and return the list
    of results in order. Each config must be picklable (e.g. a dict)."""
    processes = processes or int(os.environ.get('N_PROCS', 0)) or max(1, (os.cpu_count() or 2) - 2)
    t0 = time.perf_counter()
    with Pool(processes) as pool:
        results = pool.map(fn, configs, chunksize=1)
    print(f'{len(configs)} runs in {time.perf_counter() - t0:.0f} s '
          f'on {processes} processes')
    return results
