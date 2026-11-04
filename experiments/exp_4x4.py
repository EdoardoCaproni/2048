"""TD(0) n-tuple learning on the real 4x4 game (target 2048, score reward).

Every run (tuple set, alpha, seed) trains in its own process and saves
    results/4x4/<run>/train.csv     statistics per block of 1000 games
    results/4x4/<run>/w_<games>.npz checkpoints
    results/4x4/<run>/config.json
Run from the project root:
    python experiments/exp_4x4.py                 # the whole grid
    python experiments/exp_4x4.py 100000          # fewer games per run
    python experiments/exp_4x4.py 100000 rcs      # one tuple set
Evaluate the checkpoints afterwards with eval_4x4.py.
"""
import csv
import sys
import time

import numpy as np

from common import out_dir, run_grid, save_json

import ntuple2048 as nt

N_GAMES = 300_000
CHECKPOINTS = (1_000, 3_000, 10_000, 30_000, 100_000, 200_000, 300_000, 500_000, 700_000, 1_000_000)
GRID = {  # tuple set -> (alphas, seeds)
    'rcs': ((0.05, 0.1, 0.25, 0.5), (0, 1)),
    'rect': ((0.1, 0.25, 0.5), (0, 1)),
}


def run_name(c):
    return f"{c['tuples']}_a{c['alpha']:g}_s{c['seed']}"


def train_one(c):
    net = nt.NTupleNet(4, 4, 2048, tuples=c['tuples'], symmetric=True,
                       dtype=np.float32)
    d = out_dir('4x4', run_name(c))
    save_json(dict(c, net=repr(net)), d / 'config.json')
    t0 = time.perf_counter()
    rows = []

    def checkpoint(net, games):
        net.save(d / f'w_{games}.npz')
        with open(d / 'train.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ['games'])
            w.writeheader()
            w.writerows(rows)

    hist = net.train(c['n_games'], alpha=c['alpha'], reward='score',
                     seed=c['seed'], block=1_000,
                     eval_at=[g for g in CHECKPOINTS if g <= c['n_games']],
                     checkpoint=checkpoint)
    rows[:] = hist
    checkpoint(net, net.games)
    last = hist[-1]
    print(f"{run_name(c):>16}: {len(net.cells)} features, {net.W.size:,} weights; "
          f"last block score {last['mean_score']:.0f}, P(2048) {last['p_target']:.3f}, "
          f"max tile {last['max_tile']} ({time.perf_counter() - t0:.0f} s)", flush=True)
    return run_name(c)


def main():
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else N_GAMES
    sets = sys.argv[2:] or list(GRID)
    if len(sys.argv) > 4:  # one configuration: exp_4x4.py GAMES TUPLES ALPHA SEED[,SEED...]
        configs = [dict(tuples=sys.argv[2], alpha=float(sys.argv[3]), seed=int(s), n_games=n_games)
                   for s in sys.argv[4].split(',')]
    else:
        configs = [dict(tuples=t, alpha=a, seed=s, n_games=n_games)
                   for t in sets for a in GRID[t][0] for s in GRID[t][1]]
    run_grid(train_one, configs)


if __name__ == '__main__':
    main()
