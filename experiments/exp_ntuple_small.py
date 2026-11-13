"""Semi-gradient Sarsa with n-tuple features on the small boards, validated against the exact DP.

Phase 1 (parallel): every configuration is trained with the numba code
and the weights are saved at the checkpoints in EVAL_AT.
Phase 2 (sequential, in the main process because the DP of the bigger
boards needs about 1.5 GB): every checkpoint is evaluated exactly with
tabular_control.ExactEvaluator (exact P(win) of the greedy policy, agreement with
the optimal policy, weighted RMSE of the values against V*).

Outputs, under results/ntuple_small/<board>/:
    <run>/train.csv       training statistics per block
    <run>/w_<games>.npz   checkpoints
    <run>/exact.csv       exact quantities per checkpoint
    exact_all.csv         all the exact rows of the board
    benchmarks.json       exact P(win) of optimal, corner, greedy, random

Run from the project root:
    python experiments/exp_ntuple_small.py            # all the boards
    python experiments/exp_ntuple_small.py 2x3_64     # one board
    python experiments/exp_ntuple_small.py 2x3_64 train|eval
"""
import csv
import json
import sys
import time

import numpy as np

from common import RESULTS, out_dir, run_grid, save_csv, save_json

import dynamic_programming as dp
import ntuple_network as nt
import tabular_control as rl
from game import Game2048

OUT = RESULTS / 'ntuple_small'
ALPHA = 0.1

# board name -> game, seeds, checkpoints and (tuples, symmetric, reward,
# init, eps) configurations
BOARDS = {
    '2x3_64': dict(
        game=dict(rows=2, cols=3, target=64), with_score=True, seeds=range(3),
        n_games=300_000, eval_at=(1_000, 3_000, 10_000, 30_000, 100_000, 300_000),
        configs=[('full', False, 'win', 1.0, 0.0),     # tabular, optimistic
                 ('full', False, 'win', 0.0, 0.0),     # tabular, sparse reward: fails
                 ('rcs', True, 'win', 1.0, 0.0),
                 ('rcs', False, 'win', 1.0, 0.0),
                 ('rcs', True, 'score', 0.0, 0.0),
                 ('full', False, 'score', 0.0, 0.0),   # no exploration: fails
                 ('full', False, 'score', 0.0, 0.05),  # epsilon-greedy
                 ('full', False, 'score', 480.0, 0.0)]),  # optimistic
    '2x4_128': dict(
        game=dict(rows=2, cols=4, target=128), with_score=False, seeds=range(2),
        n_games=300_000, eval_at=(10_000, 30_000, 100_000, 300_000),
        configs=[('rcs', True, 'win', 0.0, 0.0),
                 ('rcs', True, 'win', 1.0, 0.0),
                 ('rcs', False, 'win', 0.0, 0.0),
                 ('rect', True, 'win', 0.0, 0.0),
                 ('rcs', True, 'score', 0.0, 0.0),
                 ('rect', True, 'score', 0.0, 0.0)]),
    '3x3_64': dict(
        game=dict(rows=3, cols=3, target=64), with_score=False, seeds=range(1),
        n_games=300_000, eval_at=(10_000, 30_000, 100_000, 300_000),
        configs=[('rcs', True, 'win', 0.0, 0.0),
                 ('rcs', False, 'win', 0.0, 0.0),
                 ('rect', True, 'win', 0.0, 0.0),
                 ('rcs', True, 'score', 0.0, 0.0),
                 ('rect', True, 'score', 0.0, 0.0)]),
}


def run_name(c):
    tuples, symmetric, reward, init, eps = c['config']
    return (f"{tuples}_{'sym' if symmetric else 'nosym'}_{reward}"
            f"_i{init:g}_e{eps:g}_s{c['seed']}")


def cache_path(board):
    g = BOARDS[board]['game']
    return RESULTS / 'dp_cache' / f"eval_{g['rows']}x{g['cols']}_{g['target']}.pkl"


def train_one(c):
    spec = BOARDS[c['board']]
    tuples, symmetric, reward, init, eps = c['config']
    g = spec['game']
    net = nt.NTupleNet(g['rows'], g['cols'], g['target'], tuples=tuples,
                       symmetric=symmetric, dtype=np.float64, init=init)
    d = out_dir('ntuple_small', c['board'], run_name(c))

    def checkpoint(net, games):
        net.save(d / f'w_{games}.npz')

    t0 = time.perf_counter()
    hist = net.train(spec['n_games'], alpha=ALPHA, reward=reward, seed=c['seed'],
                     block=1_000, eps=eps, eval_at=spec['eval_at'],
                     checkpoint=checkpoint)
    with open(d / 'train.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    print(f"{c['board']} {run_name(c):>28}: {len(net.cells)} features, "
          f"{net.W.size:,} weights, final score {hist[-1]['mean_score']:.1f}, "
          f"P(target) {hist[-1]['p_target']:.3f} ({time.perf_counter() - t0:.0f} s)",
          flush=True)
    return run_name(c)


def evaluate_board(board):
    spec = BOARDS[board]
    game = Game2048(**spec['game'])
    t0 = time.perf_counter()
    ev = rl.ExactEvaluator.cached(game, cache_path(board), spec['with_score'])
    boards = [b for layer in ev.layers_win.values() for b in layer]
    print(f'{board}: evaluator with {len(boards):,} states ready '
          f'({time.perf_counter() - t0:.0f} s)', flush=True)
    bench = {'optimal': dp.start_value(game, ev.V_win)}
    for name, pol in dp.HEURISTICS.items():
        bench[name] = dp.start_value(game, dp.evaluate_policy(game, pol, 'win', ev.layers_win))
    save_json(bench, out_dir('ntuple_small', board) / 'benchmarks.json')
    all_rows = []
    for config in spec['configs']:
        for seed in spec['seeds']:
            c = dict(board=board, config=config, seed=seed)
            d = OUT / board / run_name(c)
            path = d / 'exact.csv'
            done = {}
            if path.exists():
                with open(path) as f:
                    done = {int(r['games']): r for r in csv.DictReader(f)}
            rows = []
            for games in spec['eval_at']:
                if games in done:
                    rows.append(done[games])
                    continue
                w = d / f'w_{games}.npz'
                if not w.exists():
                    continue
                t0 = time.perf_counter()
                net = nt.NTupleNet.load(w)
                row = ev.evaluate(net.frozen(boards, config[2]))
                row = dict(run=run_name(c), games=games, **row,
                           eval_s=time.perf_counter() - t0)
                rows.append(row)
                print(f"  {run_name(c):>28} @ {games:>7}: P(win) {row['p_win']:.4f} "
                      f"agree {row['agree']:.3f} rmse_w {row.get('rmse_w', float('nan')):.4f} "
                      f"({row['eval_s']:.0f} s)", flush=True)
                save_csv(rows, path)
            all_rows += [dict(r, tuples=config[0], symmetric=config[1],
                              reward=config[2], init=config[3], eps=config[4],
                              seed=seed) for r in rows]
    if all_rows:
        save_csv(all_rows, OUT / board / 'exact_all.csv')


def main():
    boards = [sys.argv[1]] if len(sys.argv) > 1 else list(BOARDS)
    phases = sys.argv[2:] or ['train', 'eval']
    if 'train' in phases:
        configs = [dict(board=b, config=c, seed=s)
                   for b in boards for c in BOARDS[b]['configs']
                   for s in BOARDS[b]['seeds']]
        run_grid(train_one, configs)
    if 'eval' in phases:
        for b in boards:
            evaluate_board(b)


if __name__ == '__main__':
    main()
