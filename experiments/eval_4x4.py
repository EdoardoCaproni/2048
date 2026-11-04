"""Evaluate the 4x4 checkpoints and the heuristics by Monte Carlo.

Every checkpoint results/4x4/<run>/w_<games>.npz is played greedily
(no learning) for N_EVAL games with a fixed seed. The corner, greedy and
random heuristics are played with the same engine. Outputs:
    results/4x4/eval_all.csv      one row per checkpoint
    results/4x4/benchmarks.csv    one row per heuristic
Run from the project root:  python experiments/eval_4x4.py [n_games]
"""
import sys
import time
from pathlib import Path

import numpy as np

from common import RESULTS, run_grid, save_csv

import fast2048 as fb
import ntuple2048 as nt
from fast2048 import njit

N_EVAL = 1000
SEED = 12345
OUT = RESULTS / '4x4'


@njit(cache=True)
def play_heuristic(kind, rows, cols, target_exp, p4, n_games, seed):
    """kind 0: corner (first legal move in ACTIONS order), 1: greedy
    (largest points), 2: random. Returns (scores, max_exps, moves, won)."""
    fb.seed(seed)
    scores = np.zeros(n_games, dtype=np.int64)
    max_exps = np.zeros(n_games, dtype=np.int64)
    n_moves = np.zeros(n_games, dtype=np.int64)
    won = np.zeros(n_games, dtype=np.bool_)
    for g in range(n_games):
        x = fb.initial_board(rows, cols, p4)
        score = 0
        moves = 0
        while True:
            best_a = -1
            best_pts = -1
            best_after = x
            n_legal = 0
            k = -1
            if kind == 2:
                mask = fb.legal_mask(x, rows, cols)
                for a in range(4):
                    if mask & (1 << a):
                        n_legal += 1
                if n_legal > 0:
                    k = int(np.random.random() * n_legal)
            for a in range(4):
                after, pts, changed = fb.move(x, a, rows, cols)
                if not changed:
                    continue
                if kind == 0:
                    best_a, best_after = a, after
                    break
                if kind == 1 and pts > best_pts:
                    best_a, best_pts, best_after = a, pts, after
                if kind == 2:
                    if k == 0:
                        best_a, best_after = a, after
                        break
                    k -= 1
            if best_a < 0:
                break
            _, pts, _ = fb.move(x, best_a, rows, cols)
            score += pts
            moves += 1
            if fb.max_exp(best_after, rows, cols) >= target_exp:
                won[g] = True
            x = fb.spawn(best_after, rows, cols, p4)
        scores[g] = score
        max_exps[g] = fb.max_exp(x, rows, cols)
        n_moves[g] = moves
    return scores, max_exps, n_moves, won


def summary(scores, max_exps, moves, won):
    n = len(scores)
    row = {'games': n, 'mean_score': float(scores.mean()),
           'score_ci95': float(1.96 * scores.std(ddof=1) / np.sqrt(n)),
           'median_score': float(np.median(scores)),
           'mean_moves': float(moves.mean()),
           'p_2048': float(won.mean()),
           'p_2048_ci95': float(1.96 * np.sqrt(won.mean() * (1 - won.mean()) / n))}
    for e in range(8, 16):
        row[f'p_ge_{2 ** e}'] = float((max_exps >= e).mean())
    return row


def eval_checkpoint(path):
    path = Path(path)
    net = nt.NTupleNet.load(path)
    t0 = time.perf_counter()
    res = net.play(N_EVAL, 'score', seed=SEED, learn=False)
    row = dict(run=path.parent.name, trained_games=int(path.stem.split('_')[1]),
               **summary(*res), eval_s=time.perf_counter() - t0)
    print(f"{row['run']:>16} @ {row['trained_games']:>7}: score {row['mean_score']:8.0f} "
          f"P(2048) {row['p_2048']:.3f} P(4096) {row['p_ge_4096']:.3f} ({row['eval_s']:.0f} s)",
          flush=True)
    return row


def main():
    global N_EVAL
    if len(sys.argv) > 1:
        N_EVAL = int(sys.argv[1])
    bench = []
    for kind, name in enumerate(('corner', 'greedy', 'random')):
        res = play_heuristic(kind, 4, 4, 11, 0.1, max(N_EVAL, 10_000), SEED)
        bench.append(dict(policy=name, **summary(*res)))
        print(f"{name:>8}: score {bench[-1]['mean_score']:7.0f}  "
              f"P(512) {bench[-1]['p_ge_512']:.3f}  P(1024) {bench[-1]['p_ge_1024']:.3f}  "
              f"P(2048) {bench[-1]['p_2048']:.4f}", flush=True)
    save_csv(bench, OUT / 'benchmarks.csv')
    paths = sorted(OUT.glob('*/w_*.npz'))
    rows = run_grid(eval_checkpoint, [str(p) for p in paths])
    rows.sort(key=lambda r: (r['run'], r['trained_games']))
    save_csv(rows, OUT / 'eval_all.csv')


if __name__ == '__main__':
    main()
