"""Tabular RL on the 2x3 board (target 64) with exact learning curves.

Every (method, reward, initialisation) is run with several seeds in
parallel. Every eval_every episodes the exact value of the learned greedy
policy is computed with the DP. Outputs:
    results/tabular/<method>_<reward>_q<init>_s<seed>.csv   learning curves
    results/tabular/summary.csv                             final rows
Run from the project root:  python experiments/exp_tabular_2x3.py
"""
import csv
import sys
import time

from common import RESULTS, out_dir, run_grid, save_csv

import rl2048 as rl
from game2048 import Env2048, Game2048

BOARD = dict(rows=2, cols=3, target=64)
N_EPISODES = 200_000
EVAL_EVERY = 2_000
SEEDS = range(5)
CACHE = RESULTS / 'dp_cache' / 'eval_2x3_64.pkl'
OUT = out_dir('tabular')

# (reward, q_init): dense score reward; sparse win reward from zero and
# with optimistic initialisation
SETTINGS = [('score', 0.0), ('win', 0.0), ('win', 1.0)]
ALPHA = {'mc_control': None, 'sarsa': 0.1, 'q_learning': 0.1,
         'expected_sarsa': 0.1, 'nstep_sarsa': 0.1, 'td_afterstate': 0.1}
EPSILON = 0.1
N_STEP = 3


def run_name(cfg):
    return f"{cfg['method']}_{cfg['reward']}_q{cfg['q_init']:g}_s{cfg['seed']}"


def run(cfg):
    game = Game2048(**BOARD)
    ev = rl.ExactEvaluator.cached(game, CACHE)
    env = Env2048(game, reward=cfg['reward'])
    t0 = time.perf_counter()
    _, hist = rl.train(cfg['method'], env, N_EPISODES, alpha=ALPHA[cfg['method']],
                       epsilon=EPSILON, n=N_STEP, q_init=cfg['q_init'],
                       seed=cfg['seed'], eval_every=EVAL_EVERY, evaluator=ev)
    path = OUT / (run_name(cfg) + '.csv')
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    final = dict(cfg, **hist[-1], wall_s=time.perf_counter() - t0)
    print(f"{run_name(cfg):>32}: P(win) {final['p_win']:.4f} score {final['score']:.1f} "
          f"({final['wall_s']:.0f} s)", flush=True)
    return final


def main():
    game = Game2048(**BOARD)
    t0 = time.perf_counter()
    rl.ExactEvaluator.cached(game, CACHE)  # build once, workers load it
    print(f'evaluator ready ({time.perf_counter() - t0:.1f} s)', flush=True)
    configs = [dict(method=m, reward=r, q_init=q, seed=s)
               for r, q in SETTINGS for m in rl.METHODS for s in SEEDS]
    if len(sys.argv) > 1:  # quick test: python exp_tabular_2x3.py quick
        global N_EPISODES
        N_EPISODES = 10_000
        configs = configs[::15]
    finals = run_grid(run, configs)
    save_csv(finals, OUT / 'summary.csv')


if __name__ == '__main__':
    main()
