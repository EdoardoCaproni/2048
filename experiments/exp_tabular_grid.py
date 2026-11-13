"""Hyper-parameters of the tabular methods on the 2x3 board (target 64).

Two experiments, score reward, exact evaluation at the end:
  grid      Q-learning and Expected Sarsa on a grid of constant alpha x
            epsilon (200k episodes, 2 seeds)
  schedule  all the methods with decaying schedules (epsilon from 0.3 to
            0.01, alpha from 0.2 to 0.02, linearly over the run), 5 seeds,
            exact evaluation every 2k episodes: the curves to compare with
            the constant-parameter runs of exp_tabular_2x3.py
Outputs: results/tabular_grid/grid.csv, results/tabular_grid/schedule_*.csv,
fig_tab_grid.png (final exact P(win) over the grid).
Run from the project root:  python experiments/exp_tabular_grid.py [grid|schedule]
"""
import csv
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, out_dir, run_grid, save_csv, savefig

import tabular_control as rl
from game import Env2048, Game2048

BOARD = dict(rows=2, cols=3, target=64)
N_EPISODES = 200_000
CACHE = RESULTS / 'dp_cache' / 'eval_2x3_64.pkl'
OUT = out_dir('tabular_grid')
ALPHAS = (0.02, 0.05, 0.1, 0.2, 0.4)
EPSILONS = (0.02, 0.05, 0.1, 0.2, 0.3)


def run_point(cfg):
    game = Game2048(**BOARD)
    ev = rl.ExactEvaluator.cached(game, CACHE)
    env = Env2048(game, reward='score')
    t0 = time.perf_counter()
    _, hist = rl.train(cfg['method'], env, N_EPISODES, alpha=cfg['alpha'],
                       epsilon=cfg['epsilon'], seed=cfg['seed'],
                       eval_every=N_EPISODES, evaluator=ev)
    row = dict(cfg, **hist[-1], wall_s=time.perf_counter() - t0)
    print(f"{cfg['method']:>15} a={cfg['alpha']:<5g} e={cfg['epsilon']:<5g} s{cfg['seed']}: "
          f"P(win) {row['p_win']:.4f} ({row['wall_s']:.0f} s)", flush=True)
    return row


def run_schedule(cfg):
    game = Game2048(**BOARD)
    ev = rl.ExactEvaluator.cached(game, CACHE)
    env = Env2048(game, reward='score')
    alpha = None if cfg['method'] == 'mc_control' else rl.linear(0.2, 0.02, N_EPISODES)
    _, hist = rl.train(cfg['method'], env, N_EPISODES, alpha=alpha,
                       epsilon=rl.linear(0.3, 0.01, N_EPISODES), n=3,
                       seed=cfg['seed'], eval_every=2_000, evaluator=ev)
    path = OUT / f"schedule_{cfg['method']}_s{cfg['seed']}.csv"
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    print(f"schedule {cfg['method']:>15} s{cfg['seed']}: P(win) {hist[-1]['p_win']:.4f} "
          f"score {hist[-1]['score']:.1f}", flush=True)
    return dict(cfg, **hist[-1])


def plot_grid(rows):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for ax, method in zip(axes, ('q_learning', 'expected_sarsa')):
        m = np.full((len(ALPHAS), len(EPSILONS)), np.nan)
        for i, a in enumerate(ALPHAS):
            for j, e in enumerate(EPSILONS):
                v = [float(r['p_win']) for r in rows
                     if r['method'] == method and float(r['alpha']) == a
                     and float(r['epsilon']) == e]
                if v:
                    m[i, j] = np.mean(v)
        im = ax.imshow(m, vmin=0, vmax=1, cmap='viridis', origin='lower')
        for i in range(len(ALPHAS)):
            for j in range(len(EPSILONS)):
                if not np.isnan(m[i, j]):
                    ax.text(j, i, f'{m[i, j]:.2f}', ha='center', va='center',
                            fontsize=8, color='w' if m[i, j] < 0.6 else 'k')
        ax.set_xticks(range(len(EPSILONS)))
        ax.set_xticklabels(EPSILONS)
        ax.set_yticks(range(len(ALPHAS)))
        ax.set_yticklabels(ALPHAS)
        ax.set_xlabel(r'$\epsilon$')
        ax.set_title(method.replace('_', ' '))
    axes[0].set_ylabel(r'$\alpha$')
    fig.colorbar(im, ax=axes, label='exact P(win) after 200k episodes', shrink=0.8)
    savefig(fig, OUT / 'fig_tab_grid.png')


def main():
    what = sys.argv[1:] or ['grid', 'schedule']
    game = Game2048(**BOARD)
    rl.ExactEvaluator.cached(game, CACHE)
    if 'grid' in what:
        configs = [dict(method=m, alpha=a, epsilon=e, seed=s)
                   for m in ('q_learning', 'expected_sarsa')
                   for a in ALPHAS for e in EPSILONS for s in (0, 1)]
        rows = run_grid(run_point, configs)
        save_csv(rows, OUT / 'grid.csv')
        plot_grid(rows)
    if 'schedule' in what:
        configs = [dict(method=m, seed=s) for m in rl.METHODS for s in range(5)]
        rows = run_grid(run_schedule, configs)
        save_csv(rows, OUT / 'schedule_summary.csv')


if __name__ == '__main__':
    main()
