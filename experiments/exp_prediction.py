"""Prediction on the 2x3 board (target 64): TD(0), n-step TD and Monte Carlo
estimate the win probability of the optimal policy, against the exact
values of the DP (Sutton and Barto, Sec. 6.1-6.3, 7.1; the same study as
the random-walk examples of the course, Example 6.2 and Figure 7.2).

Every run simulates its own N_EPISODES episodes with the optimal policy
and all the (n, alpha) pairs learn from the same episodes. The error is
sqrt(VE): the RMS error weighted by the on-policy distribution of the
policy (Sec. 9.2). Outputs in results/prediction/:
    curves.csv              mean error vs episodes for the curve methods
    sweep.csv               mean error over the first episodes vs alpha, n
    fig_pred_curves.png     error vs episodes (TD(0), 4-step TD, MC)
    fig_pred_nstep.png      average error vs alpha, one line per n
    summary.json            numbers quoted in the README
Run from the project root:  python experiments/exp_prediction.py [quick]
"""
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from common import out_dir, save_csv, save_json, savefig

import dynamic_programming as dp
import tabular_prediction as pr
from game import Game2048

OUT = out_dir('prediction')
BOARD = dict(rows=2, cols=3, target=64)
V0 = 0.5                      # initial estimate of every state, as in Example 6.2
N_EPISODES = 20_000
N_RUNS = 50
WINDOWS = (1_000, 5_000, 20_000)   # average error over the first episodes
NS = (1, 2, 4, 8, 16, pr.MC_N)
ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0)
CURVES = [(1, 0.1), (1, 0.2), (1, 0.4), (4, 0.1), (4, 0.2),
          (pr.MC_N, 0.02), (pr.MC_N, 0.05), (pr.MC_N, 0.1)]


def n_label(n):
    return 'MC' if n == pr.MC_N else (f'TD(0)' if n == 1 else f'{n}-step TD')


def main():
    n_episodes, n_runs = N_EPISODES, N_RUNS
    if len(sys.argv) > 1:  # quick test
        n_episodes, n_runs = 2_000, 3
    windows = [w for w in WINDOWS if w <= n_episodes]
    game = Game2048(**BOARD)
    V, table, layers = dp.solve_optimal(game)
    prob = pr.Problem(game, table, layers=layers)
    print(f'{len(prob.boards)} states, {(prob.mu > 0).sum()} visited by the '
          f'optimal policy, {prob.expected_length:.2f} moves per game', flush=True)

    pairs = sorted({(n, a) for n in NS for a in ALPHAS} | set(CURVES))
    curve_sum = {c: np.zeros(n_episodes + 1) for c in CURVES}
    avg = {(n, a): {w: [] for w in windows} for n, a in pairs}
    final = {(n, a): [] for n, a in pairs}
    t0 = time.perf_counter()
    for run in range(n_runs):
        S, R, starts = prob.episodes(n_episodes, seed=1000 + run)
        for n, a in pairs:
            _, rms = pr.nstep_td(S, R, starts, n, a, V0, prob.v_true, prob.mu)
            for w in windows:
                avg[(n, a)][w].append(rms[1:w + 1].mean())
            final[(n, a)].append(rms[-1])
            if (n, a) in curve_sum:
                curve_sum[(n, a)] += rms
        print(f'run {run + 1}/{n_runs} ({time.perf_counter() - t0:.0f} s)', flush=True)

    # error vs episodes
    episodes = np.arange(n_episodes + 1)
    rows = []
    step = max(1, n_episodes // 400)
    for (n, a), s in curve_sum.items():
        for e in range(0, n_episodes + 1, step):
            rows.append({'method': n_label(n), 'n': n, 'alpha': a,
                         'episode': e, 'rms': s[e] / n_runs})
    save_csv(rows, OUT / 'curves.csv')
    fig, ax = plt.subplots()
    styles = {1: '-', 4: '-.', pr.MC_N: '--'}
    for (n, a), s in curve_sum.items():
        ax.plot(episodes, s / n_runs, ls=styles[n], label=f'{n_label(n)}, α={a:g}')
    ax.set_xscale('symlog', linthresh=100)
    ax.set_xlabel('episodes')
    ax.set_ylabel(r'$\sqrt{\overline{VE}}$ (weighted RMS error)')
    ax.set_title(f'Prediction of the optimal policy, 2x3 board, target 64 '
                 f'(mean of {n_runs} runs)')
    ax.legend(fontsize=7, ncol=2)
    savefig(fig, OUT / 'fig_pred_curves.png')
    plt.close(fig)

    # average error over the first episodes vs alpha, one line per n
    rows = []
    for n, a in pairs:
        row = {'method': n_label(n), 'n': n, 'alpha': a,
               'final_rms': float(np.mean(final[(n, a)]))}
        for w in windows:
            row[f'avg_rms_{w}'] = float(np.mean(avg[(n, a)][w]))
            row[f'avg_rms_{w}_std'] = float(np.std(avg[(n, a)][w]))
        rows.append(row)
    save_csv(rows, OUT / 'sweep.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for ax, w in zip(axes, (windows[1] if len(windows) > 1 else windows[0], windows[-1])):
        for n in NS:
            ys = [np.mean(avg[(n, a)][w]) for a in ALPHAS]
            ax.plot(ALPHAS, ys, marker='o', ms=3, label=n_label(n))
        ax.set_xlabel(r'step size $\alpha$')
        ax.set_ylabel(f'mean error over the first {w:,} episodes')
        ax.set_xscale('log')
        ax.set_title(f'first {w:,} episodes')
        ax.legend(fontsize=8)
    fig.suptitle(f'n-step TD prediction, 2x3 board, target 64 (mean of {n_runs} runs)')
    savefig(fig, OUT / 'fig_pred_nstep.png')
    plt.close(fig)

    # summary: best alpha for every n and window
    summary = {'states': len(prob.boards), 'visited': int((prob.mu > 0).sum()),
               'expected_length': prob.expected_length, 'v0': V0,
               'n_runs': n_runs, 'n_episodes': n_episodes, 'best': {}}
    for w in windows:
        for n in NS:
            best_a = min(ALPHAS, key=lambda a: np.mean(avg[(n, a)][w]))
            summary['best'][f'{n_label(n)}@{w}'] = {
                'alpha': best_a, 'avg_rms': float(np.mean(avg[(n, best_a)][w])),
                'final_rms': float(np.mean(final[(n, best_a)]))}
    save_json(summary, OUT / 'summary.json')


if __name__ == '__main__':
    main()
