"""Figure and table of the tabular methods on the 4x4 game
(experiments/exp_tabular_4x4.py). Outputs in results/tabular_4x4/:
    fig_tabular_4x4.png/pdf  epsilon = 0.1: mean score of the greedy policy
                             (200 games, the corner rule on boards not in the
                             table) and share of its moves on boards not in
                             the table, against the training episodes; band =
                             min-max over the 5 seeds
    table_tabular_4x4.csv    final numbers (1 000 games) for epsilon 0.1, 0.2
                             and 0.3, mean over the seeds
Run from the project root:  python experiments/plot_tabular_4x4.py
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, save_csv, savefig

OUT = RESULTS / 'tabular_4x4'
METHODS = ('mc_control', 'sarsa', 'expected_sarsa', 'q_learning', 'nstep_sarsa', 'afterstate_sarsa')
LABELS = {'mc_control': 'MC control', 'sarsa': 'Sarsa', 'q_learning': 'Q-learning',
          'expected_sarsa': 'Expected Sarsa', 'nstep_sarsa': '3-step Sarsa',
          'afterstate_sarsa': 'Sarsa (afterstates)'}
EPS = {0.1: '', 0.2: '_eps0.2', 0.3: '_eps0.3'}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    bench = {r['policy']: float(r['mean_score']) for r in read(RESULTS / '4x4' / 'benchmarks.csv')}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for m in METHODS:
        runs = [[r for r in read(p) if r.get('eval_score')]
                for p in sorted(OUT.glob(f'{m}_s*/train.csv'))]
        x = [int(r['games']) for r in runs[0]]
        for ax, key, scale in ((axes[0], 'eval_score', 1.0), (axes[1], 'eval_unseen', 100.0)):
            y = np.array([[float(r[key]) * scale for r in rows] for rows in runs])
            line, = ax.plot(x, y.mean(0), lw=2, marker='o', ms=3, label=LABELS[m])
            ax.fill_between(x, y.min(0), y.max(0), color=line.get_color(), alpha=0.15)
    for name, ls in (('greedy', ':'), ('corner', '-.')):
        axes[0].axhline(bench[name], c='gray', ls=ls, lw=1.2, label=f'{name} rule')
    axes[0].set_ylabel('mean score of the greedy policy (200 games)')
    axes[0].set_title('4x4 board, tabular methods, epsilon 0.1: score')
    axes[1].set_ylabel('moves on boards not in the table (%)')
    axes[1].set_title('4x4 board: moves played by the corner rule')
    axes[1].set_ylim(0, 100)
    for ax in axes:
        ax.set_xlabel('training episodes')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v / 1000:.0f}k'))
    axes[0].legend(fontsize=7, ncol=2, loc='upper right', bbox_to_anchor=(1.0, 0.93))
    savefig(fig, OUT / 'fig_tabular_4x4.png')
    plt.close(fig)

    table = []
    for eps, suffix in EPS.items():
        rows = read(OUT / f'summary{suffix}.csv')
        for m in METHODS:
            rs = [r for r in rows if r['method'] == m]
            if not rs:
                continue
            f = lambda k: np.array([float(r[k]) for r in rs])
            table.append({'method': m, 'eps': eps, 'seeds': len(rs),
                          'score': f('final_score').mean(), 'score_min': f('final_score').min(),
                          'score_max': f('final_score').max(), 'unseen': f('final_unseen').mean(),
                          'p512': f('final_p512').mean(), 'train_score': f('train_score').mean(),
                          'train_moves': f('train_moves').mean(), 'entries': f('states').mean(),
                          'entries_visited_2': f('entries_visited_2').mean(),
                          'entries_visited_10': f('entries_visited_10').mean()})
    save_csv(table, OUT / 'table_tabular_4x4.csv', columns=list(table[0]))


if __name__ == '__main__':
    main()
