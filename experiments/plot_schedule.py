"""Learning curves of the tabular methods with decaying alpha and epsilon
(results/tabular_grid/schedule_*.csv from exp_tabular_grid.py), compared
with the constant-parameter runs of exp_tabular_2x3.py (score reward).

Writes results/tabular_grid/fig_tab_pwin_schedule.png and
table_schedule_final.csv. Run from the project root:
    python experiments/plot_schedule.py
"""
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, save_csv, savefig

import dp2048 as dp
import rl2048 as rl
from game2048 import Game2048
from plot_tabular import LABELS, load_runs, series

OUT = RESULTS / 'tabular_grid'


def main():
    game = Game2048(2, 3, target=64)
    V, _, _ = dp.solve_optimal(game)
    optimal = dp.start_value(game, V)
    runs = defaultdict(list)
    for path in sorted(OUT.glob('schedule_*_s*.csv')):
        method = path.stem[len('schedule_'):].rsplit('_s', 1)[0]
        with open(path) as f:
            runs[method].append(list(csv.DictReader(f)))
    constant = load_runs()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (title, source) in zip(axes, (('constant $\\alpha=0.1$, $\\epsilon=0.1$',
                                           {m: constant[(m, 'score_q0')] for m in rl.METHODS
                                            if (m, 'score_q0') in constant}),
                                          ('$\\alpha$: 0.2 to 0.02, $\\epsilon$: 0.3 to 0.01',
                                           runs))):
        for method in rl.METHODS:
            if method not in source:
                continue
            x, y = series(source[method], 'p_win')
            line, = ax.plot(x, y.mean(0), label=LABELS[method])
            ax.fill_between(x, y.min(0), y.max(0), color=line.get_color(), alpha=0.2)
        ax.axhline(optimal, ls='--', c='k', lw=1, label='optimal (DP)')
        ax.set_xlabel('episodes')
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel('exact P(win) of the greedy policy')
    axes[1].legend(fontsize=8, loc='lower right')
    fig.suptitle('2x3 board, target 64, score reward: constant versus decaying parameters')
    savefig(fig, OUT / 'fig_tab_pwin_schedule.png')
    plt.close(fig)
    table = []
    for method in rl.METHODS:
        if method not in runs:
            continue
        lasts = [r[-1] for r in runs[method]]
        row = {'method': method, 'seeds': len(lasts)}
        for k in ('p_win', 'score', 'agree', 'unseen', 'rmse_w'):
            vals = [float(r[k]) for r in lasts if r.get(k)]
            row[k] = float(np.mean(vals)) if vals else ''
            row[k + '_std'] = float(np.std(vals)) if vals else ''
        table.append(row)
    save_csv(table, OUT / 'table_schedule_final.csv')
    for row in table:
        print(f"{row['method']:>15}: P(win) {row['p_win']:.4f} +- {row['p_win_std']:.4f}  "
              f"score {row['score']:.1f}  agree {row['agree']:.3f}  rmse_w {row['rmse_w']:.3f}")


if __name__ == '__main__':
    main()
