"""Figures and table of the tabular experiment (results/tabular/*.csv).

Outputs in results/tabular/:
    fig_tab_pwin_<setting>.png   exact P(win) of the greedy policy vs
                                 episodes, one line per method, band =
                                 min/max over seeds
    fig_tab_rmse_<setting>.png   weighted RMSE of the learned values
    fig_tab_unseen_<setting>.png share of optimal-policy visits to
                                 boards never seen by the agent
    table_tabular_final.csv      final exact quantities (mean over seeds)
Run from the project root:  python experiments/plot_tabular.py
"""
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, save_csv, savefig

import dp2048 as dp
import rl2048 as rl
from game2048 import Game2048

OUT = RESULTS / 'tabular'
LABELS = {'mc_control': 'MC control', 'sarsa': 'Sarsa', 'q_learning': 'Q-learning',
          'expected_sarsa': 'Expected Sarsa', 'nstep_sarsa': '3-step Sarsa',
          'td_afterstate': 'TD(0) afterstates'}
SETTINGS = {'score_q0': 'score reward', 'win_q0': 'win reward',
            'win_q1': 'win reward, optimistic initialisation'}


def load_runs():
    runs = defaultdict(list)  # (method, setting) -> list of (episodes, rows)
    for path in sorted(OUT.glob('*_s*.csv')):
        name = path.stem  # method_reward_qX_sN
        parts = name.rsplit('_', 3)
        method, reward, q, _ = parts
        with open(path) as f:
            rows = list(csv.DictReader(f))
        runs[(method, f'{reward}_{q}')].append(rows)
    return runs


def series(rows_list, key):
    x = np.array([int(r['episode']) for r in rows_list[0]])
    y = np.array([[float(r[key]) for r in rows] for rows in rows_list])
    return x, y


def main():
    game = Game2048(2, 3, target=64)
    V, _, layers = dp.solve_optimal(game)
    bench = {'optimal': dp.start_value(game, V)}
    for name, pol in dp.HEURISTICS.items():
        bench[name] = dp.start_value(game, dp.evaluate_policy(game, pol, layers=layers))
    runs = load_runs()
    final = []
    for setting, title in SETTINGS.items():
        for key, ylabel, fname in (('p_win', 'exact P(win) of the greedy policy', 'pwin'),
                                   ('rmse_w', 'weighted RMSE of the learned values', 'rmse'),
                                   ('unseen', 'share of visits to unseen boards', 'unseen')):
            fig, ax = plt.subplots()
            for method in rl.METHODS:
                if (method, setting) not in runs:
                    continue
                x, y = series(runs[(method, setting)], key)
                line, = ax.plot(x, y.mean(0), label=LABELS[method])
                ax.fill_between(x, y.min(0), y.max(0), color=line.get_color(), alpha=0.2)
            if key == 'p_win':
                ax.axhline(bench['optimal'], ls='--', c='k', lw=1, label='optimal (DP)')
                ax.set_ylim(0, 1)
            if key == 'rmse_w':
                ax.set_yscale('log')
            ax.set_xlabel('episodes')
            ax.set_ylabel(ylabel)
            ax.set_title(f'2x3 board, target 64: {title}')
            ax.legend(fontsize=8)
            savefig(fig, OUT / f'fig_tab_{fname}_{setting}.png')
            plt.close(fig)
        for method in rl.METHODS:
            if (method, setting) not in runs:
                continue
            lasts = [rows[-1] for rows in runs[(method, setting)]]
            row = {'setting': setting, 'method': method, 'seeds': len(lasts)}
            for k in ('p_win', 'score', 'agree', 'unseen', 'rmse_w', 'states', 'time_s'):
                vals = [float(r[k]) for r in lasts if r.get(k)]
                row[k] = float(np.mean(vals)) if vals else ''
                row[k + '_std'] = float(np.std(vals)) if vals else ''
            final.append(row)
    for name, value in bench.items():
        final.append({'setting': 'exact', 'method': name, 'p_win': value})
    save_csv(final, OUT / 'table_tabular_final.csv',
             columns=['setting', 'method', 'seeds', 'p_win', 'p_win_std', 'score',
                      'score_std', 'agree', 'agree_std', 'unseen', 'unseen_std',
                      'rmse_w', 'rmse_w_std', 'states', 'states_std', 'time_s',
                      'time_s_std'])


if __name__ == '__main__':
    main()
