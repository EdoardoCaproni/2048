"""Figures and table of the DQN final test (results/dqn/).

Outputs in results/dqn/:
    fig_dqn_2x3.png     2x3 board, target 64: exact P(win) and expected score
                        of the greedy policy vs training games, DQN against
                        the tabular Q-learning and Sarsa on afterstates
                        (score reward) and the optimum
    fig_dqn_4x4.png     4x4 board: mean score vs training games (DQN against
                        semi-gradient Sarsa with n-tuple features) and the
                        largest tile reached at the end of the evaluation
    table_dqn_final.csv final numbers (mean over seeds)
Run from the project root:  python experiments/plot_dqn.py
"""
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, save_csv, savefig

OUT = RESULTS / 'dqn'
TABULAR = RESULTS / 'tabular'
NTUPLE = RESULTS / '4x4'
TILES = (256, 512, 1024, 2048, 4096)


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def runs(board):
    out = []
    for d in sorted(OUT.glob(f'{board}_s*')):
        if (d / 'train.csv').exists() and (d / 'final.json').exists():
            out.append((read(d / 'train.csv'), json.loads((d / 'final.json').read_text())))
    return out


def band(ax, x, ys, label, **kw):
    ys = np.array(ys)
    line, = ax.plot(x, ys.mean(0), label=label, **kw)
    if len(ys) > 1:
        ax.fill_between(x, ys.min(0), ys.max(0), color=line.get_color(), alpha=0.2)


def main():
    table = []
    # ---------------- 2x3 board, exact evaluation --------------------------
    r23 = runs('2x3')
    if r23:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        x = np.array([int(r['episode']) for r in r23[0][0]])
        horizon = x[-1]
        for ax, key, ylabel in ((axes[0], 'p_win', 'exact P(win) of the greedy policy'),
                                (axes[1], 'score', 'exact expected score of the greedy policy')):
            band(ax, x, [[float(r[key]) for r in rows] for rows, _ in r23], 'DQN', lw=2)
            for method, label in (('q_learning', 'Q-learning (tabular)'),
                                  ('afterstate_sarsa', 'Sarsa on afterstates (tabular)')):
                ys = []
                for path in sorted(TABULAR.glob(f'{method}_score_q0_s*.csv')):
                    rows = [r for r in read(path) if int(r['episode']) <= horizon]
                    ys.append([float(r[key]) for r in rows])
                    xt = [int(r['episode']) for r in rows]
                if ys:
                    band(ax, xt, ys, label, ls='--')
            ax.axhline(0.9393 if key == 'p_win' else 480.26, ls=':', c='k', lw=1,
                       label='optimal (DP)')
            ax.set_xlabel('training games')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        axes[0].set_ylim(0, 1)
        fig.suptitle('Final test: DQN on the 2x3 board, target 64, score reward '
                     f'(mean of {len(r23)} seeds, band = min-max)')
        savefig(fig, OUT / 'fig_dqn_2x3.png')
        plt.close(fig)
        finals = [f for _, f in r23]
        row = {'board': '2x3', 'seeds': len(finals),
               'episodes': int(np.mean([f['episode'] for f in finals])),
               'steps': int(np.mean([f['steps'] for f in finals])),
               'wall_s': float(np.mean([f['wall_s'] for f in finals]))}
        for k in ('p_win', 'score', 'agree'):
            v = [float(f[k]) for f in finals]
            row[k], row[k + '_std'] = float(np.mean(v)), float(np.std(v))
        table.append(row)

    # ---------------- 4x4 board, Monte Carlo evaluation --------------------
    r44 = runs('4x4')
    if r44:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        x = np.array([int(r['episode']) for r in r44[0][0]])
        band(ax, x, [[float(r['eval_score']) for r in rows] for rows, _ in r44],
             'DQN (200 games per point)', lw=2, marker='o', ms=3)
        evals = defaultdict(list)
        for r in read(NTUPLE / 'eval_all.csv'):
            evals[r['run']].append(r)
        for group, label in (('rcs_a0.1', 'Sarsa, n-tuple rows/cols/squares'),
                             ('rect_a0.1', 'Sarsa, n-tuple 2x3 rectangles')):
            seeds = [sorted(rows, key=lambda r: int(r['trained_games']))
                     for run, rows in evals.items() if run.rsplit('_s', 1)[0] == group]
            common = sorted(set.intersection(*[{int(r['trained_games']) for r in s} for s in seeds]))
            ys = [[float(next(r['mean_score'] for r in s if int(r['trained_games']) == g))
                   for g in common] for s in seeds]
            band(ax, common, ys, label, ls='--', marker='.', ms=3)
        bench = {r['policy']: r for r in read(NTUPLE / 'benchmarks.csv')}
        for name, ls in (('greedy', ':'), ('corner', '-.')):
            ax.axhline(float(bench[name]['mean_score']), c='gray', ls=ls, lw=1,
                       label=f'{name} heuristic')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('training games')
        ax.set_ylabel('mean score (greedy play, no learning)')
        ax.set_title('4x4 board: learning curves')
        ax.legend(fontsize=7)

        ax = axes[1]
        finals = [f for _, f in r44]
        width = 0.2
        series = {'DQN (final, 1000 games)': [np.mean([f[f'final_p{t}'] for f in finals]) for t in TILES]}
        best = [r for run, rows in evals.items() if run.startswith('rect_a0.1')
                for r in rows if int(r['trained_games']) == 10_000]
        if best:
            series['Sarsa n-tuple 2x3 rect., 10k games'] = [
                np.mean([float(r[f'p_ge_{t}']) for r in best]) for t in TILES]
        series['greedy heuristic'] = [float(bench['greedy'][f'p_ge_{t}']) for t in TILES]
        for i, (name, p) in enumerate(series.items()):
            ax.bar(np.arange(len(TILES)) + (i - (len(series) - 1) / 2) * width, p, width, label=name)
        ax.set_xticks(np.arange(len(TILES)))
        ax.set_xticklabels(TILES)
        ax.set_xlabel('tile reached at least once')
        ax.set_ylabel('share of games')
        ax.set_title('4x4 board: tiles reached')
        ax.legend(fontsize=7)
        fig.suptitle(f'Final test: DQN on the 4x4 game (mean of {len(r44)} seeds)')
        savefig(fig, OUT / 'fig_dqn_4x4.png')
        plt.close(fig)
        row = {'board': '4x4', 'seeds': len(finals),
               'episodes': int(np.mean([f['episode'] for f in finals])),
               'steps': int(np.mean([f['steps'] for f in finals])),
               'wall_s': float(np.mean([f['wall_s'] for f in finals]))}
        for k in ['final_score', 'final_moves'] + [f'final_p{t}' for t in TILES]:
            v = [float(f[k]) for f in finals]
            row[k], row[k + '_std'] = float(np.mean(v)), float(np.std(v))
        row['final_max_tile'] = max(int(f['final_max_tile']) for f in finals)
        table.append(row)
    if table:
        save_csv(table, OUT / 'table_dqn_final.csv',
                 columns=sorted({k for r in table for k in r}))


if __name__ == '__main__':
    main()
