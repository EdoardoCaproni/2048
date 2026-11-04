"""Figures and table of the 4x4 experiment.

Reads results/4x4/*/train.csv, results/4x4/eval_all.csv and
results/4x4/benchmarks.csv and writes in results/4x4/
    fig_4x4_score.png     mean evaluation score vs training games
    fig_4x4_winrate.png   P(2048) and P(4096) vs training games
    fig_4x4_maxtile.png   max-tile distribution of the best run vs heuristics
    fig_4x4_train.png     training score per block (learning curve)
    table_4x4_final.csv   final evaluation of every run and the heuristics
Run from the project root:  python experiments/plot_4x4.py
"""
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, save_csv, savefig

OUT = RESULTS / '4x4'
TILES = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    evals = read(OUT / 'eval_all.csv')
    bench = read(OUT / 'benchmarks.csv')
    by_run = defaultdict(list)
    for r in evals:
        by_run[r['run']].append(r)
    # group the seeds of the same (tuples, alpha)
    groups = defaultdict(list)
    for run, rows in by_run.items():
        groups[run.rsplit('_s', 1)[0]].append(sorted(rows, key=lambda r: int(r['trained_games'])))

    for key, ylabel, fname, logy in (('mean_score', 'mean score (1000 evaluation games)', 'score', True),
                                     ('p_2048', 'P(2048 reached)', 'winrate', False)):
        fig, ax = plt.subplots()
        for name, runs in sorted(groups.items()):
            x = [int(r['trained_games']) for r in runs[0]]
            y = np.array([[float(r[key]) for r in rows] for rows in runs])
            line, = ax.plot(x, y.mean(0), marker='o', ms=3, label=name)
            ax.fill_between(x, y.min(0), y.max(0), color=line.get_color(), alpha=0.2)
            if key == 'p_2048':
                y4 = np.array([[float(r['p_ge_4096']) for r in rows] for rows in runs])
                ax.plot(x, y4.mean(0), ls='--', color=line.get_color(), lw=1)
        for b in bench:
            ax.axhline(float(b[key]), ls=':', lw=1, c='gray')
            ax.annotate(b['policy'], (x[0], float(b[key])), fontsize=7, color='gray')
        ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')
        ax.set_xlabel('training games')
        ax.set_ylabel(ylabel)
        ax.set_title('4x4 board: TD(0) n-tuple agent' +
                     (' (dashed: P(4096))' if key == 'p_2048' else ''))
        ax.legend(fontsize=7)
        savefig(fig, OUT / f'fig_4x4_{fname}.png')
        plt.close(fig)

    # training curves (score per block of 1000 games)
    fig, ax = plt.subplots()
    for name, runs in sorted(groups.items()):
        for rows in runs:
            run = rows[0]['run']
            tr = read(OUT / run / 'train.csv')
            x = [int(r['games']) for r in tr]
            y = [float(r['mean_score']) for r in tr]
            ax.plot(x, y, lw=0.8, label=run)
    ax.set_xlabel('training games')
    ax.set_ylabel('mean score of the block (learning on)')
    ax.set_title('4x4 board: training curves')
    ax.legend(fontsize=6, ncol=2)
    savefig(fig, OUT / 'fig_4x4_train.png')
    plt.close(fig)

    # final table and max-tile distribution of the best run
    finals = [rows[-1] for rows in by_run.values()]
    finals.sort(key=lambda r: -float(r['mean_score']))
    cols = ['run', 'trained_games', 'games', 'mean_score', 'score_ci95', 'median_score',
            'mean_moves', 'p_2048', 'p_2048_ci95'] + [f'p_ge_{t}' for t in TILES]
    table = [{c: r.get(c, '') for c in cols} for r in finals]
    table += [dict({c: '' for c in cols}, run=b['policy'],
                   **{c: b[c] for c in cols if c in b}) for b in bench]
    save_csv(table, OUT / 'table_4x4_final.csv', columns=cols)

    best = finals[0]
    fig, ax = plt.subplots()
    width = 0.2
    series = [(best['run'], best)] + [(b['policy'], b) for b in bench]
    for i, (name, r) in enumerate(series):
        # P(max tile = t) = P(>= t) - P(>= 2t)
        p = [float(r.get(f'p_ge_{t}', 0)) - float(r.get(f'p_ge_{2 * t}', 0)) for t in TILES]
        ax.bar(np.arange(len(TILES)) + (i - 1.5) * width, p, width, label=name)
    ax.set_xticks(range(len(TILES)))
    ax.set_xticklabels(TILES)
    ax.set_xlabel('largest tile at the end of the game')
    ax.set_ylabel('share of games')
    ax.set_title('4x4 board: largest tile reached')
    ax.legend(fontsize=8)
    savefig(fig, OUT / 'fig_4x4_maxtile.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
