"""Figures and table of the small-board n-tuple experiment.

Reads results/ntuple_small/<board>/exact_all.csv and benchmarks.json and
writes, per board,
    fig_nt_<board>_pwin.png    exact P(win) of the greedy policy vs games
    fig_nt_<board>_rmse.png    weighted RMSE of the values vs V* (win reward)
and results/ntuple_small/table_ntuple_small.csv (last checkpoint of
every configuration, mean over seeds, with the benchmark policies).
Run from the project root:  python experiments/plot_ntuple_small.py
"""
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, save_csv, savefig

OUT = RESULTS / 'ntuple_small'


def label(cfg):
    tuples, sym, reward, init, eps = cfg
    s = {'full': 'full board (tabular)', 'rcs': 'rows+cols+squares',
         'rect': '2x3 rectangles'}[tuples]
    s += ', sym' if sym == 'True' else ''
    s += f', {reward} reward'
    if float(init):
        s += f', init {float(init):g}'
    if float(eps):
        s += f', eps {float(eps):g}'
    return s


def main():
    table = []
    for board_dir in sorted(OUT.iterdir()):
        path = board_dir / 'exact_all.csv'
        if not path.exists():
            continue
        board = board_dir.name
        bench = json.loads((board_dir / 'benchmarks.json').read_text())
        with open(path) as f:
            rows = list(csv.DictReader(f))
        groups = defaultdict(lambda: defaultdict(list))  # cfg -> games -> rows
        for r in rows:
            cfg = (r['tuples'], r['symmetric'], r['reward'], r['init'], r['eps'])
            groups[cfg][int(r['games'])].append(r)
        for key, ylabel, fname in (('p_win', 'exact P(win) of the greedy policy', 'pwin'),
                                   ('rmse_w', 'weighted RMSE of the values vs $V^*$', 'rmse')):
            fig, ax = plt.subplots()
            for cfg, by_games in groups.items():
                games = sorted(by_games)
                vals = [[float(r[key]) for r in by_games[g] if r.get(key)] for g in games]
                if not any(vals):
                    continue
                mean = [np.mean(v) if v else np.nan for v in vals]
                lo = [np.min(v) if v else np.nan for v in vals]
                hi = [np.max(v) if v else np.nan for v in vals]
                line, = ax.plot(games, mean, marker='o', ms=3, label=label(cfg))
                ax.fill_between(games, lo, hi, color=line.get_color(), alpha=0.2)
            ax.set_xscale('log')
            if key == 'p_win':
                ax.axhline(bench['optimal'], ls='--', c='k', lw=1, label='optimal (DP)')
                ax.axhline(bench['corner'], ls=':', c='gray', lw=1, label='corner rule')
                ax.set_ylim(0, 1.02)
            else:
                ax.set_yscale('log')
            ax.set_xlabel('training games')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{board.replace("_", " board, target ")}')
            ax.legend(fontsize=7)
            savefig(fig, board_dir / f'fig_nt_{board}_{fname}.png')
            plt.close(fig)
        for cfg, by_games in groups.items():
            last = by_games[max(by_games)]
            row = {'board': board, 'tuples': cfg[0], 'symmetric': cfg[1],
                   'reward': cfg[2], 'init': cfg[3], 'eps': cfg[4],
                   'games': max(by_games), 'seeds': len(last)}
            for k in ('p_win', 'agree', 'rmse_w', 'unseen', 'score'):
                v = [float(r[k]) for r in last if r.get(k)]
                row[k] = float(np.mean(v)) if v else ''
            table.append(row)
        for name, value in bench.items():
            table.append({'board': board, 'tuples': name, 'p_win': value})
    save_csv(table, OUT / 'table_ntuple_small.csv',
             columns=['board', 'tuples', 'symmetric', 'reward', 'init', 'eps',
                      'games', 'seeds', 'p_win', 'agree', 'rmse_w', 'unseen', 'score'])


if __name__ == '__main__':
    main()
