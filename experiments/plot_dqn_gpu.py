"""Figures and table of the DQN final test: the tutorial's schedule
(results/dqn/) against many games on the GPU (results/dqn_gpu/), with the
perceptron and the convolutional network. Outputs in results/dqn_gpu/:
    fig_dqn_gpu_2x3.png    2x3 board, target 64: exact P(win) of the greedy
                           policy against the training games, with the tabular
                           Q-learning and Sarsa on afterstates (score reward)
    fig_dqn_gpu_4x4.png    4x4 board: mean score of the greedy policy against
                           the training games, and the tiles reached at the end,
                           with the rules of thumb
    table_dqn_gpu.csv      final numbers, mean over the seeds
Run from the project root:  python experiments/plot_dqn_gpu.py
"""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, save_csv, savefig

OUT = RESULTS / 'dqn_gpu'
TILES = (256, 512, 1024, 2048, 4096)


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def runs(folder, prefix):
    out = []
    for d in sorted(folder.glob(f'{prefix}_s*')):
        if (d / 'train.csv').exists() and (d / 'final.json').exists():
            out.append((read(d / 'train.csv'), json.loads((d / 'final.json').read_text())))
    return out


def band(ax, x, ys, label, **kw):
    n = min(len(y) for y in ys)
    x = np.array(x[:n], dtype=float)
    y = np.array([yy[:n] for yy in ys], dtype=float)
    line, = ax.plot(x, y.mean(0), label=label, **kw)
    if len(ys) > 1:
        ax.fill_between(x, y.min(0), y.max(0), color=line.get_color(), alpha=0.2)


def games(rows):
    return [int(r.get('games') or r.get('episode')) for r in rows]


def main():
    table = []
    # ---------------- 2x3: exact P(win) ----------------
    tut, gpu = runs(RESULTS / 'dqn', '2x3'), runs(OUT, '2x3')
    fig, ax = plt.subplots(figsize=(7, 4.2))
    if gpu:
        band(ax, games(gpu[0][0]), [[float(r['p_win']) for r in rows] for rows, _ in gpu],
             'DQN, many games on the GPU', lw=2, marker='o', ms=3)
    if tut:
        band(ax, games(tut[0][0]), [[float(r['p_win']) for r in rows] for rows, _ in tut],
             'DQN, schedule of the tutorial', lw=2, ls='--')
    for method, label in (('q_learning', 'Q-learning, tabular'),
                          ('afterstate_sarsa', 'Sarsa on afterstates, tabular')):
        rs = [read(p) for p in sorted((RESULTS / 'tabular').glob(f'{method}_score_q0_s*.csv'))]
        if rs:
            band(ax, [int(r['episode']) for r in rs[0]], [[float(r['p_win']) for r in rows] for rows in rs],
                 label, ls=':')
    ax.axhline(0.9393, c='k', lw=1, ls='--', label='optimal policy (DP)')
    ax.set_xscale('log')
    ax.set_ylim(0, 1)
    ax.set_xlabel('training games')
    ax.set_ylabel('exact P(win) of the greedy policy')
    ax.set_title('2x3 board, target 64, score reward (band = min-max over the seeds)')
    ax.legend(fontsize=8, loc='lower right')
    savefig(fig, OUT / 'fig_dqn_gpu_2x3.png')
    plt.close(fig)
    for name, rs in (('tutorial', tut), ('GPU, perceptron', gpu)):
        if rs:
            last = [rows[-1] for rows, _ in rs]
            table.append({'board': '2x3', 'agent': name, 'seeds': len(rs), 'games': games(last)[0],
                          'p_win': np.mean([float(r['p_win']) for r in last]),
                          'p_win_min': min(float(r['p_win']) for r in last),
                          'p_win_max': max(float(r['p_win']) for r in last),
                          'score': np.mean([float(r['score']) for r in last])})

    # ---------------- 4x4: score and tiles ----------------
    tut4, mlp4, cnn4 = runs(RESULTS / 'dqn', '4x4'), runs(OUT, '4x4'), runs(OUT, 'cnn_4x4')
    bench = {r['policy']: r for r in read(RESULTS / '4x4' / 'benchmarks.csv')}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    color = {'tutorial': 'tab:green', 'perceptron': 'tab:orange', 'cnn': 'tab:blue', 'rule': 'tab:gray'}
    for rs, label, kw in ((cnn4, 'DQN on the GPU, convolutional network', dict(lw=2, marker='s', ms=3, color=color['cnn'])),
                          (mlp4, 'DQN on the GPU, perceptron', dict(lw=2, marker='o', ms=3, color=color['perceptron'])),
                          (tut4, 'DQN, schedule of the tutorial', dict(lw=2, ls='--', color=color['tutorial']))):
        if rs:
            band(ax, games(rs[0][0]), [[float(r['eval_score']) for r in rows] for rows, _ in rs], label, **kw)
    for name, ls in (('greedy', ':'), ('corner', '-.')):
        ax.axhline(float(bench[name]['mean_score']), c='gray', ls=ls, lw=1, label=f'{name} rule')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('training games')
    ax.set_ylabel('mean score of the greedy policy (200 games)')
    ax.set_title('4x4 board: learning curves (band = min-max over the seeds)')
    ax.legend(fontsize=7)
    ax = axes[1]
    series = []
    for rs, label, c in ((tut4, 'DQN, tutorial, 10k games', color['tutorial']),
                         (mlp4, 'DQN GPU, perceptron, 100k', color['perceptron']),
                         (cnn4, 'DQN GPU, convolutional, 100k', color['cnn'])):
        if rs:
            series.append((label, c, [np.mean([f.get(f'final_p{t}', 0.0) for _, f in rs]) for t in TILES]))
    series.append(('greedy rule', color['rule'], [float(bench['greedy'][f'p_ge_{t}']) for t in TILES]))
    width = 0.8 / len(series)
    for i, (label, c, p) in enumerate(series):
        ax.bar(np.arange(len(TILES)) + (i - (len(series) - 1) / 2) * width, p, width, label=label, color=c)
    ax.set_xticks(np.arange(len(TILES)))
    ax.set_xticklabels(TILES)
    ax.set_xlabel('tile reached at least once')
    ax.set_ylabel('share of the final evaluation games')
    ax.set_title('4x4 board: tiles reached (1 000 games; rule: 10 000)')
    ax.legend(fontsize=7)
    savefig(fig, OUT / 'fig_dqn_gpu_4x4.png')
    plt.close(fig)
    for name, rs in (('tutorial', tut4), ('GPU, perceptron', mlp4), ('GPU, convolutional', cnn4)):
        finals = [f for _, f in rs]
        if finals:
            row = {'board': '4x4', 'agent': name, 'seeds': len(finals),
                   'games': int(finals[0].get('games') or finals[0].get('episode')),
                   'final_score': np.mean([f['final_score'] for f in finals]),
                   'final_score_min': min(f['final_score'] for f in finals),
                   'final_score_max': max(f['final_score'] for f in finals),
                   'final_moves': np.mean([f['final_moves'] for f in finals])}
            for t in TILES:
                row[f'final_p{t}'] = np.mean([f.get(f'final_p{t}', 0.0) for f in finals])
            table.append(row)
    save_csv(table, OUT / 'table_dqn_gpu.csv', columns=sorted({k for r in table for k in r}))


if __name__ == '__main__':
    main()
