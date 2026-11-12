"""Figure of the reward study: Sarsa on afterstates on the 2x3 board
(target 64) trained with the naive reward, the score and the win, exact
win probability of the greedy policy against the episodes.

Inputs:  results/reward/afterstate_sarsa_naive_q0_s*.csv
         results/tabular/afterstate_sarsa_{score,win}_q0_s*.csv
Output:  results/reward/fig_reward_learning.{png,pdf}
         results/reward/table_reward_learning.csv
Run from the project root:  python experiments/plot_reward.py
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from common import RESULTS, save_csv, savefig

OUT = RESULTS / 'reward'
RUNS = {  # label -> glob
    'naive reward': (OUT, 'afterstate_sarsa_naive_q0_s*.csv'),
    'score': (RESULTS / 'tabular', 'afterstate_sarsa_score_q0_s*.csv'),
    'win': (RESULTS / 'tabular', 'afterstate_sarsa_win_q0_s*.csv'),
}
CHECKPOINTS = (2_000, 10_000, 20_000, 50_000, 100_000, 200_000)
OPTIMAL = 0.9393


def load(folder, pattern, key='p_win'):
    runs = []
    for path in sorted(folder.glob(pattern)):
        with open(path) as f:
            rows = list(csv.DictReader(f))
        runs.append(rows)
    x = np.array([int(r['episode']) for r in runs[0]])
    y = np.array([[float(r[key]) for r in rows] for rows in runs])
    return x, y


def main():
    fig, ax = plt.subplots()
    table = []
    for label, (folder, pattern) in RUNS.items():
        x, y = load(folder, pattern)
        _, s = load(folder, pattern, 'score')
        line, = ax.plot(x, y.mean(0), label=label)
        ax.fill_between(x, y.min(0), y.max(0), color=line.get_color(), alpha=0.2)
        for c in CHECKPOINTS:
            i = int(np.searchsorted(x, c))
            table.append({'reward': label, 'episode': c, 'seeds': len(y),
                          'p_win': float(y[:, i].mean()), 'p_win_min': float(y[:, i].min()),
                          'p_win_max': float(y[:, i].max()), 'score': float(s[:, i].mean())})
    ax.axhline(OPTIMAL, ls='--', color='k', lw=1, label='optimal')
    ax.set_xlabel('episodes')
    ax.set_ylabel('exact P(win) of the greedy policy')
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.set_title('Sarsa on afterstates, 2x3 board, target 64')
    savefig(fig, OUT / 'fig_reward_learning.png')
    save_csv(table, OUT / 'table_reward_learning.csv')
    for r in table:
        print(f"{r['reward']:>13} @ {r['episode']:>7}: P(win) {r['p_win']:.3f} "
              f"[{r['p_win_min']:.3f}, {r['p_win_max']:.3f}]  score {r['score']:.0f}")


if __name__ == '__main__':
    main()
