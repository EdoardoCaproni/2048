"""Choosing the reward: the rules-of-thumb reward against the score and the win.

The "naive" reward of shaped_rewards.py pays a bonus for every new largest
tile, the growth of the tile sum and the change of the number of free
cells. Two studies on the 2x3 board with target 64:

dp       the optimal policy of every reward, computed exactly, and its
         exact P(win), P(reaching 128), expected score, expected number of
         moves and agreement with the win-optimal policy; check that the
         free-cell term changes no decision.
         -> results/reward/table_reward_dp.csv
tabular  Sarsa on afterstates trained with the naive reward (same setting
         as the 'score_q0' runs of exp_tabular_2x3.py: alpha = epsilon =
         0.1, zero initial values, 200k episodes, 5 seeds), with exact
         learning curves.
         -> results/reward/afterstate_sarsa_naive_q0_s<seed>.csv

Run from the project root:
    python experiments/exp_reward.py dp
    python experiments/exp_reward.py tabular [quick]
"""
import csv
import sys
import time

from common import RESULTS, out_dir, run_grid, save_csv

import dynamic_programming as dp
import tabular_control as rl
from game import Env2048, Game2048

BOARD = dict(rows=2, cols=3, target=64)
CACHE = RESULTS / 'dp_cache' / 'eval_2x3_64.pkl'
OUT = out_dir('reward')
OBJECTIVES = ('win', 'score', 'naive', 'naive_nofree', 'milestone', 'sum', 'free')
N_EPISODES = 200_000
EVAL_EVERY = 2_000
SEEDS = range(5)


def study_dp():
    game = Game2048(**BOARD)
    t0 = time.perf_counter()
    ev = rl.ExactEvaluator.cached(game, CACHE)
    game128 = Game2048(BOARD['rows'], BOARD['cols'], target=128)
    layers128 = dp.reachable_layers(game128, stop_at_target=True)
    print(f'evaluator ready ({time.perf_counter() - t0:.0f} s)', flush=True)
    per_move = 2 + 2 * game.p4  # the 'sum' reward pays this once per move
    tables, values, rows = {}, {}, []
    _, table_score, _ = dp.solve_optimal(game, 'score', ev.layers_score)
    for obj in OBJECTIVES:
        t0 = time.perf_counter()
        layers = ev.layers_win if obj == 'win' else ev.layers_score
        V, table, _ = dp.solve_optimal(game, obj, layers)
        tables[obj], values[obj] = table, V
        if obj == 'win':
            # the win-optimal policy stops at the target: after it, play on
            # with the score-optimal policy (only the score, the moves and
            # P(128) depend on this choice)
            full = dict(table_score)
            full.update(table)
            pi = dp.table_policy(full)
        else:
            pi = dp.table_policy(table)
        row = {
            'reward': obj,
            'p_win': dp.start_value(game, dp.evaluate_policy(game, pi, 'win', ev.layers_win)),
            'p_128': dp.start_value(game128, dp.evaluate_policy(game128, pi, 'win', layers128)),
            'score': dp.start_value(game, dp.evaluate_policy(game, pi, 'score', ev.layers_score)),
            'moves': dp.start_value(game, dp.evaluate_policy(game, pi, 'sum', ev.layers_score)) / per_move,
            'agree_win_opt': dp.compare_with_heuristic(
                game, ev.V_win, tables.get('win') or dp.solve_optimal(game, 'win', ev.layers_win)[1],
                ev.layers_win, pi, 'win')['share'],
            'own_value': dp.start_value(game, V),
            'time_s': time.perf_counter() - t0,
        }
        rows.append(row)
        print(f"{obj:>13}: P(win) {row['p_win']:.4f}  P(128) {row['p_128']:.4f}  "
              f"score {row['score']:6.1f}  moves {row['moves']:5.2f}  "
              f"agree {row['agree_win_opt']:.3f}  ({row['time_s']:.0f} s)", flush=True)
    save_csv(rows, OUT / 'table_reward_dp.csv')
    # the free-cell term: same decisions, values shifted by -free(x)
    same = sum(tables['naive'][b] == tables['naive_nofree'][b] for b in tables['naive'])
    shift = max(abs(values['naive'][b] - (values['naive_nofree'][b] - b.count(0)))
                for b in values['naive'])
    print(f"naive vs naive_nofree: same move on {same} of {len(tables['naive'])} boards, "
          f"max |V_naive - (V_nofree - free)| = {shift:.2e}")


def run_name(cfg):
    return f"afterstate_sarsa_{cfg['reward']}_q0_s{cfg['seed']}"


def run(cfg):
    game = Game2048(**BOARD)
    ev = rl.ExactEvaluator.cached(game, CACHE)
    ev.add_reference(cfg['reward'])
    env = Env2048(game, reward=cfg['reward'])
    t0 = time.perf_counter()
    _, hist = rl.train('afterstate_sarsa', env, cfg['n_episodes'], alpha=0.1,
                       epsilon=0.1, q_init=0.0, seed=cfg['seed'],
                       eval_every=EVAL_EVERY, evaluator=ev)
    with open(OUT / (run_name(cfg) + '.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0]))
        w.writeheader()
        w.writerows(hist)
    final = dict(cfg, **hist[-1], wall_s=time.perf_counter() - t0)
    print(f"{run_name(cfg):>36}: P(win) {final['p_win']:.4f} score {final['score']:.1f} "
          f"({final['wall_s']:.0f} s)", flush=True)
    return final


def study_tabular(quick=False):
    rl.ExactEvaluator.cached(Game2048(**BOARD), CACHE)
    n = 10_000 if quick else N_EPISODES
    configs = [dict(reward='naive', seed=s, n_episodes=n)
               for s in (SEEDS[:1] if quick else SEEDS)]
    finals = run_grid(run, configs)
    if not quick:
        save_csv(finals, OUT / 'summary_tabular.csv')


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'dp'
    if what == 'dp':
        study_dp()
    elif what == 'tabular':
        study_tabular(quick='quick' in sys.argv[2:])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
