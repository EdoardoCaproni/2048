"""Figures and tables of the DP part, in results/dp/:
    fig_dp_stages.png        reachable states per stage, 2x3 board
    fig_dp_states.png        reachable states per board (log scale)
    fig_dp_length.png        game length under the optimal policy (MC)
    fig_dp_p4.png            optimal P(win) vs p4
    table_dp_boards.csv      states, DP time, P(win) of optimal and heuristics
    table_dp_exceptions.csv  most important departures from the corner rule
    dp_summary.json          the numbers quoted in the text
Run from the project root:  python experiments/dp_figures.py [big]
('big' adds the 3x3 board: about 5 minutes and 1.5 GB)
"""
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

from common import out_dir, save_csv, save_json, savefig

import dynamic_programming as dp
from game import Game2048

OUT = out_dir('dp')
CONFIGS = [(2, 2, 16), (2, 2, 32), (2, 3, 32), (2, 3, 64), (2, 3, 128),
           (2, 4, 64), (2, 4, 128)]


def main():
    summary = {}
    game = Game2048(2, 3, target=64)
    t0 = time.perf_counter()
    layers = dp.reachable_layers(game)
    V, table, _ = dp.solve_optimal(game, layers=layers)
    p_opt = dp.start_value(game, V)
    summary['2x3_64'] = {'states': dp.count_states(layers), 'stages': len(layers),
                         'dp_time_s': time.perf_counter() - t0, 'optimal': p_opt}

    # states per stage
    sigmas = sorted(layers)
    fig, ax = plt.subplots()
    ax.bar(sigmas, [len(layers[s]) for s in sigmas], width=1.6)
    ax.set_xlabel(r'tile sum $\sigma$ (stage)')
    ax.set_ylabel(r'number of states $|X_\sigma|$')
    ax.set_title('Reachable states per stage, 2x3 board, target 64')
    savefig(fig, OUT / 'fig_dp_stages.png')
    plt.close(fig)

    # heuristics on the 2x3 board
    for name, policy in dp.HEURISTICS.items():
        summary['2x3_64'][name] = dp.start_value(
            game, dp.evaluate_policy(game, policy, layers=layers))

    # Monte Carlo check and game length
    n_games = 20_000
    results = dp.simulate(game, dp.table_policy(table), n_games, seed=42)
    wins = np.array([r['won'] for r in results], dtype=float)
    summary['2x3_64']['mc_p_win'] = wins.mean()
    summary['2x3_64']['mc_half_width_95'] = 1.96 * wins.std(ddof=1) / np.sqrt(n_games)
    moves = [r['moves'] for r in results]
    summary['2x3_64']['mc_mean_moves'] = float(np.mean(moves))
    fig, ax = plt.subplots()
    ax.hist(moves, bins=np.arange(min(moves), max(moves) + 2) - 0.5)
    ax.set_xlabel('number of moves')
    ax.set_ylabel('games')
    ax.set_title('Game length under the optimal policy (2x3, target 64)')
    savefig(fig, OUT / 'fig_dp_length.png')
    plt.close(fig)

    # exceptions to the corner rule
    cmp = dp.compare_with_heuristic(game, V, table, layers, dp.corner_policy)
    summary['2x3_64']['corner_share'] = cmp['share']
    summary['2x3_64']['expected_moves'] = cmp['expected_moves']
    rows = []
    for importance, board, corner_move, best_move, loss in cmp['exceptions'][:10]:
        rows.append({'importance': importance, 'board': ' '.join(game.render(board).split('\n')),
                     'corner_move': corner_move, 'optimal_move': best_move,
                     'loss': loss, 'value': V[board]})
    save_csv(rows, OUT / 'table_dp_exceptions.csv')

    # win probability vs expected score
    V_s, table_s, _ = dp.solve_optimal(game, objective='score')
    score_policy = dp.table_policy(table_s)
    summary['2x3_64']['max_score'] = dp.start_value(game, V_s)
    summary['2x3_64']['p_win_of_score_policy'] = dp.start_value(
        game, dp.evaluate_policy(game, score_policy, layers=layers))
    summary['2x3_64']['score_win_agreement'] = dp.compare_with_heuristic(
        game, V, table, layers, score_policy)['share']
    # past the target the win-optimal policy is undefined: play score-optimally
    def win_then_score(board, moves):
        return {table.get(board, table_s.get(board)): 1.0} if moves else {}
    summary['2x3_64']['score_of_win_policy'] = dp.start_value(
        game, dp.evaluate_policy(game, win_then_score, 'score'))

    # sensitivity to p4
    p4_values = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
    p_win = []
    for p4 in p4_values:
        g = Game2048(2, 3, target=64, p4=p4)
        V_p4, _, _ = dp.solve_optimal(g)
        p_win.append(dp.start_value(g, V_p4))
    summary['p4'] = dict(zip(map(str, p4_values), p_win))
    fig, ax = plt.subplots()
    ax.plot(p4_values, p_win, marker='o')
    ax.set_xlabel(r'probability that a new tile is a 4 ($p_4$)')
    ax.set_ylabel('optimal P(win)')
    ax.set_title('2x3 board, target 64')
    savefig(fig, OUT / 'fig_dp_p4.png')
    plt.close(fig)

    # boards comparison
    configs = CONFIGS + ([(3, 3, 64)] if 'big' in sys.argv else [])
    rows = []
    for r, c, target in configs:
        row = dp.compare_policies(Game2048(r, c, target=target))
        row['bound'] = (target.bit_length() - 1) ** (r * c)
        rows.append(row)
        print(f"{row['board']} target {row['target']:>3}: {row['states']:>9,} states, "
              f"DP {row['dp_time_s']:5.1f} s | optimal {row['optimal']:.4f}, "
              f"corner {row['corner']:.4f}, greedy {row['greedy']:.4f}, "
              f"random {row['random']:.4f}", flush=True)
    save_csv(rows, OUT / 'table_dp_boards.csv')
    fig, ax = plt.subplots()
    labels = [f"{row['board']}\n{row['target']}" for row in rows]
    ax.bar(labels, [row['states'] for row in rows])
    ax.set_yscale('log')
    ax.set_ylabel('reachable states (log scale)')
    ax.set_title('State space of small 2048 boards (board, target)')
    savefig(fig, OUT / 'fig_dp_states.png')
    plt.close(fig)
    save_json(summary, OUT / 'dp_summary.json')


if __name__ == '__main__':
    main()
