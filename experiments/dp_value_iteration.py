"""Value iteration on the total-cost formulation (slides 11-Infinite-DP)
against the backward recursion over the stages, in results/dp/:
    table_dp_vi.csv   per board and objective: states, stages, iterations
                      of value iteration until V^(i+1) = V^(i), largest
                      difference from the backward recursion, times, and
                      sweeps of the Gauss-Seidel (backward order) version
    fig_dp_vi.png     2x3 board, target 64: error max|V^(i) - V*| and value
                      at the start V^(i)(x0) (= P(win within i moves)) vs i
Run from the project root:  python experiments/dp_value_iteration.py
"""
import time

import matplotlib.pyplot as plt
import numpy as np

from common import out_dir, save_csv, savefig

import dynamic_programming as dp
from game import Game2048

OUT = out_dir('dp')
CONFIGS = [(2, 2, 16, 'win'), (2, 2, 32, 'win'), (2, 3, 32, 'win'),
           (2, 3, 64, 'win'), (2, 3, 128, 'win'), (2, 2, 16, 'score'),
           (2, 3, 64, 'score')]


def main():
    rows = []
    curves = {}
    for r, c, target, objective in CONFIGS:
        game = Game2048(r, c, target=target)
        layers = dp.reachable_layers(game, stop_at_target=(objective == 'win'))
        t0 = time.perf_counter()
        V, _, _ = dp.solve_optimal(game, objective, layers)
        t_backward = time.perf_counter() - t0
        t0 = time.perf_counter()
        V_vi, _, hist = dp.value_iteration(game, objective, layers, V_ref=V)
        t_vi = time.perf_counter() - t0
        _, _, hist_gs = dp.value_iteration(game, objective, layers,
                                           order='backward', V_ref=V)
        exact_from = next(h['iteration'] for h in hist if h['error'] == 0.0) \
            if any(h['error'] == 0.0 for h in hist) else None
        row = {'board': f'{r}x{c}', 'target': target, 'objective': objective,
               'states': dp.count_states(layers), 'stages': len(layers),
               'vi_iterations': len(hist), 'exact_from_iteration': exact_from,
               'max_abs_diff': max(abs(V_vi[b] - V[b]) for b in V),
               'start_value': dp.start_value(game, V),
               'vi_time_s': t_vi, 'backward_time_s': t_backward,
               'gauss_seidel_sweeps': len(hist_gs),
               'gauss_seidel_error_after_1': hist_gs[0]['error']}
        rows.append(row)
        print(row, flush=True)
        if (r, c, target) == (2, 3, 64):
            curves[objective] = (hist, dp.start_value(game, V), len(layers))
    save_csv(rows, OUT / 'table_dp_vi.csv')

    hist, v_start, stages = curves['win']
    it = np.array([h['iteration'] for h in hist])
    err = np.array([h['error'] for h in hist])
    start = np.array([h['start'] for h in hist])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.semilogy(it[err > 0], err[err > 0], marker='o', ms=3)
    first_exact = it[err == 0][0] if (err == 0).any() else None
    if first_exact is not None:
        ax.axvline(first_exact, ls=':', c='gray', lw=1,
                   label=f'exact from iteration {first_exact}')
    ax.set_xlabel('iteration $i$')
    ax.set_ylabel(r'$\max_x |V^{(i)}(x) - V^*(x)|$')
    ax.set_title('Error of value iteration (Jacobi), 2x3 board, target 64')
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(np.r_[0, it], np.r_[0, start], marker='o', ms=3,
            label=r'$V^{(i)}$ at the start = P(win within $i$ moves)')
    ax.axhline(v_start, ls='--', c='k', lw=1, label=f'$V^*$ = {v_start:.4f}')
    ax.set_xlabel('iteration $i$')
    ax.set_ylabel('value at the start of the game')
    ax.set_title(f'{stages} stages; Gauss-Seidel in backward order: 1 sweep')
    ax.legend(fontsize=8, loc='center right')
    savefig(fig, OUT / 'fig_dp_vi.png')
    plt.close(fig)


if __name__ == '__main__':
    main()
