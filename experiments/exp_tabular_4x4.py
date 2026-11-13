"""The six tabular control methods of the report on the original 4x4 game
(tabular_control_fast.py: the algorithms of tabular_control.py with the table in a hash table),
with the settings of the 2x3 runs: score reward, alpha = 0.1 (Monte Carlo
control: sample averages), epsilon = 0.1, zero initial values, 200 000
episodes, 5 seeds. The greedy policy plays 200 games (fixed seed) every
20 000 episodes and 1 000 games (seed 2024) at the end; on a board that is
not in the table it plays the corner rule, and the share of such decisions
is logged.

Outputs in results/tabular_4x4/<method>_s<seed>/: train.csv, final.json and
top.npz (the 200 000 most visited entries of the table, for the policy
viewer; the full tables hold tens of millions of boards and are not saved);
results/tabular_4x4/summary.csv. With eps=<e> (more exploration while
learning, e.g. eps=0.2,0.3) the runs go to <method>_eps<e>_s<seed>/ and
summary_eps<e>.csv.

    python experiments/exp_tabular_4x4.py [methods...] [seeds=0,1] [episodes=N] [eps=0.2,0.3]
    python experiments/exp_tabular_4x4.py check   # port check on 2x3: the six
        methods with the settings of Table 7, exact P(win) of the greedy policy
"""
import json
import sys
import time

import numpy as np

from common import RESULTS, out_dir, run_grid, save_csv

METHODS = ('mc_control', 'sarsa', 'expected_sarsa', 'q_learning', 'nstep_sarsa', 'afterstate_sarsa')
SEEDS = range(5)
EPISODES = 200_000
EVAL_EVERY = 20_000
FINAL_GAMES = 1000
FINAL_SEED = 2024
TOP = 200_000
PROCESSES = 3          # tables of up to 3 GB each
EPS = 0.1              # exploration of the 2x3 runs


def alpha_of(method):
    return None if method == 'mc_control' else 0.1


def tag(eps):
    return '' if eps == EPS else f'_eps{eps:g}'


def run(cfg):
    import tabular_control_fast as rf
    agent = rf.FastTabular(cfg['method'], 4, 4, 2048, seed=cfg['seed'])
    d = out_dir('tabular_4x4', f"{cfg['method']}{tag(cfg['eps'])}_s{cfg['seed']}")
    t0 = time.perf_counter()
    hist = agent.train(cfg['episodes'], alpha_of(cfg['method']), cfg['eps'], eval_every=EVAL_EVERY)
    save_csv(hist, d / 'train.csv', columns=sorted({k for r in hist for k in r}))
    final = dict(cfg, **hist[-1], wall_s=time.perf_counter() - t0)
    final.update(agent.evaluate(FINAL_GAMES, FINAL_SEED, prefix='final_'))
    keys, vals, visits = agent.entries()
    for k in (2, 5, 10, 100, 1000):
        final[f'entries_visited_{k}'] = int((visits >= k).sum())
    top = agent.most_visited(TOP)
    np.savez(d / 'top.npz', keys=top[0], vals=top[1], visits=top[2],
             config=json.dumps(agent.config()))
    (d / 'final.json').write_text(json.dumps(final, indent=2))
    print(f"{cfg['method']} eps {cfg['eps']:g} seed {cfg['seed']}: greedy score {final['final_score']:.0f}, "
          f"unseen boards {final['final_unseen']:.1%}, {agent.n_states():,} entries "
          f"({final['wall_s']:.0f} s)", flush=True)
    return final


def check(cfg):
    """One run of the port check on 2x3."""
    import dynamic_programming as dp
    import tabular_control as rl
    import tabular_control_fast as rf
    from game import Game2048
    ev = rl.ExactEvaluator.cached(Game2048(2, 3, target=64), RESULTS / 'dp_cache' / 'eval_2x3_64.pkl')
    boards = sorted({b for layers in (ev.layers_win, ev.layers_score)
                     for layer in layers.values() for b in layer})
    agent = rf.FastTabular(cfg['method'], 2, 3, 64, seed=cfg['seed'])
    agent.learn(200_000, alpha_of(cfg['method']), 0.1, cfg['seed'] * 1_000_003)
    acts, known = agent.greedy_actions(boards)
    table = {b: a for b, a, k in zip(boards, acts, known) if k and a is not None}

    class Frozen:  # what ExactEvaluator needs from an agent
        reward = None

        @staticmethod
        def greedy_action(board, moves):
            return table.get(board)

        @staticmethod
        def policy():
            def pi(board, moves):
                if not moves:
                    return {}
                a = table.get(board)
                return {a: 1.0} if a is not None else dp.corner_policy(board, moves)
            return pi
    res = ev.evaluate(Frozen)
    return dict(cfg, p_win=res['p_win'], score=res['score'], agree=res['agree'],
                entries=agent.n_states())


def main():
    args = sys.argv[1:]
    opts = dict(a.split('=', 1) for a in args if '=' in a)
    methods = [a for a in args if a in METHODS] or list(METHODS)
    seeds = [int(s) for s in opts['seeds'].split(',')] if 'seeds' in opts else list(SEEDS)
    if 'check' in args:
        import tabular_control as rl
        from game import Game2048
        rl.ExactEvaluator.cached(Game2048(2, 3, target=64), RESULTS / 'dp_cache' / 'eval_2x3_64.pkl')
        rows = run_grid(check, [{'method': m, 'seed': s} for m in methods for s in seeds],
                        processes=12)
        report = {}
        with open(RESULTS / 'tabular' / 'summary.csv') as f:
            import csv
            for r in csv.DictReader(f):
                if r['reward'] == 'score':
                    report.setdefault(r['method'], []).append(float(r['p_win']))
        for r in rows:
            r['report_mean_p_win'] = float(np.mean(report[r['method']]))
        save_csv(rows, out_dir('tabular_4x4') / 'port_check_2x3.csv',
                 columns=['method', 'seed', 'p_win', 'report_mean_p_win', 'score', 'agree', 'entries'])
        for m in methods:
            p = [r['p_win'] for r in rows if r['method'] == m]
            print(f'{m:17s} port {np.mean(p):.3f} (seeds: {", ".join(f"{x:.3f}" for x in p)}), '
                  f'report {np.mean(report[m]):.3f}')
        return
    episodes = int(opts.get('episodes', EPISODES))
    epsilons = [float(e) for e in opts['eps'].split(',')] if 'eps' in opts else [EPS]
    configs = [{'method': m, 'seed': s, 'episodes': episodes, 'eps': e}
               for e in epsilons for m in methods for s in seeds]
    finals = run_grid(run, configs, processes=PROCESSES)
    for e in epsilons:
        rows = [f for f in finals if f['eps'] == e]
        save_csv(rows, out_dir('tabular_4x4') / f'summary{tag(e)}.csv',
                 columns=sorted({k for f in rows for k in f}))


if __name__ == '__main__':
    main()
