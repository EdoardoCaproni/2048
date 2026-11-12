"""Data for the policy viewer (viewer/index.html): the best policy found by
every method, retrained with the configuration and the seed of its best run
in the report (the training is deterministic, so these are the same agents),
and exported as JavaScript files that the page loads.

2x3 board, target 64: every policy is a table over the 27 022 decision
states (greedy action of the agent; unseen boards fall back to the corner
rule, as in the exact evaluation), with its exact P(win) and expected score:
    optimal policy (DP), corner rule, greedy rule;
    MC control, Sarsa, Q-learning, Expected Sarsa, 3-step Sarsa: the best of
    the 5 seeds x 3 settings of results/tabular/summary.csv;
    Sarsa on afterstates: full-board table, greedy, optimistic (n-tuple code);
    semi-gradient Sarsa with n-tuple features (rows, columns, squares);
    DQN (final test, seed 1).
4x4 board, target 2048: the page computes the policies itself:
    corner, greedy and random rules;
    semi-gradient Sarsa with 4-tuple features (rows, columns, squares,
    alpha = 0.1, seed 0, 300 000 games: the weights are exported; the
    6-tuple network is 134 MB, too large for a web page);
    DQN (final test, seed 1: the weights are exported).

Outputs: viewer/data_2x3.js, viewer/data_4x4.js; the trained agents are
cached in results/viewer/.
Run from the project root:  python experiments/export_viewer.py [2x3] [4x4]
(the 4x4 stage trains the n-tuple network for about one hour the first time)
"""
import base64
import csv
import json
import sys
import time

import numpy as np

from common import RESULTS, ROOT, out_dir, run_grid

import dp2048 as dp
from game2048 import ACTIONS, Game2048

OUT = out_dir('viewer')
VIEWER = ROOT / 'viewer'
CACHE = RESULTS / 'dp_cache' / 'eval_2x3_64.pkl'
CHARS = 'dlru'  # ACTIONS order: down, left, right, up
TAB_ALPHA = {'mc_control': None, 'sarsa': 0.1, 'q_learning': 0.1,
             'expected_sarsa': 0.1, 'nstep_sarsa': 0.1}
NT_SMALL = {  # name -> (tuples, symmetric, reward, init, seed): best runs on 2x3
    'afterstate_sarsa': ('full', False, 'win', 1.0, 2),
    'ntuple_sarsa': ('rcs', True, 'win', 1.0, 2),
}
NT_4X4 = dict(tuples='rcs', alpha=0.1, seed=0, games=300_000)
DQN_SEED = 1


def board_key(board):
    return ''.join('%x' % e for e in board)


def boards_2x3():
    game = Game2048(2, 3, target=64)
    layers = dp.reachable_layers(game)
    boards = sorted((b for layer in layers.values() for b in layer), key=board_key)
    return game, layers, boards


def actions_string(boards, action_of):
    """One character per board: d/l/r/u, '-' at game over."""
    out = []
    for b in boards:
        a = action_of(b)
        out.append('-' if a is None else CHARS[ACTIONS.index(a)])
    return ''.join(out)


# ----------------------------------------------------------------------
# training jobs (run in parallel), each saves results/viewer/<name>.json
# ----------------------------------------------------------------------
def job(spec):
    kind, name = spec['kind'], spec['name']
    path = OUT / f'{name}.json'
    if path.exists():
        return name
    t0 = time.perf_counter()
    if kind == 'tabular':
        import rl2048 as rl
        from game2048 import Env2048
        game, layers, boards = boards_2x3()
        cfg = spec['cfg']
        env = Env2048(game, reward=cfg['reward'])
        agent, _ = rl.train(name, env, 200_000, alpha=TAB_ALPHA[name], epsilon=0.1,
                            n=3, q_init=cfg['q_init'], seed=cfg['seed'])
        pi = agent.policy(fallback=dp.corner_policy)
        acts = actions_string(boards, lambda b: next(iter(pi(b, game.moves(b))), None))
        res = rl.ExactEvaluator.cached(game, CACHE).evaluate(agent)
        info = dict(cfg, episodes=200_000, p_win=res['p_win'], score=res['score'],
                    agree=res['agree'])
    elif kind == 'ntuple_small':
        import ntuple2048 as nt
        import rl2048 as rl
        game, layers, boards = boards_2x3()
        tuples, symmetric, reward, init, seed = NT_SMALL[name]
        net = nt.NTupleNet(2, 3, 64, tuples=tuples, symmetric=symmetric,
                           dtype=np.float64, init=init)
        net.train(300_000, alpha=0.1, reward=reward, seed=seed, block=1_000)
        frozen = net.frozen(boards, reward)
        acts = actions_string(boards, lambda b: frozen.actions.get(b))
        res = rl.ExactEvaluator.cached(game, CACHE).evaluate(frozen)
        info = dict(tuples=tuples, symmetric=symmetric, reward=reward, init=init,
                    seed=seed, games=300_000, alpha=0.1, p_win=res['p_win'],
                    score=res['score'], agree=res['agree'])
    elif kind == 'ntuple_4x4':
        import ntuple2048 as nt
        c = NT_4X4
        net = nt.NTupleNet(4, 4, 2048, tuples=c['tuples'], symmetric=True,
                           dtype=np.float32)
        net.train(c['games'], alpha=c['alpha'], reward='score', seed=c['seed'], block=1_000)
        net.save(OUT / 'ntuple_4x4.npz')
        scores, max_exps, moves, won = net.play(1_000, 'score', seed=2024)
        acts = None
        info = dict(c, eval_games=1_000, eval_score=float(scores.mean()),
                    eval_moves=float(moves.mean()),
                    **{f'eval_p{2 ** e}': float((max_exps >= e).mean()) for e in range(9, 14)})
    else:
        raise ValueError(kind)
    info['time_s'] = time.perf_counter() - t0
    path.write_text(json.dumps({'actions': acts, 'info': info}))
    print(f'{name}: {json.dumps(info)}', flush=True)
    return name


def best_tabular_runs():
    rows = list(csv.DictReader(open(RESULTS / 'tabular' / 'summary.csv')))
    best = {}
    for r in rows:
        if r['method'] in TAB_ALPHA and (r['method'] not in best or
                                         float(r['p_win']) > float(best[r['method']]['p_win'])):
            best[r['method']] = r
    return {m: dict(reward=r['reward'], q_init=float(r['q_init']), seed=int(r['seed']),
                    report_p_win=float(r['p_win'])) for m, r in best.items()}


# ----------------------------------------------------------------------
# export
# ----------------------------------------------------------------------
def b64(arr, dtype=np.float32):
    return base64.b64encode(np.ascontiguousarray(arr, dtype=dtype).tobytes()).decode('ascii')


def write_js(name, obj):
    VIEWER.mkdir(exist_ok=True)
    path = VIEWER / f'data_{name}.js'
    path.write_text('window.VIEWER_DATA = window.VIEWER_DATA || {};\n'
                    f"window.VIEWER_DATA['{name}'] = {json.dumps(obj, separators=(',', ':'))};\n",
                    encoding='utf-8')
    print(f'wrote {path} ({path.stat().st_size / 1e6:.2f} MB)')


def export_2x3():
    import torch
    import dqn2048 as dq
    import rl2048 as rl
    game, layers, boards = boards_2x3()
    ev = rl.ExactEvaluator.cached(game, CACHE)
    V = ev.V_win
    _, table, _ = dp.solve_optimal(game, layers=layers)

    def exact(pi):
        def safe(board, moves):  # past the target a table has no entry: corner rule
            try:
                return pi(board, moves)
            except KeyError:
                return dp.corner_policy(board, moves)
        return (dp.start_value(game, dp.evaluate_policy(game, safe, 'win', ev.layers_win)),
                dp.start_value(game, dp.evaluate_policy(game, safe, 'score', ev.layers_score)))

    policies = []

    def add(pid, group, label, acts, p_win, score, note):
        policies.append({'id': pid, 'group': group, 'label': label, 'actions': acts,
                         'p_win': round(p_win, 4), 'score': round(score, 1), 'note': note})

    acts = actions_string(boards, lambda b: table.get(b))
    add('optimal', 'Dynamic programming', 'Optimal policy (value iteration / backward recursion)',
        acts, *exact(dp.table_policy(table)), 'maximises the probability of creating 64')
    for name, pol, label in (('corner', dp.corner_policy, 'Corner rule'),
                             ('greedy', dp.greedy_policy, 'Greedy rule')):
        acts = actions_string(boards, lambda b: next(iter(pol(b, game.moves(b))), None))
        add(name, 'Rules of thumb', label, acts, *exact(pol),
            'down > left > right > up' if name == 'corner' else 'largest immediate score')
    labels = {'mc_control': 'Monte Carlo control', 'sarsa': 'Sarsa', 'q_learning': 'Q-learning',
              'expected_sarsa': 'Expected Sarsa', 'nstep_sarsa': '3-step Sarsa'}
    for name in labels:
        d = json.loads((OUT / f'{name}.json').read_text())
        i = d['info']
        reward = 'win reward' + (', optimistic' if i['q_init'] else '') if i['reward'] == 'win' else 'score reward'
        add(name, 'Tabular RL', labels[name], d['actions'], i['p_win'], i['score'],
            f"200 000 episodes, {reward}, epsilon 0.1, seed {i['seed']}")
    d = json.loads((OUT / 'afterstate_sarsa.json').read_text())
    add('afterstate_sarsa', 'Tabular RL', 'Sarsa on afterstates', d['actions'],
        d['info']['p_win'], d['info']['score'],
        '300 000 games, win reward, optimistic, greedy, seed 2')
    d = json.loads((OUT / 'ntuple_sarsa.json').read_text())
    add('ntuple_sarsa', 'Function approximation', 'Semi-gradient Sarsa, n-tuple features',
        d['actions'], d['info']['p_win'], d['info']['score'],
        'rows, columns, squares; 300 000 games, win reward, optimistic, seed 2')
    agent = dq.DQNAgent(game, seed=0)
    ckpt = torch.load(RESULTS / 'dqn' / f'2x3_s{DQN_SEED}' / 'model.pt')
    agent.policy_net.load_state_dict(ckpt['state_dict'])
    a, _ = agent.greedy_batch(boards)
    acts = ''.join('-' if x < 0 else CHARS[x] for x in a)
    final = json.loads((RESULTS / 'dqn' / f'2x3_s{DQN_SEED}' / 'final.json').read_text())
    add('dqn', 'Deep RL (final test)', 'DQN (PyTorch tutorial)', acts, final['p_win'],
        final['score'], f"20 000 games, score reward, seed {DQN_SEED}")
    write_js('2x3', {'rows': 2, 'cols': 3, 'target': 64, 'p4': game.p4,
                     'keys': ''.join(board_key(b) for b in boards),
                     'vstar': [int(round(1e6 * V[b])) for b in boards],  # millionths
                     'policies': policies})


def export_4x4():
    import torch
    import ntuple2048 as nt
    out = {'rows': 4, 'cols': 4, 'target': 2048, 'p4': 0.1}
    bench = {r['policy']: r for r in csv.DictReader(open(RESULTS / '4x4' / 'benchmarks.csv'))}
    out['rules'] = {k: {'score': round(float(v['mean_score'])),
                        'p512': float(v['p_ge_512']), 'p1024': float(v['p_ge_1024'])}
                    for k, v in bench.items()}
    npz = OUT / 'ntuple_4x4.npz'
    if npz.exists():
        net = nt.NTupleNet.load(npz)
        info = json.loads((OUT / 'ntuple_4x4.json').read_text())['info']
        out['ntuple'] = {'k': int(net.k), 'cells': net.cells.tolist(), 'owner': net.owner.tolist(),
                         'rows': int(net.W.shape[0]), 'W': b64(net.W),
                         'stats': {k: info[k] for k in info if k.startswith('eval_')},
                         'note': f"4-tuples (rows, columns, squares), alpha {info['alpha']}, "
                                 f"{info['games']:,} games, seed {info['seed']}"}
    ckpt = torch.load(RESULTS / 'dqn' / f'4x4_s{DQN_SEED}' / 'model.pt')
    sd = ckpt['state_dict']
    out['dqn'] = {'layers': [{'w': b64(sd[f'layer{i}.weight'].numpy()),
                              'b': b64(sd[f'layer{i}.bias'].numpy()),
                              'in': int(sd[f'layer{i}.weight'].shape[1]),
                              'out': int(sd[f'layer{i}.weight'].shape[0])} for i in (1, 2, 3)]}
    final = json.loads((RESULTS / 'dqn' / f'4x4_s{DQN_SEED}' / 'final.json').read_text())
    out['dqn']['stats'] = {'eval_score': final['final_score'], 'eval_moves': final['final_moves'],
                           'eval_p512': final['final_p512'], 'eval_p1024': final['final_p1024'],
                           'eval_p2048': final['final_p2048']}
    out['dqn']['note'] = f"10 000 games, score reward, seed {DQN_SEED}"
    write_js('4x4', out)


def main():
    stages = [a for a in sys.argv[1:] if a in ('2x3', '4x4')] or ['2x3', '4x4']
    specs = []
    if '2x3' in stages:
        import rl2048 as rl
        rl.ExactEvaluator.cached(Game2048(2, 3, target=64), CACHE)
        for name, cfg in best_tabular_runs().items():
            specs.append({'kind': 'tabular', 'name': name, 'cfg': cfg})
        for name in NT_SMALL:
            specs.append({'kind': 'ntuple_small', 'name': name})
    if '4x4' in stages:
        specs.append({'kind': 'ntuple_4x4', 'name': 'ntuple_4x4'})
    todo = [s for s in specs if not (OUT / f"{s['name']}.json").exists()]
    if todo:
        run_grid(job, todo, processes=len(todo))
    if '2x3' in stages:
        export_2x3()
    if '4x4' in stages:
        export_4x4()


if __name__ == '__main__':
    main()
