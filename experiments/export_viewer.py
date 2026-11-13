"""Data for the policy viewer (viewer/index.html, also published as the
project's GitHub page): the best policy found by every method of the report,
exported as JavaScript files that the page loads. The tabular agents are
retrained with the configuration and the seed of their best run (the
training is deterministic, so these are the same agents); the DQN networks
are the saved ones of the best seed of each configuration of the final test.

2x3 board, target 64: every policy is a table over the 27 022 decision
states (greedy action of the agent; unseen boards fall back to the corner
rule, as in the exact evaluation), with its exact P(win) and expected score:
    optimal policy (DP), corner rule, greedy rule;
    MC control, Sarsa, Q-learning, Expected Sarsa, 3-step Sarsa, Sarsa on
    afterstates: the best of the 5 seeds x 3 settings of
    results/tabular/summary.csv;
    DQN with the tutorial's schedule (results/dqn/) and on the GPU
    (results/dqn_gpu/): the seed with the largest exact P(win);
    not in the report: semi-gradient Sarsa with n-tuple features (rows,
    columns, squares).
4x4 board, target 2048: the page computes the policies itself:
    corner, greedy and random rules;
    MC control, Sarsa, Expected Sarsa, Q-learning, 3-step Sarsa, Sarsa on
    afterstates (experiments/exp_tabular_4x4.py, best of 5 seeds): the
    50 000 boards (afterstates) met most often in training, in
    viewer/data_4x4_tabular.js, loaded when one of them is chosen; the
    corner rule plays elsewhere;
    DQN with the tutorial's schedule, on the GPU with the perceptron, and on
    the GPU with the convolutional network: the seed with the largest final
    mean score; the weights are exported (the convolutional network, 2
    million weights, in half precision);
    not in the report: semi-gradient Sarsa with 4-tuple features (rows,
    columns, squares, alpha = 0.1, seed 0, 300 000 games), the approach of
    the n-tuple players of the literature.

Outputs: viewer/data_2x3.js, viewer/data_4x4.js (the page loads the second
one only when the 4x4 board is chosen); the retrained agents are cached in
results/viewer/.
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

import dynamic_programming as dp
from game import ACTIONS, Game2048

OUT = out_dir('viewer')
VIEWER = ROOT / 'viewer'
CACHE = RESULTS / 'dp_cache' / 'eval_2x3_64.pkl'
CHARS = 'dlru'  # ACTIONS order: down, left, right, up
TAB_ALPHA = {'mc_control': None, 'sarsa': 0.1, 'q_learning': 0.1,
             'expected_sarsa': 0.1, 'nstep_sarsa': 0.1, 'afterstate_sarsa': 0.1}
NT_SMALL = {  # name -> (tuples, symmetric, reward, init, seed); not in the report
    'ntuple_sarsa': ('rcs', True, 'win', 1.0, 2),
}
NT_4X4 = dict(tuples='rcs', alpha=0.1, seed=0, games=300_000)
GROUP_DQN = 'Deep RL (final test)'
GROUP_EXTRA = 'Extra'


def num(n):
    """Integer with spaces between the thousands, as in the report."""
    return f'{int(n):,}'.replace(',', ' ')


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
    t0 = time.perf_counter()
    if kind == 'tabular':
        import tabular_control as rl
        from game import Env2048
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
        import ntuple_network as nt
        import tabular_control as rl
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
        import ntuple_network as nt
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


def cached(spec):
    """True if results/viewer/ already holds this agent (for a tabular
    method: trained with the configuration of its current best run)."""
    path = OUT / f"{spec['name']}.json"
    if not path.exists():
        return False
    if spec['kind'] == 'tabular':
        info = json.loads(path.read_text())['info']
        return all(info.get(k) == v for k, v in spec['cfg'].items())
    return True


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
    import dqn_tutorial as dq
    import tabular_control as rl
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
              'expected_sarsa': 'Expected Sarsa', 'nstep_sarsa': '3-step Sarsa',
              'afterstate_sarsa': 'Sarsa on afterstates'}
    for name in labels:
        d = json.loads((OUT / f'{name}.json').read_text())
        i = d['info']
        reward = 'win reward' + (', optimistic' if i['q_init'] else '') if i['reward'] == 'win' else 'score reward'
        add(name, 'Tabular RL', labels[name], d['actions'], i['p_win'], i['score'],
            f"200 000 episodes, {reward}, epsilon 0.1, seed {i['seed']}")
    for folder, pid, label, how in DQN_2X3:
        run = best_run(folder, '2x3', 'p_win')
        final = json.loads((run / 'final.json').read_text())
        agent = dq.DQNAgent(game, seed=0)
        agent.policy_net.load_state_dict(torch.load(run / 'model.pt', map_location='cpu')['state_dict'])
        a, _ = agent.greedy_batch(boards)
        acts = ''.join('-' if x < 0 else CHARS[x] for x in a)
        p_win, score = exact(dp.table_policy({b: ACTIONS[x] for b, x in zip(boards, a) if x >= 0}))
        if abs(p_win - final['p_win']) > 1e-6:  # the same policy as in the report
            raise RuntimeError(f'{run}: exact P(win) {p_win} against {final["p_win"]} in final.json')
        games = final.get('games') or final['episode']
        add(pid, GROUP_DQN, label, acts, p_win, score,
            f"{how}, {num(round(games, -3))} games, score reward, seed {final['seed']}")
    d = json.loads((OUT / 'ntuple_sarsa.json').read_text())
    add('ntuple_sarsa', GROUP_EXTRA, 'Semi-gradient Sarsa, n-tuple features',
        d['actions'], d['info']['p_win'], d['info']['score'],
        'the approach of the n-tuple players of the literature (Szubert & Jaśkowski 2014); '
        'rows, columns, squares; 300 000 games, win reward, optimistic, seed 2')
    write_js('2x3', {'rows': 2, 'cols': 3, 'target': 64, 'p4': game.p4,
                     'keys': ''.join(board_key(b) for b in boards),
                     'vstar': [int(round(1e6 * V[b])) for b in boards],  # millionths
                     'policies': policies})


# DQN of the final test: (results folder, viewer id, label, how it was trained)
DQN_2X3 = (('dqn', 'dqn', 'DQN, tutorial schedule', "schedule of the PyTorch tutorial"),
           ('dqn_gpu', 'dqn_gpu', 'DQN on the GPU', "1 024 games at once on the GPU"))
DQN_4X4 = (('dqn', '4x4', 'dqn', "schedule of the PyTorch tutorial, perceptron"),
           ('dqn_gpu', '4x4', 'dqn_gpu', "1 024 games at once on the GPU, perceptron"),
           ('dqn_gpu', 'cnn_4x4', 'dqn_cnn', "1 024 games at once on the GPU, "
                                             "convolutional network over the pairs of cells"))


def best_run(folder, prefix, key):
    """Folder of the seed of a DQN configuration with the largest final key."""
    runs = [d for d in sorted((RESULTS / folder).glob(f'{prefix}_s*'))
            if (d / 'final.json').exists() and (d / 'model.pt').exists()]
    return max(runs, key=lambda d: json.loads((d / 'final.json').read_text())[key])


def export_network(run):
    """Weights of a saved DQN for the page: the perceptron in single
    precision, the convolutional network (2 million weights) in half
    precision, with the shapes of PyTorch."""
    import torch
    ck = torch.load(run / 'model.pt', map_location='cpu')
    sd = ck['state_dict']
    if ck.get('net') == 'cnn':
        rows, cols, _ = ck['board']
        return {'kind': 'cnn', 'rows': rows, 'cols': cols,
                'second': [k.split('.')[1] for k in sd if k.startswith('second.') and k.endswith('.weight')],
                'tensors': {k: {'shape': list(v.shape), 'data': b64(v.numpy(), np.float16)}
                            for k, v in sd.items()}}
    return {'kind': 'mlp',
            'layers': [{'w': b64(sd[f'layer{i}.weight'].numpy()), 'b': b64(sd[f'layer{i}.bias'].numpy()),
                        'in': int(sd[f'layer{i}.weight'].shape[1]),
                        'out': int(sd[f'layer{i}.weight'].shape[0])} for i in (1, 2, 3)]}


TABLE_ENTRIES = 50_000  # boards (afterstates) of each 4x4 table in the viewer


def export_4x4_tabular():
    """The six tabular methods on 4x4 (experiments/exp_tabular_4x4.py), best
    seed of each: their tables hold tens of millions of boards, so the page
    gets the TABLE_ENTRIES boards (afterstates) met most often in training,
    and plays the corner rule elsewhere; its statistics are those of this
    restricted table, on the same 1 000 games as the full one. Returns the
    metadata for data_4x4.js and the tables for data_4x4_tabular.js."""
    import game_fast as fb
    import tabular_control_fast as rf
    meta, tables = {}, {}
    for method in rf.METHODS:
        runs = sorted((RESULTS / 'tabular_4x4').glob(f'{method}_s*'))
        runs = [d for d in runs if (d / 'final.json').exists() and (d / 'top.npz').exists()]
        if not runs:
            continue
        run = max(runs, key=lambda d: json.loads((d / 'final.json').read_text())['final_score'])
        final = json.loads((run / 'final.json').read_text())
        data = np.load(run / 'top.npz')
        cfg = json.loads(str(data['config']))
        n = min(TABLE_ENTRIES, len(data['keys']))
        agent = rf.FastTabular.from_entries(cfg, data['keys'][:n], data['vals'][:n], data['visits'][:n])
        res = agent.evaluate(1000, 2024)  # the final evaluation of the experiment
        boards = [fb.decode(k, 4, 4) for k in data['keys'][:n]]
        after = method == 'afterstate_sarsa'
        if after:
            tables[method] = {'keys': ''.join(board_key(b) for b in boards),
                              'values': b64(data['vals'][:n, 0])}
        else:
            acts, _ = agent.greedy_actions(boards)
            tables[method] = {'keys': ''.join(board_key(b) for b in boards),
                              'actions': ''.join(CHARS[ACTIONS.index(a)] for a in acts)}
        what = 'afterstates' if after else 'boards'
        meta[method] = {
            'after': after,
            'stats': {f'eval_{k}': res[k] for k in ('score', 'moves', 'p256', 'p512', 'p1024', 'p2048', 'unseen')},
            'note': (f"{num(final['episodes'])} episodes, score reward, "
                     f"{'sample averages' if method == 'mc_control' else 'alpha 0.1'}, epsilon 0.1, "
                     f"seed {final['seed']}, the best of {len(runs)}. The table holds {num(final['states'])} "
                     f"{what}; the page uses the {num(n)} met most often in training (with the whole table "
                     f"{num(round(final['final_score']))} points, {final['final_unseen']:.0%} of the moves on "
                     f"unseen boards); elsewhere it plays the corner rule")}
        print(f"4x4 {method}: seed {final['seed']}, {n} of {final['states']:,} {what}: "
              f"{res['score']:.0f} points against {final['final_score']:.0f} with the whole table; "
              f"unseen {res['unseen']:.1%} against {final['final_unseen']:.1%}")
    return meta, tables


def export_4x4():
    import ntuple_network as nt
    out = {'rows': 4, 'cols': 4, 'target': 2048, 'p4': 0.1}
    bench = {r['policy']: r for r in csv.DictReader(open(RESULTS / '4x4' / 'benchmarks.csv'))}
    out['rules'] = {k: {'score': round(float(v['mean_score'])),
                        'p512': float(v['p_ge_512']), 'p1024': float(v['p_ge_1024'])}
                    for k, v in bench.items()}
    meta, tables = export_4x4_tabular()
    if meta:
        out['tabular'] = meta
        write_js('4x4_tabular', tables)
    for folder, prefix, pid, how in DQN_4X4:
        run = best_run(folder, prefix, 'final_score')
        final = json.loads((run / 'final.json').read_text())
        games = final.get('games') or final['episode']
        net = export_network(run)
        half = '; the page plays it with the weights rounded to half precision' if net['kind'] == 'cnn' else ''
        out[pid] = dict(net,
                        stats={'eval_score': final['final_score'], 'eval_moves': final['final_moves'],
                               **{f'eval_p{t}': final.get(f'final_p{t}', 0.0)
                                  for t in (512, 1024, 2048, 4096)}},
                        note=f"{how}; {num(round(games, -3))} games, score reward, seed {final['seed']}; "
                             f"statistics of 1 000 games{half}")
    npz = OUT / 'ntuple_4x4.npz'
    if npz.exists():
        net = nt.NTupleNet.load(npz)
        info = json.loads((OUT / 'ntuple_4x4.json').read_text())['info']
        out['ntuple'] = {'k': int(net.k), 'cells': net.cells.tolist(), 'owner': net.owner.tolist(),
                         'rows': int(net.W.shape[0]), 'W': b64(net.W),
                         'stats': {k: info[k] for k in info if k.startswith('eval_')},
                         'note': f"the approach of the n-tuple players of the literature (Szubert & "
                                 f"Jaśkowski 2014), with 4-tuples (rows, columns, squares); alpha {info['alpha']}, "
                                 f"{num(info['games'])} games, score reward, seed {info['seed']}"}
    write_js('4x4', out)


def main():
    stages = [a for a in sys.argv[1:] if a in ('2x3', '4x4')] or ['2x3', '4x4']
    specs = []
    if '2x3' in stages:
        import tabular_control as rl
        rl.ExactEvaluator.cached(Game2048(2, 3, target=64), CACHE)
        for name, cfg in best_tabular_runs().items():
            specs.append({'kind': 'tabular', 'name': name, 'cfg': cfg})
        for name in NT_SMALL:
            specs.append({'kind': 'ntuple_small', 'name': name})
    if '4x4' in stages:
        specs.append({'kind': 'ntuple_4x4', 'name': 'ntuple_4x4'})
    todo = [s for s in specs if not cached(s)]
    if todo:
        run_grid(job, todo, processes=len(todo))
    if '2x3' in stages:
        export_2x3()
    if '4x4' in stages:
        export_4x4()


if __name__ == '__main__':
    main()
