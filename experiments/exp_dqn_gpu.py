"""DQN on the GPU with many more games (dqn_gpu.py): the agent of the
final test, with 1024 games played in parallel, a replay memory of 1 000 000
moves and every move replayed 8 times on average (minibatches of 1024), as in
the original DQN paper, instead of the tutorial's 10 000 moves and 128 times.

    2x3 board, target 64: 200 000 games, exact evaluation every 20 000
    4x4 board:            100 000 games, 200 greedy games every 10 000, and
                          1 000 games with a fixed seed at the end

Outputs in results/dqn_gpu/<board>_s<seed>/: train.csv and model.pt (saved at
every evaluation), final.json; results/dqn_gpu/summary.csv.
Run from the project root:  python experiments/exp_dqn_gpu.py [2x3|4x4]
Options: games=<n>, seeds=0,1,..., n_envs=<n>, batch=<n>, ratio=<n>,
net=cnn (the convolutional network PairCNN instead of the perceptron; outputs
in results/dqn_gpu/cnn_<board>_s<seed>/), e.g.
    python experiments/exp_dqn_gpu.py 4x4 net=cnn games=50000 eval_every=5000 seeds=0,1
resume=1 continues the runs in the same folders up to games=<n>: exactly from
state.pt when it exists, otherwise from the saved network (model.pt), with the
replay memory first refilled with 1 000 000 moves of that network (without
it the restart from an empty memory costs several thousand points), e.g.
    python experiments/exp_dqn_gpu.py 4x4 net=cnn games=100000 eval_every=5000 seeds=0,1 resume=1
"""
import csv
import json
import sys
import time

from common import RESULTS, out_dir, run_grid, save_csv

BOARDS = {'2x3': dict(rows=2, cols=3, target=64, games=200_000, eval_every=20_000),
          '4x4': dict(rows=4, cols=4, target=2048, games=100_000, eval_every=10_000)}
SEEDS = (0, 1, 2)
CACHE = RESULTS / 'dp_cache' / 'eval_2x3_64.pkl'
FINAL_GAMES = 1000
FINAL_SEED = 2024


def run(cfg):
    import torch
    torch.set_num_threads(1)
    if cfg['net'] == 'cnn':  # TF32 tensor cores for the convolutions
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    import dqn_gpu as dg
    import tabular_control as rl
    from game import Game2048

    b = BOARDS[cfg['board']]
    agent = dg.VecDQN(b['rows'], b['cols'], b['target'], n_envs=cfg['n_envs'],
                      batch=cfg['batch'], replay_ratio=cfg['ratio'], seed=cfg['seed'],
                      device='cuda', net=cfg['net'])
    exact = None
    if cfg['board'] == '2x3':
        exact = dg.exact_evaluator(rl.ExactEvaluator.cached(Game2048(2, 3, target=64), CACHE))
    prefix = '' if cfg['net'] == 'mlp' else f"{cfg['net']}_"
    d = out_dir('dqn_gpu', f"{prefix}{cfg['board']}_s{cfg['seed']}")
    old, t_old = [], 0.0
    if cfg.get('resume'):  # continue an earlier run of this folder
        with open(d / 'train.csv') as f:
            old = list(csv.DictReader(f))
        t_old = float(old[-1]['time_s'])
        done_games = int(old[-1]['games'])
        if (d / 'state.pt').exists():
            agent.load_state(d / 'state.pt')
        else:  # only the network was saved: refill the memory before learning again
            agent.warm_start(d / 'model.pt', done_games)
            agent.prefill(agent.capacity)
        if (d / 'final.json').exists():
            (d / 'final.json').replace(d / f'final_{done_games}.json')

    def checkpoint(agent, hist):
        rows = old + [dict(r, time_s=r['time_s'] + t_old) for r in hist]
        with open(d / 'train.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[-1]))
            w.writeheader()
            w.writerows(rows)
        agent.save(d / 'model.pt')
        agent.save_state(d / 'state.pt')  # to continue exactly later (not committed)

    t0 = time.perf_counter()
    hist = dg.train(agent, cfg['games'], cfg['eval_every'], callback=checkpoint, exact=exact)
    final = dict(cfg, **hist[-1], wall_s=time.perf_counter() - t0 + t_old)
    res = agent.evaluate(FINAL_GAMES, FINAL_SEED)
    final.update(dg.eval_row(res, prefix='final_'))
    (d / 'final.json').write_text(json.dumps(final, indent=2))
    print(f"{cfg['board']} seed {cfg['seed']}: done in {final['wall_s']:.0f} s, final score "
          f"{final['final_score']:.0f}", flush=True)
    return final


def main():
    args = sys.argv[1:]
    boards = [a for a in args if a in BOARDS] or list(BOARDS)
    opts = dict(a.split('=', 1) for a in args if '=' in a)
    seeds = [int(s) for s in opts['seeds'].split(',')] if 'seeds' in opts else list(SEEDS)
    if '2x3' in boards:
        import tabular_control as rl
        from game import Game2048
        rl.ExactEvaluator.cached(Game2048(2, 3, target=64), CACHE)
    configs = [{'board': b, 'seed': s, 'games': int(opts.get('games', BOARDS[b]['games'])),
                'eval_every': int(opts.get('eval_every', BOARDS[b]['eval_every'])),
                'n_envs': int(opts.get('n_envs', 1024)), 'batch': int(opts.get('batch', 1024)),
                'ratio': int(opts.get('ratio', 8)), 'net': opts.get('net', 'mlp'),
                'resume': opts.get('resume', '0') == '1'}
               for b in boards for s in seeds]
    finals = run_grid(run, configs, processes=len(configs))
    net = configs[0]['net']
    save_csv(finals, out_dir('dqn_gpu') / ('summary.csv' if net == 'mlp' else f'summary_{net}.csv'),
             columns=sorted({k for f in finals for k in f}))


if __name__ == '__main__':
    main()
