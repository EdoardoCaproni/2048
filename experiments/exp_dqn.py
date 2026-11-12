"""Final test of the RL part: deep Q-learning (DQN) as in the PyTorch tutorial
(see dqn2048.py), on the 2x3 board with target 64 (exact evaluation of the
greedy policy with the DP) and on the original 4x4 game (Monte Carlo
evaluation). Score reward, as for the other agents.

Outputs in results/dqn/<board>_s<seed>/:
    train.csv       one row every EVAL_EVERY games (training statistics and
                    the evaluation of the greedy policy)
    model.pt        the trained policy network
    final.json      final evaluation (4x4: 1000 games with a fixed seed)
Run from the project root:  python experiments/exp_dqn.py [2x3|4x4] [quick]
Options: scale=<s> divides the rewards by s instead of the target tile
(outputs in results/dqn/<board>_r<s>_s<seed>/), episodes=<n>, seeds=0,1,...
e.g. the reward-scale check of the report:
    python experiments/exp_dqn.py 4x4 scale=64 episodes=2000 seeds=0
    python experiments/exp_dqn.py 4x4 scale=256 episodes=2000 seeds=0
"""
import json
import sys
import time

from common import RESULTS, out_dir, run_grid, save_csv

BOARDS = {'2x3': dict(rows=2, cols=3, target=64, episodes=20_000, eval_every=1_000),
          '4x4': dict(rows=4, cols=4, target=2048, episodes=10_000, eval_every=1_000)}
SEEDS = (0, 1, 2)
CACHE = RESULTS / 'dp_cache' / 'eval_2x3_64.pkl'
FINAL_GAMES_4X4 = 1000
FINAL_SEED = 2024


def run(cfg):
    import torch
    torch.set_num_threads(1)
    import dqn2048 as dq
    import rl2048 as rl
    from game2048 import Game2048

    b = BOARDS[cfg['board']]
    game = Game2048(b['rows'], b['cols'], target=b['target'])
    if cfg['board'] == '2x3':
        evaluator = dq.exact_evaluator(rl.ExactEvaluator.cached(game, CACHE))
    else:
        evaluator = dq.mc_evaluator(n_games=200)
    agent = dq.DQNAgent(game, seed=cfg['seed'], reward_scale=cfg.get('scale'))
    t0 = time.perf_counter()
    hist = dq.train(agent, cfg['episodes'], eval_every=cfg['eval_every'],
                    evaluator=evaluator, verbose=True)
    d = out_dir('dqn', run_name(cfg))
    save_csv(hist, d / 'train.csv')
    agent.save(d / 'model.pt')
    final = dict(cfg, **hist[-1], wall_s=time.perf_counter() - t0)
    if cfg['board'] == '4x4':
        res = agent.play_greedy(cfg['final_games'], FINAL_SEED)
        final.update({'final_games': cfg['final_games'],
                      'final_score': float(res['score'].mean()),
                      'final_score_std': float(res['score'].std(ddof=1)),
                      'final_moves': float(res['moves'].mean()),
                      'final_max_tile': int(res['max_tile'].max())})
        for t in (256, 512, 1024, 2048, 4096):
            final[f'final_p{t}'] = float((res['max_tile'] >= t).mean())
        tiles, counts = __import__('numpy').unique(res['max_tile'], return_counts=True)
        final['final_tiles'] = {int(t): int(c) for t, c in zip(tiles, counts)}
    (d / 'final.json').write_text(json.dumps(final, indent=2))
    print(f"{run_name(cfg)} done in {final['wall_s']:.0f} s", flush=True)
    return final


def run_name(cfg):
    if cfg.get('scale') is None:
        return f"{cfg['board']}_s{cfg['seed']}"
    return f"{cfg['board']}_r{cfg['scale']:g}_s{cfg['seed']}"


def main():
    args = sys.argv[1:]
    boards = [a for a in args if a in BOARDS] or list(BOARDS)
    quick = 'quick' in args
    opts = dict(a.split('=', 1) for a in args if '=' in a)
    scale = float(opts['scale']) if 'scale' in opts else None
    seeds = [int(s) for s in opts['seeds'].split(',')] if 'seeds' in opts else list(SEEDS)
    if '2x3' in boards:  # build the DP evaluator once, the workers load it
        import rl2048 as rl
        from game2048 import Game2048
        rl.ExactEvaluator.cached(Game2048(2, 3, target=64), CACHE)
    configs = []
    for board in boards:
        b = BOARDS[board]
        episodes = int(opts.get('episodes', b['episodes']))
        for seed in seeds[:1] if quick else seeds:
            configs.append({'board': board, 'seed': seed, 'scale': scale,
                            'episodes': 200 if quick else episodes,
                            'eval_every': 100 if quick else b['eval_every'],
                            'final_games': 20 if quick else FINAL_GAMES_4X4})
    finals = run_grid(run, configs, processes=len(configs))
    for f in finals:
        f.pop('final_tiles', None)
    out = out_dir('dqn')
    name = 'summary_quick' if quick else 'summary' if scale is None else f'summary_r{scale:g}'
    save_csv(finals, out / f'{name}.csv', columns=sorted({k for f in finals for k in f}))


if __name__ == '__main__':
    main()
