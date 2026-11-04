"""Throughput of the pure-Python engine and of the numba bitboard engine.

Writes results/bench_engine.json. Run from the project root:
    python experiments/bench_engine.py
"""
import random
import time

from common import RESULTS, save_json  # noqa: F401 (sets sys.path)

import fast2048 as fb
from game2048 import Game2048


def bench_python(rows, cols, target, n_games):
    g = Game2048(rows, cols, target=target)
    rng = random.Random(0)
    moves = 0
    t0 = time.perf_counter()
    for _ in range(n_games):
        board = g.sample_initial(rng)
        while True:
            legal = g.moves(board)
            if not legal:
                break
            _, after, _ = rng.choice(legal)
            moves += 1
            board = g.sample_spawn(after, rng)
    dt = time.perf_counter() - t0
    return moves, dt


def bench_fast(rows, cols, target_exp, n_games):
    fb.seed(0)
    t0 = time.perf_counter()
    _, _, n_moves, _ = fb.play_random(rows, cols, target_exp, 0.1, n_games, False)
    dt = time.perf_counter() - t0
    return int(n_moves.sum()), dt


def main():
    t0 = time.perf_counter()
    fb.warm_up()
    compile_s = time.perf_counter() - t0
    out = {'numba_compile_s': compile_s}
    print(f'numba compile/load: {compile_s:.1f} s')
    for rows, cols, target, n_py, n_fast in ((2, 3, 64, 5000, 200000),
                                            (4, 4, 2048, 300, 20000)):
        key = f'{rows}x{cols}'
        moves, dt = bench_python(rows, cols, target, n_py)
        out[f'python_{key}_moves_per_s'] = moves / dt
        print(f'python  {key}: {moves / dt:10,.0f} moves/s ({n_py} games)')
        moves, dt = bench_fast(rows, cols, target.bit_length() - 1, n_fast)
        out[f'fast_{key}_moves_per_s'] = moves / dt
        print(f'numba   {key}: {moves / dt:10,.0f} moves/s ({n_fast} games)')
    save_json(out, RESULTS / 'bench_engine.json')


if __name__ == '__main__':
    main()
