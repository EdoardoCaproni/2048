"""Checks for the 2048 engine and the exact DP.

Run from this folder with:  python test_2048.py
"""
import random
import time

import dp2048 as dp
from game2048 import ACTIONS, Env2048, Game2048, slide_line


def test_slide_line():
    cases = {
        (1, 1, 0, 0): ((2, 0, 0, 0), 4),     # 2 2 . . -> 4 . . .
        (1, 1, 1, 1): ((2, 2, 0, 0), 8),     # 2 2 2 2 -> 4 4 . .
        (2, 1, 1, 0): ((2, 2, 0, 0), 4),     # 4 2 2 . -> 4 4 . .
        (1, 0, 1, 0): ((2, 0, 0, 0), 4),     # 2 . 2 . -> 4 . . .
        (1, 2, 3, 0): ((1, 2, 3, 0), 0),     # nothing to merge
        (0, 0, 0, 1): ((1, 0, 0, 0), 0),     # pure slide
        (1, 1, 2, 2): ((2, 3, 0, 0), 12),    # 2 2 4 4 -> 4 8 . .
        (2, 2, 1, 1): ((3, 2, 0, 0), 12),    # 4 4 2 2 -> 8 4 . .
    }
    for line, expected in cases.items():
        assert slide_line(line) == expected, (line, slide_line(line))


def test_directions():
    g = Game2048(2, 3, target=64)
    board = (1, 1, 0,
             0, 2, 2)                 # rows: [2 2 .] and [. 4 4]
    assert g.afterstate(board, 'left') == ((2, 0, 0, 3, 0, 0), 12)
    assert g.afterstate(board, 'right') == ((0, 0, 2, 0, 0, 3), 12)
    assert g.afterstate(board, 'up') == ((1, 1, 2, 0, 2, 0), 0)
    assert g.afterstate(board, 'down') == ((0, 1, 0, 1, 2, 2), 0)
    assert g.legal_actions(board) == list(ACTIONS)


def test_tile_sum_and_distributions():
    g = Game2048(3, 3, target=64)
    rng = random.Random(1)
    assert abs(sum(p for _, p in g.initial_distribution()) - 1) < 1e-12
    board = g.sample_initial(rng)
    while g.moves(board):
        for _, after, _ in g.moves(board):
            # the slide/merge keeps the tile sum, the new tile adds 2 or 4
            assert g.tile_sum(after) == g.tile_sum(board)
            dist = g.spawn_distribution(after)
            assert abs(sum(p for _, p in dist) - 1) < 1e-12
            assert {g.tile_sum(b) - g.tile_sum(after) for b, _ in dist} <= {2, 4}
        _, after, _ = rng.choice(g.moves(board))
        if g.is_win(after):
            break
        board = g.sample_spawn(after, rng)


def _mirror(board, rows, cols):
    return tuple(board[r * cols + (cols - 1 - c)]
                 for r in range(rows) for c in range(cols))


def _flip(board, rows, cols):
    return tuple(board[(rows - 1 - r) * cols + c]
                 for r in range(rows) for c in range(cols))


def _transpose(board, n):
    return tuple(board[c * n + r] for r in range(n) for c in range(n))


def test_symmetry():
    """Optimal values are invariant under the symmetries of the board."""
    for rows, cols, target in ((2, 2, 32), (2, 3, 64)):
        V, _, _ = dp.solve_optimal(Game2048(rows, cols, target=target))
        for board, v in V.items():
            images = [_mirror(board, rows, cols), _flip(board, rows, cols)]
            if rows == cols:
                images.append(_transpose(board, rows))
            for image in images:
                assert abs(V[image] - v) < 1e-12


def test_known_values():
    """Values obtained with an independent recursive implementation."""
    expected = {  # (rows, cols, target): (states, optimal, corner, random)
        (2, 2, 16): (214, 0.9625, 0.2355, 0.1218),
        (2, 3, 64): (27022, 0.9393, 0.0007, 0.0002),
    }
    for (rows, cols, target), (n, opt, corner, rnd) in expected.items():
        row = dp.compare_policies(Game2048(rows, cols, target=target))
        assert row['states'] == n
        for key, value in (('optimal', opt), ('corner', corner),
                           ('random', rnd)):
            assert abs(row[key] - value) < 5e-5, (key, row[key], value)


def test_policy_evaluation_matches_optimum():
    g = Game2048(2, 3, target=32)
    V, table, layers = dp.solve_optimal(g)
    V_pi = dp.evaluate_policy(g, dp.table_policy(table), layers=layers)
    assert max(abs(V[b] - V_pi[b]) for b in V) < 1e-12


def test_monte_carlo_agrees_with_dp():
    g = Game2048(2, 3, target=64)
    V, table, _ = dp.solve_optimal(g)
    p = dp.start_value(g, V)
    n = 5000
    wins = sum(r['won'] for r in
               dp.simulate(g, dp.table_policy(table), n, seed=0))
    assert abs(wins / n - p) < 4 * (p * (1 - p) / n) ** 0.5


def test_score_objective():
    g = Game2048(2, 2, target=8)  # the target is ignored for 'score'
    V, table, layers = dp.solve_optimal(g, objective='score')
    best = dp.start_value(g, V)
    for policy in dp.HEURISTICS.values():
        value = dp.start_value(g, dp.evaluate_policy(g, policy, 'score', layers))
        assert value <= best + 1e-9
    res = dp.simulate(g, dp.table_policy(table), 4000, seed=1,
                      objective='score')
    mean = sum(r['score'] for r in res) / len(res)
    var = sum((r['score'] - mean) ** 2 for r in res) / (len(res) - 1)
    assert abs(mean - best) < 4 * (var / len(res)) ** 0.5


def test_env():
    env = Env2048(Game2048(2, 3, target=32), seed=3)
    rng = random.Random(0)
    for _ in range(50):
        env.reset()
        done, total = False, 0.0
        while not done:
            _, reward, done, _, info = env.step(rng.choice(env.legal_actions()))
            total += reward
        assert total in (0.0, 1.0) and info['won'] == (total == 1.0)
    # an action that does not change the board must be rejected
    env.reset(seed=5)
    illegal = [a for a in ACTIONS if a not in env.legal_actions()]
    if illegal:
        try:
            env.step(illegal[0])
        except ValueError:
            pass
        else:
            raise AssertionError('illegal action accepted')


def test_viz():
    import matplotlib
    matplotlib.use('Agg')  # no window
    from viz2048 import animate, caption, play_episode
    g = Game2048(2, 3, target=32)
    V, table, _ = dp.solve_optimal(g)
    frames = play_episode(g, dp.table_policy(table), seed=0, V=V)
    assert frames[0]['phase'] == 'start'
    assert frames[-1]['phase'] in ('won', 'lost')
    for prev, cur in zip(frames, frames[1:]):
        if cur['phase'] in ('slide', 'won'):    # slide/merge keeps the sum
            assert g.tile_sum(cur['board']) == g.tile_sum(prev['board'])
        elif cur['phase'] == 'spawn':           # the new tile adds 2 or 4
            assert g.tile_sum(cur['board']) - g.tile_sum(prev['board']) in (2, 4)
    assert all(0.0 <= f['value'] <= 1.0 for f in frames)
    assert all(caption(g, f) for f in frames)
    assert animate(g, frames[:5]).to_jshtml().count('data:image/png') == 5


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items())
             if name.startswith('test_')]
    for test in tests:
        t0 = time.perf_counter()
        test()
        print(f'ok  {test.__name__}  ({time.perf_counter() - t0:.1f} s)')
    print('all tests passed')
