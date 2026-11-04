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


# ----------------------------------------------------------------------
# fast2048: the bitboard engine must reproduce game2048 exactly
# ----------------------------------------------------------------------
def test_fast_tables():
    """Every row table entry agrees with slide_line."""
    import fast2048 as fb
    for w in (2, 3, 4):
        for row in range(65536):
            cells = [(row >> (4 * c)) & 0xF for c in range(4)]
            if any(cells[w:]):
                continue
            line = tuple(cells[:w])
            moved, p = slide_line(line)
            if max(moved) > 15:
                continue  # overflow guard, not reachable in a game
            got = fb.LEFT[w - 2, row]
            assert tuple((int(got) >> (4 * c)) & 0xF for c in range(w)) == moved
            assert fb.PTS_LEFT[w - 2, row] == p
            moved, p = slide_line(line[::-1])
            moved = moved[::-1]
            got = fb.RIGHT[w - 2, row]
            assert tuple((int(got) >> (4 * c)) & 0xF for c in range(w)) == moved
            assert fb.PTS_RIGHT[w - 2, row] == p


def test_fast_transpose():
    import fast2048 as fb
    rng = random.Random(0)
    for _ in range(2000):
        board = tuple(rng.randrange(16) for _ in range(16))
        x = fb.encode(board, 4, 4)
        assert fb.decode(x, 4, 4) == board
        assert fb.decode(fb.transpose(x), 4, 4) == _transpose(board, 4)


def test_fast_moves_match_game():
    """Random games: every move, legal set, terminal and win flag agree."""
    import fast2048 as fb
    rng = random.Random(7)
    for rows, cols, target in ((2, 3, 64), (2, 4, 64), (3, 3, 64),
                               (4, 4, 2048)):
        g = Game2048(rows, cols, target=target)
        for _ in range(60):
            board = g.sample_initial(rng)
            while True:
                x = fb.encode(board, rows, cols)
                assert fb.decode(x, rows, cols) == board
                assert fb.tile_sum(x, rows, cols) == g.tile_sum(board)
                assert (1 << fb.max_exp(x, rows, cols)) == max(g.max_tile(board), 1)
                assert fb.count_empty(x, rows, cols) == board.count(0)
                assert fb.is_win(x, rows, cols, g.target_exp) == g.is_win(board)
                for a, action in enumerate(ACTIONS):
                    after, pts, changed = fb.move(x, a, rows, cols)
                    ref_after, ref_pts = g.afterstate(board, action)
                    assert fb.decode(after, rows, cols) == ref_after, (board, action)
                    assert pts == ref_pts
                    assert changed == (ref_after != board)
                assert fb.legal_actions(x, rows, cols) == g.legal_actions(board)
                assert fb.is_terminal(x, rows, cols) == g.is_game_over(board)
                moves = g.moves(board)
                if not moves:
                    break
                _, after, _ = rng.choice(moves)
                if g.is_win(after):
                    break
                board = g.sample_spawn(after, rng)


def _check_frequencies(counts, expected, n):
    """counts: {outcome: count}; expected: {outcome: probability}."""
    assert set(counts) <= set(expected), set(counts) - set(expected)
    for outcome, p in expected.items():
        freq = counts.get(outcome, 0) / n
        assert abs(freq - p) < 4 * (p * (1 - p) / n) ** 0.5 + 1e-12, (outcome, freq, p)


def test_fast_spawn_distribution():
    import fast2048 as fb
    from collections import Counter
    fb.seed(1)
    g = Game2048(2, 3, target=64, p4=0.1)
    after = (2, 0, 0, 3, 0, 0)
    x = fb.encode(after, 2, 3)
    n = 30000
    counts = Counter(fb.decode(fb.spawn(x, 2, 3, g.p4), 2, 3) for _ in range(n))
    _check_frequencies(counts, dict(g.spawn_distribution(after)), n)
    g = Game2048(2, 2, target=16, p4=0.3)
    counts = Counter(fb.decode(fb.initial_board(2, 2, g.p4), 2, 2) for _ in range(n))
    _check_frequencies(counts, dict(g.initial_distribution()), n)


def test_fast_env_step():
    import fast2048 as fb
    fb.seed(2)
    rows, cols, K = 2, 3, 5  # target 32
    for mode in (0, 1):
        for _ in range(200):
            x = fb.initial_board(rows, cols, 0.1)
            total, points = 0.0, 0
            while True:
                mask = fb.legal_mask(x, rows, cols)
                legal = [a for a in range(4) if mask & (1 << a)]
                if not legal:
                    break
                a = random.choice(legal)
                after_ref, pts_ref, _ = fb.move(x, a, rows, cols)
                nxt, reward, terminal, after, created = fb.env_step(
                    x, a, rows, cols, K, 0.1, mode)
                assert after == after_ref
                assert created == fb.is_win(after, rows, cols, K)
                total += reward
                points += pts_ref
                if mode == 0:
                    assert reward == (1.0 if created else 0.0)
                    if created:
                        assert terminal and nxt == after
                        break
                else:
                    assert reward == pts_ref
                assert fb.tile_sum(nxt, rows, cols) - fb.tile_sum(after, rows, cols) in (2, 4)
                assert terminal == fb.is_terminal(nxt, rows, cols)
                x = nxt
                if terminal:
                    break
            if mode == 1:
                assert total == points


def test_fast_random_play_matches_exact_values():
    """Random play on 2x3: win rate and mean score vs the exact values."""
    import fast2048 as fb
    fb.seed(3)
    n = 20000
    g = Game2048(2, 3, target=32)
    p_exact = dp.start_value(g, dp.evaluate_policy(g, dp.random_policy))
    _, _, _, won = fb.play_random(2, 3, g.target_exp, g.p4, n, True)
    p = won.mean()
    assert abs(p - p_exact) < 4 * (p_exact * (1 - p_exact) / n) ** 0.5, (p, p_exact)
    g = Game2048(2, 3, target=64)
    s_exact = dp.start_value(g, dp.evaluate_policy(g, dp.random_policy, 'score'))
    scores, _, _, _ = fb.play_random(2, 3, g.target_exp, g.p4, n, False)
    assert abs(scores.mean() - s_exact) < 4 * scores.std(ddof=1) / n ** 0.5


# ----------------------------------------------------------------------
# rl2048: tabular agents learn the 2x2 game from the simulator
# ----------------------------------------------------------------------
def test_tabular_rl_converges_on_2x2():
    import rl2048 as rl
    g = Game2048(2, 2, target=16)
    ev = rl.ExactEvaluator(g)
    optimum = dp.start_value(g, ev.V_win)
    for method in rl.METHODS:
        env = Env2048(g, reward='win', seed=0)
        agent, hist = rl.train(method, env, 20000, alpha=0.05, epsilon=0.1,
                               q_init=0.5, seed=1, eval_every=20000,
                               evaluator=ev)
        p = hist[-1]['p_win']
        assert p > optimum - 0.05, (method, p, optimum)
        assert hist[-1]['unseen'] < 0.01, (method, hist[-1]['unseen'])
    # the score objective: the exact score of the learned policy is close
    # to the maximum expected score
    V_s, _, _ = dp.solve_optimal(g, 'score')
    best = dp.start_value(g, V_s)
    env = Env2048(g, reward='score', seed=0)
    agent, hist = rl.train('td_afterstate', env, 20000, alpha=0.05,
                           epsilon=0.1, seed=2, eval_every=20000, evaluator=ev)
    assert hist[-1]['score'] > 0.95 * best, (hist[-1]['score'], best)


# ----------------------------------------------------------------------
# ntuple2048: TD(0) on afterstates learns the 2x3 game
# ----------------------------------------------------------------------
def test_ntuple_small_board():
    import numpy as np
    import ntuple2048 as nt
    import rl2048 as rl
    g = Game2048(2, 3, target=64)
    ev = rl.ExactEvaluator(g, with_score=False)
    optimum = dp.start_value(g, ev.V_win)
    # the full-board tuple is a lookup table of the afterstates: with the
    # optimistic initialisation the greedy learner approaches the optimum
    net = nt.NTupleNet(2, 3, 64, tuples='full', symmetric=False,
                       dtype=np.float64, init=1.0)
    hist = net.train(100_000, alpha=0.1, reward='win', seed=1, block=10_000,
                     evaluator=nt.exact_evaluator(ev, g, 'win'))
    p = hist[-1]['p_win']
    assert p > optimum - 0.1, (p, optimum)
    assert hist[-1]['rmse_w'] < 0.05, hist[-1]['rmse_w']
    # the value of a winning afterstate is 1 (on the trained optimistic net
    # an unvisited afterstate may still be worth 1 and tie with the win, so
    # the action is checked on a fresh zero-initialised net) and the learned
    # policy can be used by the pure-Python side
    assert net.greedy((5, 5, 0, 1, 0, 0), 'win')[1] == 1.0
    fresh = nt.NTupleNet(2, 3, 64, tuples='full', symmetric=False,
                         dtype=np.float64)
    assert fresh.greedy((5, 5, 0, 1, 0, 0), 'win') == ('left', 1.0)
    frames = __import__('viz2048').play_episode(g, net.policy('win'), seed=0)
    assert frames[-1]['phase'] in ('won', 'lost')


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items())
             if name.startswith('test_')]
    for test in tests:
        t0 = time.perf_counter()
        test()
        print(f'ok  {test.__name__}  ({time.perf_counter() - t0:.1f} s)')
    print('all tests passed')
