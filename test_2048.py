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


def test_shaped_rewards():
    """The free-cell term adds up to -free(x) over any game: it shifts every
    value by a constant per state and leaves the optimal policy unchanged."""
    from shaping2048 import reward_fn
    g = Game2048(2, 3, target=32)
    V0, table0, layers = dp.solve_optimal(g, 'naive_nofree')
    V1, table1, _ = dp.solve_optimal(g, 'naive', layers)
    assert table0 == table1
    for board, v in V0.items():
        assert abs(V1[board] - (v - board.count(0))) < 1e-9
    # the environment pays the same reward as the DP, and the free-cell
    # term of a whole episode is -free(initial board)
    env = Env2048(g, reward='free', seed=2)
    r = reward_fn('naive', g.p4)
    rng = random.Random(0)
    for _ in range(20):
        board, _ = env.reset()
        start_free, total, done = board.count(0), 0.0, False
        while not done:
            board, reward, done, _, info = env.step(rng.choice(env.legal_actions()))
            total += reward
        assert total == -start_free
    env = Env2048(g, reward='naive', seed=4)
    board, _ = env.reset()
    done = False
    while not done:
        action = rng.choice(env.legal_actions())
        after, points = g.afterstate(board, action)
        expected = r(board, after, points)
        board, reward, done, _, _ = env.step(action)
        assert reward == expected


def test_score_value_unchanged():
    g = Game2048(2, 3, target=64)
    V, _, _ = dp.solve_optimal(g, 'score')
    assert abs(dp.start_value(g, V) - 480.3) < 0.05


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
    agent, hist = rl.train('afterstate_sarsa', env, 20000, alpha=0.05,
                           epsilon=0.1, seed=2, eval_every=20000, evaluator=ev)
    assert hist[-1]['score'] > 0.95 * best, (hist[-1]['score'], best)


# ----------------------------------------------------------------------
# dp2048: value iteration reaches the backward recursion exactly
# ----------------------------------------------------------------------
def test_value_iteration_matches_backward_recursion():
    for r, c, target, objective in ((2, 2, 16, 'win'), (2, 2, 32, 'win'),
                                    (2, 2, 16, 'score')):
        g = Game2048(r, c, target=target)
        V, _, layers = dp.solve_optimal(g, objective)
        V_vi, table, hist = dp.value_iteration(g, objective, layers, V_ref=V)
        assert all(V_vi[b] == V[b] for b in V), (r, c, target, objective)
        assert hist[-1]['delta'] == 0.0 and hist[-1]['error'] == 0.0
        # at most one iteration per stage, plus the confirming one
        assert len(hist) <= len(layers) + 1, (len(hist), len(layers))
        # the start value increases with the horizon (P(win within i moves))
        starts = [h['start'] for h in hist]
        assert all(a <= b + 1e-12 for a, b in zip(starts, starts[1:]))
        # Gauss-Seidel in decreasing tile sum: exact after one sweep
        _, _, hist_gs = dp.value_iteration(g, objective, layers,
                                           order='backward', V_ref=V)
        assert len(hist_gs) == 2 and hist_gs[0]['error'] == 0.0
        # the greedy policy of value iteration is optimal
        V_pol = dp.evaluate_policy(g, dp.table_policy(table), objective, layers)
        assert abs(dp.start_value(g, V_pol) - dp.start_value(g, V)) < 1e-12


# ----------------------------------------------------------------------
# predict2048: TD(0), n-step TD and Monte Carlo estimate v_pi
# ----------------------------------------------------------------------
def test_prediction_converges_on_2x2():
    import numpy as np
    import predict2048 as pr
    g = Game2048(2, 2, target=16)
    V, table, layers = dp.solve_optimal(g)
    prob = pr.Problem(g, table, layers=layers)
    assert abs(prob.mu.sum() - 1.0) < 1e-12
    S, R, starts = prob.episodes(20_000, seed=3)
    # the episodes follow the policy: their win frequency matches V*
    wins = R[starts[1:] - 1].mean()
    assert abs(wins - dp.start_value(g, V)) < 0.01, wins
    # the boards of the episodes are the boards of the DP
    assert all(prob.codes[i] == __import__('fast2048').encode(prob.boards[i], 2, 2)
               for i in np.unique(S)[:50])
    for n, alpha in ((1, 0.1), (3, 0.1), (pr.MC_N, 0.05)):
        Vh, rms = pr.nstep_td(S, R, starts, n, alpha, 0.5, prob.v_true, prob.mu)
        assert abs(rms[-1] - pr.rms_error(Vh, prob)) < 1e-9
        assert rms[-1] < 0.05 and rms[-1] < rms[0] / 5, (n, rms[0], rms[-1])


# ----------------------------------------------------------------------
# dqn2048: the DQN of the final test runs and plays legal moves
# ----------------------------------------------------------------------
def test_dqn_smoke():
    try:
        import torch
    except ImportError:
        print('     (torch not installed: DQN test skipped)')
        return
    import numpy as np
    import dqn2048 as dq
    import rl2048 as rl
    torch.set_num_threads(1)
    g = Game2048(2, 2, target=16)
    x = dq.encode([(1, 0, 3, 2)])
    assert x.shape == (1, 4 * dq.N_VALUES) and x.sum() == 4
    assert x[0, 1] == 1 and x[0, dq.N_VALUES + 0] == 1 and x[0, 3 * dq.N_VALUES + 2] == 1
    agent = dq.DQNAgent(g, seed=0)
    hist = dq.train(agent, 300, eval_every=300,
                    evaluator=dq.exact_evaluator(rl.ExactEvaluator(g)))
    assert agent.steps_done > dq.BATCH_SIZE and np.isfinite(hist[-1]['loss'])
    assert 0.0 <= hist[-1]['p_win'] <= 1.0 and hist[-1]['unseen'] == 0.0
    # greedy actions are legal, and -1 only at game over
    boards = [(1, 1, 2, 3), (1, 2, 3, 4), (0, 0, 1, 1)]
    actions, _ = agent.greedy_batch(boards)
    for b, a in zip(boards, actions):
        legal = g.legal_actions(b)
        assert (a == -1) == (not legal) and (a == -1 or ACTIONS[a] in legal)


# ----------------------------------------------------------------------
# ntuple2048: semi-gradient Sarsa with n-tuple features learns the 2x3 game
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


def test_fast_tabular():
    """rlfast2048: the six tabular methods on the bitboard engine learn on
    2x3 as the reference code does (exact P(win) of the greedy policy, score
    reward, a table that starts small and doubles several times), and a
    table restricted to its most visited boards plays the same moves there."""
    import rl2048 as rl
    import rlfast2048 as rf
    g = Game2048(2, 3, target=64)
    ev = rl.ExactEvaluator(g, with_score=False)
    boards = [b for layer in ev.layers_win.values() for b in layer]
    for method in rf.METHODS:
        agent = rf.FastTabular(method, 2, 3, 64, seed=0, capacity_log2=10)
        agent.learn(100_000, None if method == 'mc_control' else 0.1, 0.1, seed=1)
        acts, known = agent.greedy_actions(boards)
        table = {b: a for b, a, k in zip(boards, acts, known) if k and a is not None}

        class Frozen:
            reward = None

            @staticmethod
            def greedy_action(board, moves):
                return table.get(board)

            @staticmethod
            def policy():
                return lambda board, moves: (
                    {} if not moves else {table[board]: 1.0} if board in table
                    else dp.corner_policy(board, moves))
        p = ev.evaluate(Frozen)['p_win']
        assert p > 0.5, (method, p)
    agent = rf.FastTabular('q_learning', 4, 4, 2048, seed=0)
    agent.learn(300, 0.1, 0.1, seed=2)
    res = agent.evaluate(20, seed=3)
    assert 0.0 < res['unseen'] < 1.0 and res['score'] > 0, res
    keys, vals, visits = agent.most_visited(1000)
    small = rf.FastTabular.from_entries(agent.config(), keys, vals, visits)
    import fast2048 as fb
    some = [fb.decode(k, 4, 4) for k in keys[:200]]
    assert small.greedy_actions(some)[0] == agent.greedy_actions(some)[0]


if __name__ == '__main__':
    tests = [f for name, f in sorted(globals().items())
             if name.startswith('test_')]
    for test in tests:
        t0 = time.perf_counter()
        test()
        print(f'ok  {test.__name__}  ({time.perf_counter() - t0:.1f} s)')
    print('all tests passed')
