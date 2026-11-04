"""Exact dynamic programming for 2048 on small boards.

Key property (see the formulation): the slide/merge keeps the tile sum
sigma(x) unchanged and the random tile adds 2 or 4, so sigma strictly grows
at every move. The reachable states split into stages X_sigma, every
transition goes from X_sigma to X_{sigma+2} or X_{sigma+4}, and the Bellman
equation is solved exactly by one backward recursion over the stages, as in
a finite-horizon problem (no value iteration is needed).

Objectives
----------
'win'   : V(x) = maximum probability of creating the target tile from x.
'score' : V(x) = maximum expected score (sum of the merged tiles) until the
          game is over; the target tile is ignored.

Policies are functions policy(board, moves) -> {action: probability}, where
moves = game.moves(board) is the list of legal (action, afterstate, points).
They return an empty dict when the game is over.
"""
import random
import time
from collections import defaultdict

OBJECTIVES = ('win', 'score')
# Relative tolerance that detects exact ties between actions (values equal up
# to floating-point rounding). With a larger tolerance, near-ties count as
# ties too: on boards where the optimal P(win) is close to 1, the share of
# "optimal" moves of a heuristic depends on this choice.
TIE_TOL = 1e-12


def _check(objective):
    if objective not in OBJECTIVES:
        raise ValueError(f'objective must be one of {OBJECTIVES}')


def _is_tie(a, b):
    return abs(a - b) <= TIE_TOL * max(1.0, abs(a), abs(b))


# ----------------------------------------------------------------------
# State space
# ----------------------------------------------------------------------
def reachable_layers(game, stop_at_target=True):
    """Boards reachable from the initial distribution under some policy,
    grouped by tile sum: {sigma: set of boards}.

    With stop_at_target=True the boards that would come after the creation
    of the target tile are not generated, since the game ends there.
    """
    layers = defaultdict(set)
    for board, _ in game.initial_distribution():
        layers[game.tile_sum(board)].add(board)
    sigma = min(layers)
    while sigma <= max(layers):
        for board in layers.get(sigma, ()):
            for _, after, _ in game.moves(board):
                if stop_at_target and game.is_win(after):
                    continue
                # the new tile adds exponent 1 (a 2) or 2 (a 4), i.e. the
                # tile sum goes from sigma to sigma + 2 or sigma + 4
                base = sum(after)
                for nxt, _ in game.spawn_distribution(after):
                    layers[sigma + 2 ** (sum(nxt) - base)].add(nxt)
        sigma += 2
    return dict(layers)


def count_states(layers):
    return sum(len(boards) for boards in layers.values())


def _default_layers(game, objective, layers):
    _check(objective)
    if layers is None:
        layers = reachable_layers(game, stop_at_target=(objective == 'win'))
    return layers


# ----------------------------------------------------------------------
# Bellman recursion
# ----------------------------------------------------------------------
def afterstate_value(game, after, V, objective='win'):
    """W(y): value of an afterstate y, before the random tile appears."""
    if objective == 'win' and game.is_win(after):
        return 1.0
    return sum(p * V[nxt] for nxt, p in game.spawn_distribution(after))


def solve_optimal(game, objective='win', layers=None):
    """Optimal values and policy by backward recursion over the stages.

    Returns (V, table, layers): V[x] is the optimal value of board x and
    table[x] an optimal action (ties broken by the order of ACTIONS).
    """
    layers = _default_layers(game, objective, layers)
    V, table = {}, {}
    for sigma in sorted(layers, reverse=True):
        W = {}  # afterstates of the boards of stage sigma have tile sum sigma
        for board in layers[sigma]:
            best_action, best_value = None, 0.0  # game over: value 0
            for action, after, points in game.moves(board):
                w = W.get(after)
                if w is None:
                    w = W[after] = afterstate_value(game, after, V, objective)
                q = w + points if objective == 'score' else w
                if best_action is None or (q > best_value and
                                           not _is_tie(q, best_value)):
                    best_action, best_value = action, q
            V[board] = best_value
            if best_action is not None:
                table[board] = best_action
    return V, table, layers


def evaluate_policy(game, policy, objective='win', layers=None):
    """Exact value of a fixed policy: the same backward recursion, with the
    policy's choice in place of the maximisation (policy evaluation)."""
    layers = _default_layers(game, objective, layers)
    V = {}
    for sigma in sorted(layers, reverse=True):
        W = {}
        for board in layers[sigma]:
            moves = game.moves(board)
            effect = {action: (after, points) for action, after, points in moves}
            value = 0.0
            for action, prob in policy(board, moves).items():
                after, points = effect[action]
                w = W.get(after)
                if w is None:
                    w = W[after] = afterstate_value(game, after, V, objective)
                value += prob * (w + points if objective == 'score' else w)
            V[board] = value
    return V


def start_value(game, V):
    """Value at the start of a game, averaged over the two initial tiles."""
    return sum(p * V[board] for board, p in game.initial_distribution())


def action_values(game, board, V, objective='win'):
    """Q-values {action: value} of the legal moves of a board."""
    q = {}
    for action, after, points in game.moves(board):
        q[action] = afterstate_value(game, after, V, objective)
        if objective == 'score':
            q[action] += points
    return q


def optimal_actions(game, board, V, objective='win'):
    """All the actions that achieve V[board] (there can be ties)."""
    return [a for a, q in action_values(game, board, V, objective).items()
            if _is_tie(q, V[board]) or q > V[board]]


# ----------------------------------------------------------------------
# Benchmark policies
# ----------------------------------------------------------------------
def corner_policy(board, moves):
    """Fixed priority down > left > right > up: the classic rule 'keep the
    big tiles in the bottom-left corner'."""
    return {moves[0][0]: 1.0} if moves else {}


def greedy_policy(board, moves):
    """Move with the largest immediate merge score (ties: ACTIONS order)."""
    if not moves:
        return {}
    return {max(moves, key=lambda m: m[2])[0]: 1.0}


def random_policy(board, moves):
    """Uniformly random legal move."""
    return {action: 1.0 / len(moves) for action, _, _ in moves}


def table_policy(table):
    """Deterministic policy from a {board: action} table (e.g. the optimal
    one returned by solve_optimal)."""
    def policy(board, moves):
        return {table[board]: 1.0} if moves else {}
    return policy


HEURISTICS = {'corner': corner_policy, 'greedy': greedy_policy,
              'random': random_policy}


def compare_policies(game, objective='win'):
    """Exact value at the start for the optimal policy and the heuristics,
    with the number of reachable states and the DP time in seconds."""
    t0 = time.perf_counter()
    V, table, layers = solve_optimal(game, objective)
    row = {'board': f'{game.rows}x{game.cols}', 'target': game.target,
           'p4': game.p4, 'states': count_states(layers),
           'dp_time_s': time.perf_counter() - t0,
           'optimal': start_value(game, V)}
    for name, policy in HEURISTICS.items():
        row[name] = start_value(game, evaluate_policy(game, policy,
                                                      objective, layers))
    return row


# ----------------------------------------------------------------------
# Analysis of the optimal policy
# ----------------------------------------------------------------------
def occupancy(game, policy, layers, objective='win'):
    """Expected number of visits of each board under a policy: a forward
    pass over the stages in increasing tile sum."""
    stop = objective == 'win'
    mu = defaultdict(float)
    for board, p in game.initial_distribution():
        mu[board] += p
    for sigma in sorted(layers):
        for board in layers[sigma]:
            m = mu.get(board, 0.0)
            if m == 0.0:
                continue
            moves = game.moves(board)
            after_of = {action: after for action, after, _ in moves}
            for action, prob in policy(board, moves).items():
                after = after_of[action]
                if stop and game.is_win(after):
                    continue
                for nxt, p in game.spawn_distribution(after):
                    mu[nxt] += m * prob * p
    return dict(mu)


def compare_with_heuristic(game, V, table, layers, heuristic,
                           objective='win'):
    """Where does a deterministic heuristic disagree with the optimal policy?

    Decisions are weighted by their expected number of visits under the
    optimal policy. Returns a dict with
      share          fraction of decisions where the heuristic's move is optimal,
      expected_moves expected number of moves of a game played optimally,
      exceptions     list of (importance, board, heuristic action, optimal
                     action, value loss), sorted by importance = visits x loss.
    """
    mu = occupancy(game, table_policy(table), layers, objective)
    total = agree = 0.0
    exceptions = []
    for board, visits in mu.items():
        moves = game.moves(board)
        if not moves:
            continue
        (h_action,) = heuristic(board, moves)
        q_h = action_values(game, board, V, objective)[h_action]
        loss = V[board] - q_h
        total += visits
        if _is_tie(q_h, V[board]) or loss <= 0:
            agree += visits
        else:
            exceptions.append((visits * loss, board, h_action,
                               table[board], loss))
    exceptions.sort(key=lambda e: e[0], reverse=True)
    return {'share': agree / total, 'expected_moves': total,
            'exceptions': exceptions}


# ----------------------------------------------------------------------
# Monte Carlo simulation
# ----------------------------------------------------------------------
def simulate(game, policy, n_games, seed=0, objective='win'):
    """Play n_games with a policy. Returns one dict per game with
    won (target created), score, moves and max_tile."""
    rng = random.Random(seed)
    stop = objective == 'win'
    results = []
    for _ in range(n_games):
        board = game.sample_initial(rng)
        score = n_moves = 0
        won = False
        while True:
            moves = game.moves(board)
            dist = policy(board, moves)
            if not dist:
                break
            actions = list(dist)
            action = rng.choices(actions, [dist[a] for a in actions])[0]
            after, points = next((af, pt) for a, af, pt in moves
                                 if a == action)
            score += points
            n_moves += 1
            if game.is_win(after):
                won = True
                if stop:
                    board = after
                    break
            board = game.sample_spawn(after, rng)
        results.append({'won': won, 'score': score, 'moves': n_moves,
                        'max_tile': game.max_tile(board)})
    return results
