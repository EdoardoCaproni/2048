"""Rewards built from the rules of thumb of the players ("shaped" rewards).

The first idea for a reward was to pay the agent for what good players
try to do: keep cells free, make the tile sum grow, and create ever larger
tiles. Every component below depends only on the move, r(x, u), through
the board x and the afterstate y = M(x, u), so it fits the DP recursion and
the afterstate parametrisation q(x, u) = r(x, u) + V(y) exactly like the
points of the move.

Components (exponents as in game2048: 0 = empty, k = tile 2**k)
----------
milestone  2**max(y) if the move creates a new largest tile (max(y) >
           max(x)), else 0. The largest tile never decreases, so this is
           paid once for each power of two that becomes the largest tile.
sum        2 + 2 p4: the expected growth of the tile sum in one move. The
           slide keeps the sum and the random tile adds 2 or 4, whatever
           the move, so this term pays survival and nothing else.
free       free(y) - free(x) - 1 = merges - 1: the change of the number of
           free cells over the whole move (the random tile fills one).
           Over a game it adds up to free(final) - free(initial) = a
           constant, because the board is full when the game is over.

Named rewards (usable wherever 'win' or 'score' are accepted):
    naive          milestone + sum + free
    naive_nofree   milestone + sum
    milestone, sum, free   each component alone
"""

REWARDS = {
    'naive': {'milestone': 1.0, 'sum': 1.0, 'free': 1.0},
    'naive_nofree': {'milestone': 1.0, 'sum': 1.0},
    'milestone': {'milestone': 1.0},
    'sum': {'sum': 1.0},
    'free': {'free': 1.0},
}


def _free(board):
    return sum(1 for e in board if e == 0)


def reward_fn(name, p4):
    """r(board, afterstate, points) for a named reward ('score' or a key of
    REWARDS)."""
    if name == 'score':
        return lambda board, after, points: points
    if name not in REWARDS:
        raise ValueError(f"unknown reward {name!r}")
    w = REWARDS[name]
    w_m = w.get('milestone', 0.0)
    w_s = w.get('sum', 0.0) * (2.0 + 2.0 * p4)
    w_f = w.get('free', 0.0)

    def r(board, after, points):
        value = w_s
        if w_m:
            m = max(after)
            if m > max(board):
                value += w_m * 2 ** m
        if w_f:
            value += w_f * (_free(after) - _free(board) - 1)
        return value
    return r
