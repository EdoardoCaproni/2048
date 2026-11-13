# shaped_rewards.py
#
# Our first idea for a reward, before we settled on 'win' and 'score':
# pay the agent for the things a good 2048 player tries to do.
#
#   milestone : when the move makes a new biggest tile, a bonus equal to
#               that tile (2**k). The biggest tile can only go up, so this is
#               paid at most once for each power of two.
#   sum       : the tile sum goes up by 2 or 4 after every move (the slide
#               doesn't change it, the new tile adds 2 or 4), so we pay its
#               expected growth 2*(1-p4) + 4*p4 = 2 + 2*p4.
#   free      : +1 for every cell freed by the move (one per merge) and -1
#               for the cell the new tile fills, i.e. free(y) - free(x) - 1.
#
# Boards are tuples of exponents like in game.py (0 = empty, k = tile 2**k).
# A reward is a function r(board, after, points) where board is x and after is
# the afterstate y = M(x, u). It only depends on the move (x, u), like the
# points do, so the DP and the afterstate agents can use it as it is.
#
# What we found when we added the terms up over a whole game
# (checked with experiments/exp_reward.py):
#   - 'sum' is the same for every move, so it only rewards surviving;
#   - 'free' adds up to free(last board) - free(first board), and the last
#     board is full, so it's a constant: it can't change the best policy;
#   - only 'milestone' really matters, and it adds up to about twice the
#     biggest tile reached, so it's like a graded version of the win.
#
# Reward names (they can be used everywhere 'win' and 'score' can):
#   naive         = milestone + sum + free
#   naive_nofree  = milestone + sum
#   milestone, sum, free : one term alone

REWARDS = ['naive', 'naive_nofree', 'milestone', 'sum', 'free']


def count_free(board):
    n = 0
    for cell in board:
        if cell == 0:
            n += 1
    return n


def milestone(board, after):
    if max(after) > max(board):
        return 2 ** max(after)
    return 0


def free_change(board, after):
    return count_free(after) - count_free(board) - 1


def reward_fn(name, p4):
    """Returns r(board, after, points) for 'score' or one of REWARDS."""
    if name == 'score':
        return lambda board, after, points: points
    if name not in REWARDS:
        raise ValueError('unknown reward: ' + repr(name))

    def r(board, after, points):
        value = 0.0
        if name in ('naive', 'naive_nofree', 'sum'):
            value += 2 + 2 * p4
        if name in ('naive', 'naive_nofree', 'milestone'):
            value += milestone(board, after)
        if name in ('naive', 'free'):
            value += free_change(board, after)
        return value

    return r
