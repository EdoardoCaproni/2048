"""Tabular prediction on small boards: TD(0), n-step TD and Monte Carlo.

The agent estimates the value v_pi of a fixed policy pi from simulated
episodes, without knowing the model (Sutton and Barto, Sec. 6.1-6.3 and
7.1). The exact v_pi is known from the DP (policy evaluation by backward
recursion), so the error of the estimates is measured exactly, as in the
random-walk examples of the course (Example 6.2, Figure 7.2 of the book).

n-step TD with n = 1 is TD(0); when n is at least the length of the
episode the n-step return is the full return and the method is
constant-alpha Monte Carlo (MC_N below). The error is the square root of
the prediction objective VE of Sec. 9.2: the squared error weighted by the
on-policy distribution mu of pi (expected visits per episode from
dynamic_programming.occupancy, normalised to sum 1).

States never repeat within an episode (the tile sum grows at every move),
so the updates of an episode can be applied once the episode is over, in
the order of the online algorithm, with the same result: the value
V(S_{t+n}) used to bootstrap at time t+n-1 is not changed earlier in the
same episode. The episodes are generated with the bitboard engine of
game_fast.py, which follows the same rules as game.py.
"""
import math

import numpy as np

import dynamic_programming as dp
import game_fast as fb
from game_fast import njit  # numba imported with the cache directory set
from game import ACTIONS

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
MC_N = 1_000_000  # an n larger than any episode: n-step TD = Monte Carlo


class Problem:
    """A prediction problem: a board, a deterministic policy given as a
    {board: action} table, and the exact values and on-policy weights.

    Attributes (one entry per decision state of the DP, sorted by the
    bitboard code so that a board is found with a binary search):
      codes    uint64 bitboards
      actions  action index of the policy (-1 on game-over boards)
      v_true   exact v_pi (win probability for objective 'win')
      mu       on-policy distribution (0 on game-over boards)
    """

    def __init__(self, game, table, objective='win', layers=None):
        if objective != 'win':
            raise ValueError("only the win objective is used for prediction")
        self.game = game
        self.layers = layers or dp.reachable_layers(game)
        policy = dp.table_policy(table)
        V = dp.evaluate_policy(game, policy, objective, self.layers)
        visits = dp.occupancy(game, policy, self.layers, objective)
        boards = [b for layer in self.layers.values() for b in layer]
        codes = np.array([fb.encode(b, game.rows, game.cols) for b in boards],
                         dtype=np.uint64)
        order = np.argsort(codes)
        self.boards = [boards[i] for i in order]
        self.codes = codes[order]
        self.actions = np.array([ACTION_INDEX[table[b]] if b in table else -1
                                 for b in self.boards], dtype=np.int64)
        self.v_true = np.array([V[b] for b in self.boards], dtype=np.float64)
        mu = np.array([visits.get(b, 0.0) if b in table else 0.0
                       for b in self.boards], dtype=np.float64)
        self.expected_length = float(mu.sum())
        self.mu = mu / mu.sum()
        self.target_exp = game.target.bit_length() - 1
        self.max_len = game.rows * game.cols * 2 ** (self.target_exp - 2)

    def episodes(self, n_episodes, seed):
        """n_episodes simulated with the policy: (states, rewards, starts),
        where episode e has the states states[starts[e]:starts[e+1]] (as
        indices into the arrays above) and rewards[k] is the reward that
        follows states[k]."""
        g = self.game
        return _generate(self.codes, self.actions, n_episodes, g.rows, g.cols,
                         self.target_exp, g.p4, self.max_len, seed)


@njit(cache=True)
def _generate(codes, actions, n_episodes, rows, cols, target_exp, p4,
              max_len, seed):
    fb.seed(seed)
    states = np.empty(n_episodes * max_len, dtype=np.int64)
    rewards = np.empty(n_episodes * max_len, dtype=np.float64)
    starts = np.empty(n_episodes + 1, dtype=np.int64)
    k = 0
    for e in range(n_episodes):
        starts[e] = k
        x = fb.initial_board(rows, cols, p4)
        while True:
            i = np.searchsorted(codes, x)
            states[k] = i
            # reward mode 0: reward 1 when the move creates the target
            nxt, r, terminal, _, _ = fb.env_step(x, actions[i], rows, cols,
                                                 target_exp, p4, 0)
            rewards[k] = r
            k += 1
            if terminal:
                break
            x = nxt
    starts[n_episodes] = k
    return states[:k].copy(), rewards[:k].copy(), starts


@njit(cache=True)
def nstep_td(states, rewards, starts, n, alpha, v0, v_true, mu):
    """n-step TD prediction (Sec. 7.1) with gamma = 1 and a constant step
    size; n = 1 is TD(0), n = MC_N is constant-alpha Monte Carlo.
    Returns the final estimates V and the error sqrt(VE) after every
    episode (rms[0] is the error of the initial values)."""
    V = np.full(v_true.shape[0], v0)
    se = 0.0
    for s in range(V.shape[0]):
        se += mu[s] * (V[s] - v_true[s]) ** 2
    n_ep = starts.shape[0] - 1
    rms = np.empty(n_ep + 1)
    rms[0] = math.sqrt(se)
    for e in range(n_ep):
        a = starts[e]
        T = starts[e + 1] - a
        for tau in range(T):
            end = min(tau + n, T)
            G = 0.0
            for j in range(tau, end):
                G += rewards[a + j]
            if tau + n < T:
                G += V[states[a + tau + n]]
            s = states[a + tau]
            old = V[s]
            new = old + alpha * (G - old)
            se += mu[s] * ((new - v_true[s]) ** 2 - (old - v_true[s]) ** 2)
            V[s] = new
        rms[e + 1] = math.sqrt(max(se, 0.0))
    return V, rms


def rms_error(V, problem):
    """sqrt(VE) of the estimates V, computed from scratch."""
    return float(np.sqrt(np.sum(problem.mu * (V - problem.v_true) ** 2)))
