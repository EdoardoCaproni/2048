"""Tabular reinforcement learning for 2048 on small boards.

The agents interact only with the simulator Env2048 (reset/step): they do
not know the spawn distribution. The methods are the ones of Sutton and
Barto, "Reinforcement Learning: An Introduction" (2nd ed.):

  mc_control        on-policy first-visit Monte Carlo control with an
                    epsilon-soft policy (Sec. 5.4)
  sarsa             Sarsa, on-policy TD control (Sec. 6.4)
  q_learning        Q-learning, off-policy TD control (Sec. 6.5)
  expected_sarsa    Expected Sarsa (Sec. 6.6)
  nstep_sarsa       n-step Sarsa (Sec. 7.2)
  afterstate_sarsa  Sarsa (Sec. 6.4) with the action values written as
                    q(x, u) = r(x, u) + V(M(x, u)): the pairs (x, u) that
                    lead to the same board after the slide/merge
                    y = M(x, u) (the "afterstate", before the random tile)
                    share one table entry V(y). The agent uses the
                    deterministic part of the rules (game.afterstate) to
                    find M(x, u) and r(x, u), but never sees the spawn
                    distribution. See AfterstateSarsa for why the Sarsa
                    update reduces to an update of V.

Because the tile sum strictly increases, every state is visited at most
once per episode: first-visit and every-visit Monte Carlo coincide, and
gamma = 1 is safe (every episode ends in a bounded number of moves).

Actions are indices 0..3 into game.ACTIONS; the greedy choice takes
the first maximiser in that order, like the DP, so that policies can be
compared exactly. Illegal actions are never selected (the environment
reports the legal ones) and have no effect on the estimates.

Exact validation
----------------
ExactEvaluator uses dynamic_programming to compute the *exact* value of the learned
greedy policy (policy evaluation by backward recursion), the error of the
learned values against V*, and how often the greedy action is optimal.
Learning curves made this way have no Monte Carlo noise.
"""
import math
import random
import time
from collections import defaultdict

import numpy as np

import dynamic_programming as dp
from game import ACTIONS

ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
METHODS = ('mc_control', 'sarsa', 'q_learning', 'expected_sarsa',
           'nstep_sarsa', 'afterstate_sarsa')


# ----------------------------------------------------------------------
# Schedules for alpha and epsilon (functions of the episode number)
# ----------------------------------------------------------------------
def constant(c):
    return lambda t: c


def linear(start, end, n_episodes):
    """Linear from start (episode 1) to end (episode n_episodes), then end."""
    def f(t):
        frac = min(1.0, (t - 1) / max(1, n_episodes - 1))
        return start + (end - start) * frac
    return f


def exponential(start, end, n_episodes):
    """Geometric from start to end over n_episodes episodes, then end."""
    rate = math.log(end / start) / max(1, n_episodes - 1)
    return lambda t: max(end, start * math.exp(rate * (t - 1)))


def inverse(a0, tau):
    """a0 / (1 + t / tau)."""
    return lambda t: a0 / (1.0 + t / tau)


def _schedule(x):
    return x if callable(x) else constant(float(x))


# ----------------------------------------------------------------------
# Action selection
# ----------------------------------------------------------------------
def greedy(q, legal):
    """First maximiser (in ACTIONS order) among the legal action indices."""
    best, best_q = legal[0], q[legal[0]]
    for a in legal[1:]:
        if q[a] > best_q:
            best, best_q = a, q[a]
    return best


def epsilon_greedy(q, legal, eps, rng):
    if eps > 0.0 and rng.random() < eps:
        return legal[rng.randrange(len(legal))]
    return greedy(q, legal)


def epsilon_greedy_probs(q, legal, eps):
    """{action: probability} of the epsilon-greedy policy on legal."""
    base = eps / len(legal)
    probs = {a: base for a in legal}
    probs[greedy(q, legal)] += 1.0 - eps
    return probs


def _legal(info):
    return [ACTION_INDEX[a] for a in info['legal_actions']]


# ----------------------------------------------------------------------
# Agents
# ----------------------------------------------------------------------
class TabularAgent:
    """Base class: a table Q[board] -> array of 4 action values."""
    name = 'base'

    def __init__(self, env, gamma=1.0, q_init=0.0, **kwargs):
        self.env = env
        self.game = env.game
        self.gamma = gamma
        self.q_init = q_init
        self.Q = {}

    @property
    def reward(self):
        return self.env.reward

    def q(self, s):
        row = self.Q.get(s)
        if row is None:
            row = self.Q[s] = np.full(4, self.q_init)
        return row

    # --- interface used by ExactEvaluator -----------------------------
    def greedy_action(self, board, moves):
        """Greedy action name for a board, or None if never seen."""
        row = self.Q.get(board)
        if row is None:
            return None
        legal = [ACTION_INDEX[a] for a, _, _ in moves]
        return ACTIONS[greedy(row, legal)]

    def state_value(self, board, moves):
        """Estimate of V(board) = max_a Q(board, a), or None if unseen."""
        row = self.Q.get(board)
        if row is None:
            return None
        return max(row[ACTION_INDEX[a]] for a, _, _ in moves)

    def n_states(self):
        return len(self.Q)

    def policy(self, fallback=dp.corner_policy):
        """Policy function for dynamic_programming (fallback on unseen boards)."""
        def pi(board, moves):
            if not moves:
                return {}
            a = self.greedy_action(board, moves)
            return {a: 1.0} if a is not None else fallback(board, moves)
        return pi

    def episode(self, alpha, eps, rng):
        """One learning episode: returns (return, won, n_steps)."""
        raise NotImplementedError


class Sarsa(TabularAgent):
    name = 'sarsa'

    def episode(self, alpha, eps, rng):
        env, g = self.env, self.gamma
        s, info = env.reset()
        legal = _legal(info)
        a = epsilon_greedy(self.q(s), legal, eps, rng)
        G, steps = 0.0, 0
        while True:
            s2, r, done, _, info = env.step(ACTIONS[a])
            G += r
            steps += 1
            qs = self.q(s)
            if done:
                qs[a] += alpha * (r - qs[a])
                return G, info['won'], steps
            legal = _legal(info)
            a2 = epsilon_greedy(self.q(s2), legal, eps, rng)
            qs[a] += alpha * (r + g * self.Q[s2][a2] - qs[a])
            s, a = s2, a2


class QLearning(TabularAgent):
    name = 'q_learning'

    def episode(self, alpha, eps, rng):
        env, g = self.env, self.gamma
        s, info = env.reset()
        legal = _legal(info)
        G, steps = 0.0, 0
        while True:
            a = epsilon_greedy(self.q(s), legal, eps, rng)
            s2, r, done, _, info = env.step(ACTIONS[a])
            G += r
            steps += 1
            qs = self.q(s)
            if done:
                qs[a] += alpha * (r - qs[a])
                return G, info['won'], steps
            legal = _legal(info)
            q2 = self.q(s2)
            qs[a] += alpha * (r + g * max(q2[b] for b in legal) - qs[a])
            s = s2


class ExpectedSarsa(TabularAgent):
    name = 'expected_sarsa'

    def episode(self, alpha, eps, rng):
        env, g = self.env, self.gamma
        s, info = env.reset()
        legal = _legal(info)
        G, steps = 0.0, 0
        while True:
            a = epsilon_greedy(self.q(s), legal, eps, rng)
            s2, r, done, _, info = env.step(ACTIONS[a])
            G += r
            steps += 1
            qs = self.q(s)
            if done:
                qs[a] += alpha * (r - qs[a])
                return G, info['won'], steps
            legal = _legal(info)
            q2 = self.q(s2)
            expect = sum(p * q2[b] for b, p in
                         epsilon_greedy_probs(q2, legal, eps).items())
            qs[a] += alpha * (r + g * expect - qs[a])
            s = s2


class NStepSarsa(TabularAgent):
    """n-step Sarsa (Sutton-Barto, Sec. 7.2, episodic)."""
    name = 'nstep_sarsa'

    def __init__(self, env, gamma=1.0, q_init=0.0, n=3, **kwargs):
        super().__init__(env, gamma, q_init)
        self.n = n

    def episode(self, alpha, eps, rng):
        env, g, n = self.env, self.gamma, self.n
        s, info = env.reset()
        states, actions, rewards = [s], [], [0.0]
        actions.append(epsilon_greedy(self.q(s), _legal(info), eps, rng))
        T = float('inf')
        t = 0
        won = False
        while True:
            if t < T:
                s2, r, done, _, info = env.step(ACTIONS[actions[t]])
                states.append(s2)
                rewards.append(r)
                if done:
                    T = t + 1
                    won = info['won']
                else:
                    actions.append(epsilon_greedy(self.q(s2), _legal(info),
                                                  eps, rng))
            tau = t - n + 1
            if tau >= 0:
                G = 0.0
                for i in range(tau + 1, min(tau + n, T) + 1):
                    G += g ** (i - tau - 1) * rewards[i]
                if tau + n < T:
                    G += g ** n * self.Q[states[tau + n]][actions[tau + n]]
                qs = self.q(states[tau])
                qs[actions[tau]] += alpha * (G - qs[actions[tau]])
            if tau == T - 1:
                return sum(rewards), won, int(T)
            t += 1


class MCControl(TabularAgent):
    """On-policy first-visit MC control, epsilon-soft (Sec. 5.4).

    alpha=None gives sample averages of the returns (the textbook
    version); a number gives a constant-step-size update instead.
    """
    name = 'mc_control'

    def __init__(self, env, gamma=1.0, q_init=0.0, **kwargs):
        super().__init__(env, gamma, q_init)
        self.N = defaultdict(int)

    def episode(self, alpha, eps, rng):
        env, g = self.env, self.gamma
        s, info = env.reset()
        trajectory = []
        while True:
            a = epsilon_greedy(self.q(s), _legal(info), eps, rng)
            s2, r, done, _, info = env.step(ACTIONS[a])
            trajectory.append((s, a, r))
            s = s2
            if done:
                break
        G = 0.0
        # states never repeat within an episode: every visit is a first visit
        for s, a, r in reversed(trajectory):
            G = r + g * G
            qs = self.q(s)
            if alpha is None:
                self.N[(s, a)] += 1
                qs[a] += (G - qs[a]) / self.N[(s, a)]
            else:
                qs[a] += alpha * (G - qs[a])
        return G, info['won'], len(trajectory)


class AfterstateSarsa(TabularAgent):
    """Sarsa (Sec. 6.4) with action values q(x, u) = r(x, u) + V[M(x, u)].

    M is the deterministic slide/merge rule and r(x, u) the points of the
    move ('score' reward; in the 'win' objective q(x, u) = 1 if the move
    creates the target, otherwise V[M(x, u)], and the winning afterstates
    are not stored). All the pairs (x, u) with the same board after the
    slide y = M(x, u) share the entry V[y]: the parametrisation is exact,
    because the true q depends on (x, u) only through r(x, u) and y.

    The Sarsa error with this q is
        R_{t+1} + q(S_{t+1}, A_{t+1}) - q(S_t, A_t)
          = r(S_{t+1}, A_{t+1}) + V[Y_{t+1}] - V[Y_t],   Y_t = M(S_t, A_t),
    because R_{t+1} = r(S_t, A_t) cancels; so the update of q(S_t, A_t) is
    the update V[Y_t] += alpha (r' + V[Y_{t+1}] - V[Y_t]), where (Y_{t+1},
    r') come from the next move actually chosen (epsilon-greedily: the
    method is on-policy), with target 0 when the next board is game over.
    The idea of learning the values of the positions reached by one's own
    move is the one of the tic-tac-toe player of Sutton and Barto, Sec. 1.5.
    """
    name = 'afterstate_sarsa'

    def __init__(self, env, gamma=1.0, q_init=0.0, **kwargs):
        super().__init__(env, gamma, q_init)
        self.V = {}
        self.win_mode = env.reward == 'win'
        self.r = env.reward_fn  # r(board, afterstate, points); None for 'win'

    def v(self, y):
        return self.V.get(y, self.q_init)

    def action_values(self, board, moves):
        """{action name: r + V[afterstate]} over the legal moves."""
        q = {}
        for action, after, points in moves:
            if self.win_mode:
                q[action] = 1.0 if self.game.is_win(after) else self.v(after)
            else:
                q[action] = self.r(board, after, points) + self.gamma * self.v(after)
        return q

    def _choose(self, board, moves, eps, rng):
        if eps > 0.0 and rng.random() < eps:
            return moves[rng.randrange(len(moves))]
        q = self.action_values(board, moves)
        best = moves[0]
        for m in moves[1:]:
            if q[m[0]] > q[best[0]]:
                best = m
        return best

    def episode(self, alpha, eps, rng):
        env = self.env
        s, info = env.reset()
        moves = self.game.moves(s)
        action, after, _ = self._choose(s, moves, eps, rng)
        G, steps = 0.0, 0
        while True:
            s2, r, done, _, info = env.step(action)
            G += r
            steps += 1
            if done:
                if not (self.win_mode and info['won']):
                    # game over: the afterstate leads nowhere
                    self.V[after] = self.v(after) + alpha * (0.0 - self.v(after))
                return G, info['won'], steps
            moves2 = self.game.moves(s2)
            action2, after2, points2 = self._choose(s2, moves2, eps, rng)
            if self.win_mode:
                target = 1.0 if self.game.is_win(after2) else self.v(after2)
            else:
                target = self.r(s2, after2, points2) + self.gamma * self.v(after2)
            self.V[after] = self.v(after) + alpha * (target - self.v(after))
            s, action, after = s2, action2, after2

    # --- interface used by ExactEvaluator -----------------------------
    def greedy_action(self, board, moves):
        q = self.action_values(board, moves)
        known = any((self.game.is_win(after) if self.win_mode else False)
                    or after in self.V for _, after, _ in moves)
        if not known:
            return None
        best = moves[0][0]
        for action, _, _ in moves[1:]:
            if q[action] > q[best]:
                best = action
        return best

    def state_value(self, board, moves):
        if self.greedy_action(board, moves) is None:
            return None
        return max(self.action_values(board, moves).values())

    def n_states(self):
        return len(self.V)


AGENTS = {'mc_control': MCControl, 'sarsa': Sarsa, 'q_learning': QLearning,
          'expected_sarsa': ExpectedSarsa, 'nstep_sarsa': NStepSarsa,
          'afterstate_sarsa': AfterstateSarsa}


# ----------------------------------------------------------------------
# Exact evaluation against the DP
# ----------------------------------------------------------------------
class ExactEvaluator:
    """Exact quantities about a learned agent on a small board.

    Built once per game (solves the DP for both objectives). evaluate()
    returns a dict with
      p_win        exact probability of winning of the greedy policy
      score        exact expected score of the greedy policy
      unseen       share of the optimal policy's visits to boards the
                   agent has never seen (where the fallback policy acts)
      agree        share of the optimal policy's visits (to seen boards)
                   where the greedy action is optimal
      rmse_w       RMSE of the learned V against V* (of the agent's
                   objective) weighted by the visits of the optimal policy
      rmse         unweighted RMSE over the seen boards
    """

    def __init__(self, game, with_score=True):
        self.game = game
        self.V_win, table, self.layers_win = dp.solve_optimal(game, 'win')
        self.mu = dp.occupancy(game, dp.table_policy(table), self.layers_win)
        self.optimal = {b: set(dp.optimal_actions(game, b, self.V_win))
                        for b in self.mu}
        if with_score:
            self.V_score, _, self.layers_score = dp.solve_optimal(game, 'score')
        else:
            self.V_score = self.layers_score = None

    @classmethod
    def cached(cls, game, path, with_score=True):
        """Load the evaluator from a pickle, or build it and save it. A
        pickle that no longer loads (e.g. written before the modules were
        renamed) is built again."""
        import pickle
        from pathlib import Path
        path = Path(path)
        if path.exists():
            try:
                with open(path, 'rb') as f:
                    return pickle.load(f)
            except (ModuleNotFoundError, AttributeError, EOFError, pickle.UnpicklingError):
                pass
        ev = cls(game, with_score)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(ev, f, protocol=pickle.HIGHEST_PROTOCOL)
        return ev

    def add_reference(self, reward):
        """Optimal values of another reward (a name of shaped_rewards.REWARDS),
        used as the reference of the value error of agents trained with it."""
        if not hasattr(self, 'V_extra'):
            self.V_extra = {}
        if reward not in self.V_extra:
            self.V_extra[reward] = dp.solve_optimal(self.game, reward,
                                                    self.layers_score)[0]

    def evaluate(self, agent):
        g = self.game
        pi = agent.policy()
        out = {'p_win': dp.start_value(
            g, dp.evaluate_policy(g, pi, 'win', self.layers_win))}
        if self.V_score is not None:
            out['score'] = dp.start_value(
                g, dp.evaluate_policy(g, pi, 'score', self.layers_score))
        refs = {'win': self.V_win, 'score': self.V_score,
                **getattr(self, 'V_extra', {})}
        V_ref = refs.get(agent.reward)
        total = seen = agree = 0.0
        err_w = err = 0.0
        n_seen = 0
        for board, visits in self.mu.items():
            moves = g.moves(board)
            if not moves:
                continue
            total += visits
            a = agent.greedy_action(board, moves)
            if a is None:
                continue
            seen += visits
            if a in self.optimal[board]:
                agree += visits
            if V_ref is not None:
                d = agent.state_value(board, moves) - V_ref[board]
                err_w += visits * d * d
                err += d * d
                n_seen += 1
        out['unseen'] = 1.0 - seen / total
        out['agree'] = agree / seen if seen else 0.0
        if V_ref is not None and n_seen:
            out['rmse_w'] = math.sqrt(err_w / seen)
            out['rmse'] = math.sqrt(err / n_seen)
        return out


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def train(method, env, n_episodes, *, alpha=0.1, epsilon=0.1, gamma=1.0,
          n=3, q_init=0.0, seed=0, eval_every=2000, evaluator=None,
          verbose=False):
    """Train one agent and return (agent, history).

    alpha and epsilon can be numbers or schedules (callables of the
    episode number, see constant/linear/exponential/inverse). alpha=None
    with mc_control gives sample averages. history is a list of dicts,
    one every eval_every episodes, with the training statistics of the
    last window (mean return, win rate, episode length) and, if an
    ExactEvaluator is given, its exact quantities.
    """
    if method not in AGENTS:
        raise ValueError(f'method must be one of {METHODS}')
    rng = random.Random(seed)
    env.rng.seed(seed + 1)
    agent = AGENTS[method](env, gamma=gamma, q_init=q_init, n=n)
    alpha_s = _schedule(alpha) if alpha is not None else constant(None)
    eps_s = _schedule(epsilon)
    history = []
    returns, wins, lengths = [], 0, []
    t0 = time.perf_counter()
    for ep in range(1, n_episodes + 1):
        G, won, steps = agent.episode(alpha_s(ep), eps_s(ep), rng)
        returns.append(G)
        wins += won
        lengths.append(steps)
        if ep % eval_every == 0 or ep == n_episodes:
            row = {'episode': ep, 'mean_return': float(np.mean(returns)),
                   'win_rate': wins / len(returns),
                   'mean_length': float(np.mean(lengths)),
                   'states': agent.n_states(),
                   'time_s': time.perf_counter() - t0}
            if evaluator is not None:
                row.update(evaluator.evaluate(agent))
            history.append(row)
            if verbose:
                extra = (f"  exact P(win) {row['p_win']:.4f}"
                         if 'p_win' in row else '')
                print(f"{method:>15} ep {ep:>7}: return {row['mean_return']:8.2f}"
                      f"  wins {row['win_rate']:.3f}  states {row['states']:>6}"
                      f"{extra}  ({row['time_s']:.0f} s)", flush=True)
            returns, wins, lengths = [], 0, []
    return agent, history
