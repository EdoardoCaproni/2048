"""The six tabular control methods of tabular_control.py on the bitboard engine of
game_fast.py, for boards whose tables do not fit the pure-Python code (the
original 4x4 game).

The algorithms, the step size, the epsilon-greedy exploration and the
tie-breaking are those of tabular_control.py (MC control with sample averages, Sarsa,
Expected Sarsa, Q-learning, 3-step Sarsa, Sarsa on afterstates; gamma = 1,
score reward). Only the data structure changes: the table Q[board] -> 4
action values (V[afterstate] for Sarsa on afterstates) is a hash table keyed
by the 64-bit board, which holds the boards met in training, added with the
initial value q_init the first time they are met, exactly when tabular_control.py adds
them. The values are stored in single precision to save memory.

A board never met in training has no entry: the greedy policy plays there the
first legal move in the order down, left, right, up (the corner rule), as in
the exact evaluation of the report. For Sarsa on afterstates a board counts
as known if at least one of its afterstates has a value.
"""
import json
import time

import numpy as np

import game_fast as fb
from game_fast import njit  # numba imported with the cache directory set
from game import ACTIONS

METHODS = ('mc_control', 'sarsa', 'expected_sarsa', 'q_learning', 'nstep_sarsa',
           'afterstate_sarsa')
N_STEP = 3
BUFFER = 1 << 20          # longest episode stored by Monte Carlo control
HASH = np.uint64(0x9E3779B97F4A7C15)
MAX_LOAD = 0.6            # the table doubles beyond this load


# ----------------------------------------------------------------------
# Hash table: keys[i] = board (0 = empty slot: a board is never empty)
# ----------------------------------------------------------------------
@njit(cache=True)
def _find(keys, x, shift):
    """Slot of x, or the empty slot where it would go (linear probing)."""
    mask = keys.shape[0] - 1
    i = np.int64((x * HASH) >> shift)
    while True:
        k = keys[i]
        if k == x or k == 0:
            return i
        i = (i + 1) & mask


@njit(cache=True)
def _entry(keys, vals, used, x, shift, q_init):
    """Slot of board x, added with the initial values if it is new."""
    i = _find(keys, x, shift)
    if keys[i] == 0:
        keys[i] = x
        for a in range(vals.shape[1]):
            vals[i, a] = q_init
        used[0] += 1
    return i


@njit(cache=True)
def _rehash(keys, vals, visits, N, keys2, vals2, visits2, N2, shift2, with_counts):
    for i in range(keys.shape[0]):
        if keys[i] != 0:
            j = _find(keys2, keys[i], shift2)
            keys2[j] = keys[i]
            vals2[j, :] = vals[i, :]
            visits2[j] = visits[i]
            if with_counts:
                N2[j, :] = N[i, :]


# ----------------------------------------------------------------------
# Action selection (as tabular_control.greedy / epsilon_greedy)
# ----------------------------------------------------------------------
@njit(cache=True)
def _greedy(vals, i, mask):
    """First maximiser, in the order down, left, right, up, among the legal moves."""
    best = -1
    for a in range(4):
        if (mask >> a) & 1 and (best < 0 or vals[i, a] > vals[i, best]):
            best = a
    return best


@njit(cache=True)
def _first_legal(mask):
    for a in range(4):
        if (mask >> a) & 1:
            return a
    return -1


@njit(cache=True)
def _random_legal(mask):
    n = 0
    for a in range(4):
        n += (mask >> a) & 1
    k = int(np.random.random() * n)
    for a in range(4):
        if (mask >> a) & 1:
            if k == 0:
                return a
            k -= 1
    return -1


@njit(cache=True)
def _eps_greedy(vals, i, mask, eps):
    if eps > 0.0 and np.random.random() < eps:
        return _random_legal(mask)
    return _greedy(vals, i, mask)


@njit(cache=True)
def _after_values(keys, vals, shift, q_init, x, mask, rows, cols, q):
    """q[u] = r(x, u) + V[M(x, u)] for the legal moves (V = q_init if the
    afterstate has no entry); returns whether any of them has an entry."""
    known = False
    for a in range(4):
        if (mask >> a) & 1:
            after, pts, _ = fb.move(x, a, rows, cols)
            j = _find(keys, after, shift)
            if keys[j] == after:
                known = True
                q[a] = pts + vals[j, 0]
            else:
                q[a] = pts + q_init
    return known


@njit(cache=True)
def _choose_after(keys, vals, shift, q_init, x, mask, rows, cols, eps, q):
    """AfterstateSarsa._choose: a random legal move with probability eps,
    otherwise the first maximiser of r + V[afterstate]."""
    if eps > 0.0 and np.random.random() < eps:
        return _random_legal(mask)
    _after_values(keys, vals, shift, q_init, x, mask, rows, cols, q)
    best = -1
    for a in range(4):
        if (mask >> a) & 1 and (best < 0 or q[a] > q[best]):
            best = a
    return best


# ----------------------------------------------------------------------
# One learning episode of each method (tabular_control.<Agent>.episode)
# ----------------------------------------------------------------------
@njit(cache=True)
def _episode_td(method, rows, cols, p4, alpha, eps, q_init, keys, vals, visits, used, shift):
    """Sarsa (1), Expected Sarsa (2), Q-learning (3)."""
    s = fb.initial_board(rows, cols, p4)
    mask = fb.legal_mask(s, rows, cols)
    i = _entry(keys, vals, used, s, shift, q_init)
    a = _eps_greedy(vals, i, mask, eps)
    score = 0
    steps = 0
    while True:
        visits[i] += 1
        after, pts, _ = fb.move(s, a, rows, cols)
        s2 = fb.spawn(after, rows, cols, p4)
        score += pts
        steps += 1
        r = float(pts)
        mask2 = fb.legal_mask(s2, rows, cols)
        if mask2 == 0:
            vals[i, a] += alpha * (r - vals[i, a])
            return score, fb.max_exp(s2, rows, cols), steps
        i2 = _entry(keys, vals, used, s2, shift, q_init)
        a2 = -1
        if method == 1:                     # Sarsa: the next move actually chosen
            a2 = _eps_greedy(vals, i2, mask2, eps)
            target = r + vals[i2, a2]
        elif method == 2:                   # Expected Sarsa
            n = 0
            total = 0.0
            for b in range(4):
                if (mask2 >> b) & 1:
                    n += 1
                    total += vals[i2, b]
            target = r + eps * total / n + (1.0 - eps) * vals[i2, _greedy(vals, i2, mask2)]
        else:                               # Q-learning
            target = r + vals[i2, _greedy(vals, i2, mask2)]
        vals[i, a] += alpha * (target - vals[i, a])
        if method != 1:                     # the next move, from the updated table
            a2 = _eps_greedy(vals, i2, mask2, eps)
        s, i, a = s2, i2, a2


@njit(cache=True)
def _episode_nstep(n, rows, cols, p4, alpha, eps, q_init, keys, vals, visits, used, shift,
                   xs, slots, acts, rs):
    """n-step Sarsa (Sec. 7.2), with the states, moves and rewards stored
    modulo n + 1."""
    L = n + 1
    s = fb.initial_board(rows, cols, p4)
    i = _entry(keys, vals, used, s, shift, q_init)
    xs[0] = s
    slots[0] = i
    acts[0] = _eps_greedy(vals, i, fb.legal_mask(s, rows, cols), eps)
    T = np.int64(1) << 62
    t = 0
    score = 0
    last = s
    while True:
        if t < T:
            visits[slots[t % L]] += 1
            after, pts, _ = fb.move(xs[t % L], acts[t % L], rows, cols)
            s2 = fb.spawn(after, rows, cols, p4)
            last = s2
            score += pts
            rs[(t + 1) % L] = pts
            mask2 = fb.legal_mask(s2, rows, cols)
            if mask2 == 0:
                T = t + 1
            else:
                i2 = _entry(keys, vals, used, s2, shift, q_init)
                xs[(t + 1) % L] = s2
                slots[(t + 1) % L] = i2
                acts[(t + 1) % L] = _eps_greedy(vals, i2, mask2, eps)
        tau = t - n + 1
        if tau >= 0:
            G = 0.0
            for k in range(tau + 1, min(tau + n, T) + 1):
                G += rs[k % L]
            if tau + n < T:
                G += vals[slots[(tau + n) % L], acts[(tau + n) % L]]
            j = slots[tau % L]
            at = acts[tau % L]
            vals[j, at] += alpha * (G - vals[j, at])
        if tau == T - 1:
            return score, fb.max_exp(last, rows, cols), T
        t += 1


@njit(cache=True)
def _episode_mc(rows, cols, p4, alpha, eps, q_init, keys, vals, N, visits, used, shift,
                slots, acts, rs):
    """First-visit MC control (Sec. 5.4); alpha <= 0: sample averages of the
    returns, as in the report."""
    s = fb.initial_board(rows, cols, p4)
    mask = fb.legal_mask(s, rows, cols)
    T = 0
    score = 0
    while T < slots.shape[0]:
        i = _entry(keys, vals, used, s, shift, q_init)
        visits[i] += 1
        a = _eps_greedy(vals, i, mask, eps)
        after, pts, _ = fb.move(s, a, rows, cols)
        s = fb.spawn(after, rows, cols, p4)
        slots[T] = i
        acts[T] = a
        rs[T] = pts
        score += pts
        T += 1
        mask = fb.legal_mask(s, rows, cols)
        if mask == 0:
            break
    G = 0.0
    for t in range(T - 1, -1, -1):   # states never repeat: every visit is a first visit
        G += rs[t]
        i = slots[t]
        a = acts[t]
        if alpha <= 0.0:
            N[i, a] += 1
            vals[i, a] += (G - vals[i, a]) / N[i, a]
        else:
            vals[i, a] += alpha * (G - vals[i, a])
    return score, fb.max_exp(s, rows, cols), T


@njit(cache=True)
def _episode_after(rows, cols, p4, alpha, eps, q_init, keys, vals, visits, used, shift, q):
    """Sarsa on afterstates, score reward: V[Y_t] += alpha (r' + V[Y_{t+1}] - V[Y_t]),
    target 0 when the next board is game over."""
    s = fb.initial_board(rows, cols, p4)
    mask = fb.legal_mask(s, rows, cols)
    a = _choose_after(keys, vals, shift, q_init, s, mask, rows, cols, eps, q)
    y, pts, _ = fb.move(s, a, rows, cols)
    score = 0
    steps = 0
    while True:
        s2 = fb.spawn(y, rows, cols, p4)
        score += pts
        steps += 1
        mask2 = fb.legal_mask(s2, rows, cols)
        if mask2 == 0:
            target = 0.0
        else:
            a2 = _choose_after(keys, vals, shift, q_init, s2, mask2, rows, cols, eps, q)
            y2, pts2, _ = fb.move(s2, a2, rows, cols)
            j2 = _find(keys, y2, shift)
            target = pts2 + (vals[j2, 0] if keys[j2] == y2 else q_init)
        j = _entry(keys, vals, used, y, shift, q_init)
        visits[j] += 1
        vals[j, 0] += alpha * (target - vals[j, 0])
        if mask2 == 0:
            return score, fb.max_exp(s2, rows, cols), steps
        y = y2
        pts = pts2


@njit(cache=True)
def train_block(method, n_games, seed, rows, cols, p4, alpha, eps, q_init,
                keys, vals, N, visits, used, shift, max_used):
    """Up to n_games learning episodes of a method (index in METHODS) after
    seeding the generator; stops early when the table holds max_used
    entries (the caller then doubles it). Returns (games played, scores,
    max exponents, moves)."""
    fb.seed(seed)
    size = BUFFER if method == 0 else N_STEP + 1
    xs = np.zeros(size, dtype=np.uint64)
    slots = np.zeros(size, dtype=np.int64)
    acts = np.zeros(size, dtype=np.int64)
    rs = np.zeros(size)
    q = np.zeros(4)
    scores = np.zeros(n_games, dtype=np.int64)
    max_exps = np.zeros(n_games, dtype=np.int64)
    moves = np.zeros(n_games, dtype=np.int64)
    g = 0
    while g < n_games and used[0] < max_used:
        if method == 0:
            s, m, n = _episode_mc(rows, cols, p4, alpha, eps, q_init, keys, vals, N, visits,
                                  used, shift, slots, acts, rs)
        elif method <= 3:
            s, m, n = _episode_td(method, rows, cols, p4, alpha, eps, q_init, keys, vals,
                                  visits, used, shift)
        elif method == 4:
            s, m, n = _episode_nstep(N_STEP, rows, cols, p4, alpha, eps, q_init, keys, vals,
                                     visits, used, shift, xs, slots, acts, rs)
        else:
            s, m, n = _episode_after(rows, cols, p4, alpha, eps, q_init, keys, vals, visits,
                                     used, shift, q)
        scores[g] = s
        max_exps[g] = m
        moves[g] = n
        g += 1
    return g, scores[:g], max_exps[:g], moves[:g]


@njit(cache=True)
def _greedy_move(after_values, keys, vals, shift, q_init, x, mask, rows, cols, q):
    """Greedy move of the learned table at board x, and whether the board is
    known (otherwise the corner rule plays)."""
    if after_values:
        known = _after_values(keys, vals, shift, q_init, x, mask, rows, cols, q)
        if not known:
            return _first_legal(mask), False
        best = -1
        for a in range(4):
            if (mask >> a) & 1 and (best < 0 or q[a] > q[best]):
                best = a
        return best, True
    i = _find(keys, x, shift)
    if keys[i] != x:
        return _first_legal(mask), False
    return _greedy(vals, i, mask), True


@njit(cache=True)
def play_greedy(after_values, n_games, seed, rows, cols, p4, q_init, keys, vals, shift):
    """Greedy games of the learned table (no learning, no table growth).
    Returns (scores, max exponents, moves, decisions on boards not in the
    table)."""
    fb.seed(seed)
    q = np.zeros(4)
    scores = np.zeros(n_games, dtype=np.int64)
    max_exps = np.zeros(n_games, dtype=np.int64)
    moves = np.zeros(n_games, dtype=np.int64)
    unseen = np.zeros(n_games, dtype=np.int64)
    for g in range(n_games):
        x = fb.initial_board(rows, cols, p4)
        mask = fb.legal_mask(x, rows, cols)
        while mask != 0:
            a, known = _greedy_move(after_values, keys, vals, shift, q_init, x, mask, rows, cols, q)
            if not known:
                unseen[g] += 1
            after, pts, _ = fb.move(x, a, rows, cols)
            scores[g] += pts
            moves[g] += 1
            x = fb.spawn(after, rows, cols, p4)
            mask = fb.legal_mask(x, rows, cols)
        max_exps[g] = fb.max_exp(x, rows, cols)
    return scores, max_exps, moves, unseen


@njit(cache=True)
def greedy_batch(after_values, boards, rows, cols, q_init, keys, vals, shift):
    """Greedy move (-1 at game over) and whether the board is known, for an
    array of bitboards."""
    q = np.zeros(4)
    actions = np.empty(boards.shape[0], dtype=np.int64)
    known = np.zeros(boards.shape[0], dtype=np.bool_)
    for k in range(boards.shape[0]):
        mask = fb.legal_mask(boards[k], rows, cols)
        if mask == 0:
            actions[k] = -1
        else:
            actions[k], known[k] = _greedy_move(after_values, keys, vals, shift, q_init,
                                                boards[k], mask, rows, cols, q)
    return actions, known


# ----------------------------------------------------------------------
# Python side
# ----------------------------------------------------------------------
class FastTabular:
    """A tabular control method of tabular_control.py on an R x C board, its table in
    a hash table that doubles when it fills up."""

    def __init__(self, method, rows=4, cols=4, target=2048, p4=0.1, q_init=0.0, seed=0,
                 capacity_log2=20):
        if method not in METHODS:
            raise ValueError(f'unknown method {method!r}, choose from {METHODS}')
        self.method, self.rows, self.cols, self.target, self.p4 = method, rows, cols, target, p4
        self.q_init, self.seed, self.games = float(q_init), seed, 0
        self.after = method == 'afterstate_sarsa'
        self.used = np.zeros(1, dtype=np.int64)
        self._allocate(capacity_log2)

    def _allocate(self, log2):
        """Empty table of 2**log2 slots; N (the visit counts of the pairs for
        the sample averages of MC control) is a dummy row for the others."""
        n_vals = 1 if self.after else 4
        counts = self.method == 'mc_control'
        self.log2 = log2
        self.shift = np.uint64(64 - log2)
        self.keys = np.zeros(1 << log2, dtype=np.uint64)
        self.vals = np.zeros((1 << log2, n_vals), dtype=np.float32)
        self.N = np.zeros(((1 << log2) if counts else 1, n_vals), dtype=np.int32)
        self.visits = np.zeros(1 << log2, dtype=np.uint32)

    def _grow(self):
        old = self.keys, self.vals, self.visits, self.N
        self._allocate(self.log2 + 1)
        _rehash(*old, self.keys, self.vals, self.visits, self.N, self.shift,
                self.method == 'mc_control')

    def __repr__(self):
        return (f'FastTabular({self.method}, {self.rows}x{self.cols}, {self.n_states():,} '
                f'{"afterstates" if self.after else "boards"}, {self.games:,} games)')

    def n_states(self):
        return int(self.used[0])

    def table_bytes(self):
        return self.keys.nbytes + self.vals.nbytes + self.N.nbytes + self.visits.nbytes

    # --- learning and playing ---------------------------------------------------
    def learn(self, n_games, alpha, eps, seed):
        """n_games learning episodes; alpha None or <= 0 for Monte Carlo
        control means sample averages."""
        alpha = 0.0 if alpha is None else float(alpha)
        out = [], [], []
        done = 0
        while done < n_games:
            max_used = int(MAX_LOAD * self.keys.shape[0])
            if self.used[0] >= max_used:
                self._grow()
                continue
            g, s, m, n = train_block(METHODS.index(self.method), n_games - done, seed + done,
                                     self.rows, self.cols, self.p4, alpha, eps, self.q_init,
                                     self.keys, self.vals, self.N, self.visits, self.used,
                                     self.shift, max_used)
            for lst, arr in zip(out, (s, m, n)):
                lst.append(arr)
            done += g
            self.games += g
        return tuple(np.concatenate(lst) for lst in out)

    def evaluate(self, n_games, seed, prefix=''):
        """Greedy play of the learned table (epsilon = 0, no learning): mean
        score and moves, the share of the games that reach each tile, and the
        share of the decisions taken on boards not in the table."""
        scores, max_exps, moves, unseen = play_greedy(
            self.after, n_games, seed, self.rows, self.cols, self.p4, self.q_init,
            self.keys, self.vals, self.shift)
        out = {'score': float(scores.mean()), 'moves': float(moves.mean()),
               'max_tile': int(2 ** max_exps.max()), 'unseen': float(unseen.sum() / moves.sum())}
        for e in range(7, 14):
            out[f'p{2 ** e}'] = float((max_exps >= e).mean())
        return {prefix + k: v for k, v in out.items()}

    def train(self, n_games, alpha, eps, block=10_000, eval_every=20_000, eval_games=200,
              eval_seed=12_345, callback=None, verbose=False):
        """Learning in blocks of games, with a greedy evaluation every
        eval_every games (same seed, so the evaluations see the same tiles)
        and at the end; callback(agent, history) after every evaluation."""
        history = []
        t0 = time.perf_counter()
        done = 0
        while done < n_games:
            n = min(block, n_games - done)
            scores, max_exps, moves = self.learn(n, alpha, eps, self.seed * 1_000_003 + self.games)
            done += n
            row = {'games': self.games, 'train_score': float(scores.mean()),
                   'train_moves': float(moves.mean()), 'states': self.n_states(),
                   'table_mb': self.table_bytes() / 1e6}
            if self.games % eval_every == 0 or done == n_games:
                row.update(self.evaluate(eval_games, eval_seed, prefix='eval_'))
            row['time_s'] = time.perf_counter() - t0
            history.append(row)
            if 'eval_score' in row:
                if callback is not None:
                    callback(self, history)
                if verbose:
                    print(f"{self.method} {self.games:>8} games: greedy score {row['eval_score']:7.0f}, "
                          f"unseen boards {row['eval_unseen']:.1%}, {row['states']:,} entries "
                          f"({row['time_s']:.0f} s)", flush=True)
        return history

    # --- the learned policy at given boards ------------------------------------------
    def greedy_actions(self, boards):
        """(action names, known) for a list of board tuples; at a board not in
        the table the corner rule plays."""
        arr = np.array([fb.encode(b, self.rows, self.cols) for b in boards], dtype=np.uint64)
        acts, known = greedy_batch(self.after, arr, self.rows, self.cols, self.q_init,
                                   self.keys, self.vals, self.shift)
        return [ACTIONS[a] if a >= 0 else None for a in acts], known

    def entries(self, min_visits=1):
        """(keys, values, visits) of the entries visited at least min_visits
        times."""
        sel = (self.keys != 0) & (self.visits >= min_visits)
        return self.keys[sel], self.vals[sel], self.visits[sel]

    # --- persistence (tables of millions of entries: not for git) --------------------
    def config(self):
        return {'method': self.method, 'rows': self.rows, 'cols': self.cols, 'target': self.target,
                'p4': self.p4, 'q_init': self.q_init, 'seed': self.seed, 'games': self.games}

    def save(self, path):
        sel = self.keys != 0
        np.savez(path, keys=self.keys[sel], vals=self.vals[sel], visits=self.visits[sel],
                 config=json.dumps(self.config()))

    @classmethod
    def load(cls, path):
        data = np.load(path)
        cfg = json.loads(str(data['config']))
        return cls.from_entries(cfg, data['keys'], data['vals'], data['visits'])

    @classmethod
    def from_entries(cls, cfg, keys, vals, visits):
        """Agent whose table holds the given entries (e.g. the most visited
        ones of a larger table)."""
        n = len(keys)
        log2 = max(10, int(np.ceil(np.log2(n / MAX_LOAD + 1))))
        agent = cls(cfg['method'], cfg['rows'], cfg['cols'], cfg['target'], cfg['p4'],
                    cfg['q_init'], cfg['seed'], capacity_log2=log2)
        _insert_all(agent.keys, agent.vals, agent.visits, keys, vals, visits, agent.shift)
        agent.used[0] = n
        agent.games = cfg['games']
        return agent

    def most_visited(self, n):
        """(keys, values, visits) of the n entries visited most often."""
        keys, vals, visits = self.entries()
        if len(keys) > n:
            top = np.argpartition(visits, len(keys) - n)[len(keys) - n:]
            keys, vals, visits = keys[top], vals[top], visits[top]
        order = np.argsort(-visits.astype(np.int64), kind='stable')
        return keys[order], vals[order], visits[order]


@njit(cache=True)
def _insert_all(keys, vals, visits, new_keys, new_vals, new_visits, shift):
    for k in range(new_keys.shape[0]):
        i = _find(keys, new_keys[k], shift)
        keys[i] = new_keys[k]
        vals[i, :] = new_vals[k, :]
        visits[i] = new_visits[k]
