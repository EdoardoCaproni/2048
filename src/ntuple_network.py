"""Episodic semi-gradient Sarsa with a linear approximation (n-tuple
features) for 2048.

Action values (Sutton and Barto, Sec. 10.1)
-------------------------------------------
The approximate action value of a move u at a board x is

    q(x, u; w) = r(x, u) + w^T phi(M(x, u)),

where y = M(x, u) is the board after the deterministic slide/merge (before
the random tile), r(x, u) are the points of the move and phi(y) is a binary
feature vector; V(y; w) = w^T phi(y) is the learned part. The features are
built from n-tuples of cells (the "n-tuple network" of Szubert and
Jaskowski, 2014): for every tuple m, index_m(y) packs the exponents found
in those cells into an integer (4 bits per cell) and phi has a 1 in the
entry (m, index_m(y)) of table m. So

    V(y; w) = sum_m  w[owner(m), index_m(y)],

a linear function of w with M active binary features (one per tuple, like
M overlapping state aggregations, Sec. 9.3), and grad_w q(x, u; w) = phi(y).

Learning algorithm
------------------
The episodic semi-gradient one-step Sarsa update (Sec. 10.1),

    w <- w + alpha [R_{t+1} + q(S_{t+1}, A_{t+1}; w) - q(S_t, A_t; w)] grad q(S_t, A_t; w),

becomes, since R_{t+1} = r(S_t, A_t) cancels,

    w <- w + alpha [r(S_{t+1}, A_{t+1}) + V(Y_{t+1}; w) - V(Y_t; w)] phi(Y_t),

with Y_t = M(S_t, A_t), i.e. a multiple of the TD error added to the M
active weights; the target is 0 when S_{t+1} is game over. Here alpha is
the step of the value V(y): each active weight moves by alpha / M times
the error, so that alpha has the same meaning for every tuple set. The move
is chosen epsilon-greedily with respect to q; with epsilon = 0 (the default,
as in the Mountain Car example of Sec. 10.1) the random tile is the only
source of exploration, and Sarsa, Q-learning and Expected Sarsa coincide.

Symmetric weight sharing: with symmetric=True every base tuple is also
read in all its images under the symmetries of the board (8 for a square
board, 4 otherwise), and the images share the weights of the base tuple:
a choice of features that makes V invariant under the symmetries, like V*
(the idea of Exercise 1.2 of Sutton and Barto).

reward_mode 0 ('win'):   r = 1 when the move creates the target tile,
                         the episode ends there; V(winning y) = 1 fixed.
reward_mode 1 ('score'): r = points of the move; play goes on past the
                         target; whether the target was reached is logged.
"""
import json
import time

import numpy as np

import game_fast as fb
from game_fast import njit  # numba imported with the cache directory set
from game import ACTIONS

REWARD_MODES = {'win': 0, 'score': 1}


# ----------------------------------------------------------------------
# Tuple shapes and symmetries
# ----------------------------------------------------------------------
def _cell(r, c):
    return 4 * r + c


def make_tuples(rows, cols, name):
    """Base n-tuples (lists of nibble indices) for a board.

    'full'    one tuple with all the cells (a lookup table of the whole
              afterstate: tabular Sarsa on afterstates, exact on small boards)
    'rows'    all the rows
    'rcs'     rows, columns and 2x2 squares (on 4x4 the 17 tuples of
              Szubert and Jaskowski 2014; 'szubert' is an alias)
    'rect'    all the 2x3 and 3x2 rectangles (6-tuples)
    """
    row_t = [[_cell(r, c) for c in range(cols)] for r in range(rows)]
    col_t = [[_cell(r, c) for r in range(rows)] for c in range(cols)]
    squares = [[_cell(r, c), _cell(r, c + 1), _cell(r + 1, c), _cell(r + 1, c + 1)]
               for r in range(rows - 1) for c in range(cols - 1)]
    rect23 = [[_cell(r + i, c + j) for i in range(2) for j in range(3)]
              for r in range(rows - 1) for c in range(cols - 2)]
    rect32 = [[_cell(r + i, c + j) for i in range(3) for j in range(2)]
              for r in range(rows - 2) for c in range(cols - 1)]
    shapes = {
        'full': [[_cell(r, c) for r in range(rows) for c in range(cols)]],
        'rows': row_t,
        'rcs': row_t + col_t + squares,
        'szubert': row_t + col_t + squares,
        'rect': rect23 + rect32,
    }
    if name not in shapes:
        raise ValueError(f'unknown tuple set {name!r}, choose from {list(shapes)}')
    return [tuple(t) for t in shapes[name]]


def symmetries(rows, cols):
    """Cell permutations of the board symmetries, as dicts nibble -> nibble."""
    maps = [lambda r, c: (r, c), lambda r, c: (r, cols - 1 - c),
            lambda r, c: (rows - 1 - r, c), lambda r, c: (rows - 1 - r, cols - 1 - c)]
    if rows == cols:
        maps += [lambda r, c: (c, r), lambda r, c: (c, rows - 1 - r),
                 lambda r, c: (cols - 1 - c, r),
                 lambda r, c: (cols - 1 - c, rows - 1 - r)]
    out = []
    for f in maps:
        out.append({_cell(r, c): _cell(*f(r, c))
                    for r in range(rows) for c in range(cols)})
    return out


def expand(base, rows, cols, symmetric):
    """Feature arrays (cells, owner) from the base tuples.

    cells[m] lists the k cells of feature m (shorter tuples are padded by
    repeating their first cell), owner[m] is the row of the weight table
    it reads. With symmetric=True the images of a base tuple under the
    board symmetries share its row, and base tuples that are images of
    one another are merged.
    """
    k = max(len(t) for t in base)
    feats, owners = [], []
    if symmetric:
        kept = []
        for t in base:
            images = []
            for s in symmetries(rows, cols):
                img = tuple(s[n] for n in t)
                if img not in images:
                    images.append(img)
            if any(set(img) == set(k) for img in images for k in kept):
                continue  # covers the same cells as an image of an earlier base tuple
            kept.append(t)
            owner = len(kept) - 1
            for img in images:
                feats.append(img + (img[0],) * (k - len(img)))
                owners.append(owner)
        n_rows = len(kept)
    else:
        for i, t in enumerate(base):
            feats.append(t + (t[0],) * (k - len(t)))
            owners.append(i)
        n_rows = len(base)
    return (np.array(feats, dtype=np.int64), np.array(owners, dtype=np.int64),
            n_rows)


# ----------------------------------------------------------------------
# Jitted core
# ----------------------------------------------------------------------
@njit(cache=True)
def evaluate(x, W, cells, owner):
    v = 0.0
    for m in range(cells.shape[0]):
        idx = 0
        for j in range(cells.shape[1]):
            idx |= fb.nibble(x, cells[m, j]) << (4 * j)
        v += W[owner[m], idx]
    return v


@njit(cache=True)
def update(x, W, cells, owner, delta):
    for m in range(cells.shape[0]):
        idx = 0
        for j in range(cells.shape[1]):
            idx |= fb.nibble(x, cells[m, j]) << (4 * j)
        W[owner[m], idx] += delta


@njit(cache=True)
def choose(x, W, cells, owner, rows, cols, target_exp, reward_mode):
    """Greedy move at board x: (action, afterstate, points, q, creates
    target). action = -1 if no move is legal. q is the estimate of the
    value of the state, max_u [r(x,u) + V(M(x,u))]."""
    best_a = -1
    best_q = -1e300
    best_after = x
    best_pts = 0
    best_win = False
    for a in range(4):
        after, pts, changed = fb.move(x, a, rows, cols)
        if not changed:
            continue
        win = fb.max_exp(after, rows, cols) >= target_exp
        if reward_mode == 0:
            q = 1.0 if win else evaluate(after, W, cells, owner)
        else:
            q = pts + evaluate(after, W, cells, owner)
        if q > best_q:
            best_a, best_q, best_after, best_pts, best_win = a, q, after, pts, win
    return best_a, best_after, best_pts, best_q, best_win


@njit(cache=True)
def random_move(x, rows, cols, target_exp, reward_mode, W, cells, owner):
    """A uniformly random legal move, in the same format as choose()."""
    mask = fb.legal_mask(x, rows, cols)
    if mask == 0:
        return -1, x, 0, 0.0, False
    n = 0
    for a in range(4):
        if mask & (1 << a):
            n += 1
    k = int(np.random.random() * n)
    for a in range(4):
        if mask & (1 << a):
            if k == 0:
                after, pts, _ = fb.move(x, a, rows, cols)
                win = fb.max_exp(after, rows, cols) >= target_exp
                if reward_mode == 0:
                    q = 1.0 if win else evaluate(after, W, cells, owner)
                else:
                    q = pts + evaluate(after, W, cells, owner)
                return a, after, pts, q, win
            k -= 1
    return -1, x, 0, 0.0, False


@njit(cache=True)
def play_td(W, cells, owner, rows, cols, target_exp, p4, reward_mode, alpha,
            learn, eps):
    """One game of semi-gradient Sarsa learning (or of greedy play if learn is
    False). Returns (score, max exponent of the final board, moves,
    target reached)."""
    x = fb.initial_board(rows, cols, p4)
    score = 0
    moves = 0
    won = False
    if eps > 0.0 and np.random.random() < eps:
        a, after, pts, q, win = random_move(x, rows, cols, target_exp,
                                            reward_mode, W, cells, owner)
    else:
        a, after, pts, q, win = choose(x, W, cells, owner, rows, cols,
                                       target_exp, reward_mode)
    while a >= 0:
        score += pts
        moves += 1
        if win:
            won = True
            if reward_mode == 0:
                break  # terminal: reward 1 was the target of the last update
        x = fb.spawn(after, rows, cols, p4)
        if eps > 0.0 and np.random.random() < eps:
            a2, after2, pts2, q2, win2 = random_move(
                x, rows, cols, target_exp, reward_mode, W, cells, owner)
        else:
            a2, after2, pts2, q2, win2 = choose(
                x, W, cells, owner, rows, cols, target_exp, reward_mode)
        if learn:
            target = q2 if a2 >= 0 else 0.0
            # alpha is the step on V: every active weight moves by alpha / M
            delta = alpha / cells.shape[0] * (target - evaluate(after, W, cells, owner))
            update(after, W, cells, owner, delta)
        a, after, pts, q, win = a2, after2, pts2, q2, win2
    return score, fb.max_exp(x, rows, cols), moves, won


@njit(cache=True)
def play_block(W, cells, owner, rows, cols, target_exp, p4, reward_mode,
               alpha, n_games, seed, learn, eps):
    """n_games games of play_td after seeding the generator. Returns the
    arrays (scores, max_exps, moves, won)."""
    fb.seed(seed)
    scores = np.zeros(n_games, dtype=np.int64)
    max_exps = np.zeros(n_games, dtype=np.int64)
    n_moves = np.zeros(n_games, dtype=np.int64)
    won = np.zeros(n_games, dtype=np.bool_)
    for g in range(n_games):
        s, m, n, w = play_td(W, cells, owner, rows, cols, target_exp, p4,
                             reward_mode, alpha, learn, eps)
        scores[g], max_exps[g], n_moves[g], won[g] = s, m, n, w
    return scores, max_exps, n_moves, won


@njit(cache=True)
def greedy_batch(boards, W, cells, owner, rows, cols, target_exp, reward_mode):
    """Greedy action (-1 at game over) and induced state value for an
    array of boards."""
    n = boards.shape[0]
    actions = np.empty(n, dtype=np.int64)
    values = np.empty(n, dtype=np.float64)
    for i in range(n):
        a, _, _, q, _ = choose(boards[i], W, cells, owner, rows, cols,
                               target_exp, reward_mode)
        actions[i] = a
        values[i] = q if a >= 0 else 0.0
    return actions, values


# ----------------------------------------------------------------------
# Python side
# ----------------------------------------------------------------------
class NTupleNet:
    """An n-tuple network for an R x C board with target 2**K.

    init is the initial value of every afterstate (spread over the
    active features), used for optimistic initialisation.
    """

    def __init__(self, rows, cols, target, tuples='rcs', symmetric=True,
                 dtype=np.float32, p4=0.1, init=0.0):
        self.rows, self.cols, self.target, self.p4 = rows, cols, target, p4
        self.target_exp = target.bit_length() - 1
        self.tuples, self.symmetric = tuples, symmetric
        base = make_tuples(rows, cols, tuples)
        self.cells, self.owner, n_rows = expand(base, rows, cols, symmetric)
        self.k = self.cells.shape[1]
        self.W = np.full((n_rows, 16 ** self.k), init / len(self.cells),
                         dtype=dtype)
        self.games = 0

    def __repr__(self):
        return (f'NTupleNet({self.rows}x{self.cols}, target {self.target}, '
                f'{self.tuples}, symmetric={self.symmetric}, '
                f'{self.W.shape[0]} tables x 16^{self.k}, '
                f'{len(self.cells)} features, {self.W.size:,} weights)')

    # --- persistence ----------------------------------------------------
    def config(self):
        return {'rows': self.rows, 'cols': self.cols, 'target': self.target,
                'tuples': self.tuples, 'symmetric': self.symmetric,
                'dtype': np.dtype(self.W.dtype).name, 'p4': self.p4,
                'games': self.games}

    def save(self, path):
        np.savez_compressed(path, W=self.W, config=json.dumps(self.config()))

    @classmethod
    def load(cls, path):
        data = np.load(path)
        cfg = json.loads(str(data['config']))
        net = cls(cfg['rows'], cfg['cols'], cfg['target'], cfg['tuples'],
                  cfg['symmetric'], np.dtype(cfg['dtype']), cfg['p4'])
        net.W[:] = data['W']
        net.games = cfg['games']
        return net

    # --- use with the pure-Python side ------------------------------------
    def value(self, afterstate):
        """V(y) of an afterstate given as a tuple of exponents."""
        return float(evaluate(fb.encode(afterstate, self.rows, self.cols),
                              self.W, self.cells, self.owner))

    def greedy(self, board, reward='score'):
        """(action name or None, induced value) for a board tuple."""
        a, _, _, q, _ = choose(fb.encode(board, self.rows, self.cols), self.W,
                               self.cells, self.owner, self.rows, self.cols,
                               self.target_exp, REWARD_MODES[reward])
        return (ACTIONS[a], float(q)) if a >= 0 else (None, 0.0)

    def policy(self, reward='score'):
        """Policy function for dynamic_programming / visualize (greedy, deterministic)."""
        def pi(board, moves):
            if not moves:
                return {}
            return {self.greedy(board, reward)[0]: 1.0}
        return pi

    def frozen(self, boards, reward='score'):
        """FrozenPolicy over an iterable of board tuples (one batched
        pass), for the exact evaluation on small boards."""
        boards = list(boards)
        arr = np.array([fb.encode(b, self.rows, self.cols) for b in boards],
                       dtype=np.uint64)
        actions, values = greedy_batch(arr, self.W, self.cells, self.owner,
                                       self.rows, self.cols, self.target_exp,
                                       REWARD_MODES[reward])
        return FrozenPolicy(boards, actions, values, reward)

    # --- learning -----------------------------------------------------------
    def play(self, n_games, reward='score', alpha=0.0, seed=0, learn=False,
             eps=0.0):
        """Play n_games (learning if learn is True). Returns the arrays
        (scores, max_exps, moves, won)."""
        out = play_block(self.W, self.cells, self.owner, self.rows, self.cols,
                         self.target_exp, self.p4, REWARD_MODES[reward],
                         alpha, n_games, seed, learn, eps)
        if learn:
            self.games += n_games
        return out

    def train(self, n_games, alpha, reward='score', seed=0, block=1000,
              eps=0.0, evaluator=None, eval_at=(), checkpoint=None,
              verbose=False):
        """Semi-gradient Sarsa learning for n_games games, in blocks.

        alpha is the learning rate of the value V(y) (the TD error is
        spread over the M active features, each weight moving by
        alpha / M times the error); it can be a number or a function of
        the number of games played so far. After every block one row of statistics is appended to the
        history; at the game counts in eval_at (or at every block if
        eval_at is True) the evaluator, if given, is called with the net
        and its dict is merged into the row. checkpoint(net, games) is
        called at the same times.
        """
        history = []
        t0 = time.perf_counter()
        done = 0
        while done < n_games:
            n = min(block, n_games - done)
            a = alpha(self.games) if callable(alpha) else alpha
            scores, max_exps, moves, won = self.play(
                n, reward, a, seed=seed * 1000003 + self.games, learn=True, eps=eps)
            done += n
            row = {'games': self.games, 'alpha': a,
                   'mean_score': float(scores.mean()),
                   'median_score': float(np.median(scores)),
                   'p_target': float(won.mean()),
                   'mean_moves': float(moves.mean()),
                   'max_tile': int(2 ** max_exps.max()),
                   'time_s': time.perf_counter() - t0}
            for e in range(4, 17):
                row[f'n_{2 ** e}'] = int((max_exps == e).sum())
            at_eval = (eval_at is True or self.games in eval_at
                       or done == n_games)
            if at_eval and evaluator is not None:
                row.update(evaluator(self))
            if at_eval and checkpoint is not None:
                checkpoint(self, self.games)
            history.append(row)
            if verbose:
                extra = f"  P(win) exact {row['p_win']:.4f}" if 'p_win' in row else ''
                print(f"{self.games:>8} games: score {row['mean_score']:9.1f}  "
                      f"target {row['p_target']:.3f}  max {row['max_tile']:>5}"
                      f"{extra}  ({row['time_s']:.0f} s)", flush=True)
        return history


class FrozenPolicy:
    """The greedy policy of a net on a fixed set of boards, with the
    interface that tabular_control.ExactEvaluator expects from an agent."""

    def __init__(self, boards, actions, values, reward):
        self.reward = reward
        self.actions = {b: (ACTIONS[a] if a >= 0 else None)
                        for b, a in zip(boards, actions)}
        self.values = dict(zip(boards, values.tolist()))

    def greedy_action(self, board, moves):
        return self.actions.get(board)

    def state_value(self, board, moves):
        return self.values.get(board)

    def n_states(self):
        return len(self.actions)

    def policy(self, fallback=None):
        def pi(board, moves):
            if not moves:
                return {}
            a = self.actions.get(board)
            if a is None:
                return fallback(board, moves) if fallback else {moves[0][0]: 1.0}
            return {a: 1.0}
        return pi


def exact_evaluator(evaluator, game, reward):
    """Adapter: a tabular_control.ExactEvaluator as the evaluator argument of
    NTupleNet.train (evaluates the greedy policy on every decision state
    of the DP in one batched pass)."""
    boards = [b for layer in evaluator.layers_win.values() for b in layer]

    def f(net):
        return evaluator.evaluate(net.frozen(boards, reward))
    return f
