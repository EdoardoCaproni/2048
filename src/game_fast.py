"""Fast 2048 engine: a 64-bit bitboard compiled with numba.

The pure-Python engine in game.py is the reference implementation and
is used by the exact DP. This module reproduces exactly the same rules for
the RL part, where millions of moves are needed (4x4 board).

Board encoding
--------------
A board is a numpy uint64. Cell (r, c) of an R x C board (R, C <= 4) is
stored in the 4-bit nibble number 4*r + c, i.e. bits 4*(4*r + c) to
4*(4*r + c) + 3, holding the exponent of the tile (0 = empty, k = 2**k).
The layout is always the padded 4x4 one: row r occupies bits 16r to
16r + 15 and the cells outside the R x C region stay 0. This makes one
engine serve every board size, with the 4x4 game at full speed.

A slide is done row by row with lookup tables indexed by the 16-bit row:
LEFT[w-2, row] and RIGHT[w-2, row] give the row after sliding towards
column 0 or column w-1, for a row of width w in {2, 3, 4}; PTS_LEFT and
PTS_RIGHT give the points. Up and down transpose the board, use the
tables of width R, and transpose back. The tables are built from
game.slide_line, so the two engines agree by construction (see the
tests in tests.py).

Random numbers come from numba's own generator (separate from numpy's):
call seed(s) once per process or block before sampling.

Action indices follow game.ACTIONS: 0 down, 1 left, 2 right, 3 up.
"""
import os
from pathlib import Path

import numpy as np

# numba's on-disk cache goes next to the module by default; the project
# path contains a space and an accented letter, so use an ASCII path.
os.environ.setdefault('NUMBA_CACHE_DIR',
                      str(Path.home() / '.numba_cache' / 'progetto_2048'))
from numba import njit  # noqa: E402

from game import ACTIONS, slide_line  # noqa: E402

ZERO = np.uint64(0)
MASK4 = np.uint64(0xF)
MASK16 = np.uint64(0xFFFF)


def _pack(cells):
    return sum(e << (4 * c) for c, e in enumerate(cells))


def _build_tables():
    """Row tables for widths 2, 3, 4, built from game.slide_line.

    Rows whose padding nibbles (c >= w) are not empty cannot occur on a
    board of width w: they map to themselves. A merge creating a tile
    2**16 does not fit in a nibble: such rows are left unchanged too
    (unreachable in practice, but the board must never be corrupted).
    """
    left = np.zeros((3, 65536), dtype=np.uint16)
    right = np.zeros((3, 65536), dtype=np.uint16)
    pts_left = np.zeros((3, 65536), dtype=np.int32)
    pts_right = np.zeros((3, 65536), dtype=np.int32)
    for w in (2, 3, 4):
        k = w - 2
        for row in range(65536):
            cells = [(row >> (4 * c)) & 0xF for c in range(4)]
            if any(cells[w:]):
                left[k, row] = right[k, row] = row
                continue
            line = tuple(cells[:w])
            moved, p = slide_line(line)
            if max(moved) > 15:
                moved, p = line, 0
            left[k, row], pts_left[k, row] = _pack(moved), p
            moved, p = slide_line(line[::-1])
            if max(moved) > 15:
                moved, p = line[::-1], 0
            right[k, row], pts_right[k, row] = _pack(moved[::-1]), p
    return left, right, pts_left, pts_right


LEFT, RIGHT, PTS_LEFT, PTS_RIGHT = _build_tables()


# ----------------------------------------------------------------------
# Conversions (plain Python)
# ----------------------------------------------------------------------
def cell_index(r, c):
    """Nibble number of cell (r, c) in the padded layout."""
    return 4 * r + c


def encode(board, rows, cols):
    """Tuple of exponents (row-major, as in game.py) -> uint64."""
    x = 0
    for r in range(rows):
        for c in range(cols):
            x |= int(board[r * cols + c]) << (4 * cell_index(r, c))
    return np.uint64(x)


def decode(x, rows, cols):
    """uint64 -> tuple of exponents (row-major, as in game.py)."""
    x = int(x)
    return tuple((x >> (4 * cell_index(r, c))) & 0xF
                 for r in range(rows) for c in range(cols))


# ----------------------------------------------------------------------
# Board inspection
# ----------------------------------------------------------------------
@njit(cache=True)
def nibble(x, n):
    """Exponent stored in nibble n (n = 4*r + c), as an int64."""
    return np.int64((x >> np.uint64(4 * n)) & MASK4)


@njit(cache=True)
def count_empty(x, rows, cols):
    n = 0
    for r in range(rows):
        for c in range(cols):
            if ((x >> np.uint64(16 * r + 4 * c)) & MASK4) == ZERO:
                n += 1
    return n


@njit(cache=True)
def max_exp(x, rows, cols):
    m = np.int64(0)
    for r in range(rows):
        for c in range(cols):
            e = np.int64((x >> np.uint64(16 * r + 4 * c)) & MASK4)
            if e > m:
                m = e
    return m


@njit(cache=True)
def tile_sum(x, rows, cols):
    s = np.int64(0)
    for r in range(rows):
        for c in range(cols):
            e = np.int64((x >> np.uint64(16 * r + 4 * c)) & MASK4)
            if e > 0:
                s += np.int64(1) << e
    return s


@njit(cache=True)
def is_win(x, rows, cols, target_exp):
    return max_exp(x, rows, cols) >= target_exp


# ----------------------------------------------------------------------
# Moves
# ----------------------------------------------------------------------
@njit(cache=True)
def transpose(x):
    """Transpose of the 4x4 nibble matrix (cell (r, c) -> (c, r))."""
    a1 = x & np.uint64(0xF0F00F0FF0F00F0F)
    a2 = x & np.uint64(0x0000F0F00000F0F0)
    a3 = x & np.uint64(0x0F0F00000F0F0000)
    a = a1 | (a2 << np.uint64(12)) | (a3 >> np.uint64(12))
    b1 = a & np.uint64(0xFF00FF0000FF00FF)
    b2 = a & np.uint64(0x00FF00FF00000000)
    b3 = a & np.uint64(0x00000000FF00FF00)
    return b1 | (b2 >> np.uint64(24)) | (b3 << np.uint64(24))


@njit(cache=True)
def move(x, action, rows, cols):
    """Slide/merge in a direction: (afterstate, points, changed).

    action: 0 down, 1 left, 2 right, 3 up (the order of game.ACTIONS).
    A move is legal only if changed is True.
    """
    horizontal = action == 1 or action == 2
    if horizontal:
        src, nrows, w = x, rows, cols
    else:
        # columns become rows of width `rows`
        src, nrows, w = transpose(x), cols, rows
    toward_zero = action == 1 or action == 3   # left, up
    out = ZERO
    points = 0
    for r in range(nrows):
        shift = np.uint64(16 * r)
        row = (src >> shift) & MASK16
        if toward_zero:
            out |= np.uint64(LEFT[w - 2, row]) << shift
            points += PTS_LEFT[w - 2, row]
        else:
            out |= np.uint64(RIGHT[w - 2, row]) << shift
            points += PTS_RIGHT[w - 2, row]
    if not horizontal:
        out = transpose(out)
    return out, points, out != x


@njit(cache=True)
def legal_mask(x, rows, cols):
    """Bit a is set if action a changes the board."""
    mask = 0
    for a in range(4):
        _, _, changed = move(x, a, rows, cols)
        if changed:
            mask |= 1 << a
    return mask


@njit(cache=True)
def is_terminal(x, rows, cols):
    return legal_mask(x, rows, cols) == 0


# ----------------------------------------------------------------------
# Random part of a move (numba's generator)
# ----------------------------------------------------------------------
@njit(cache=True)
def seed(s):
    np.random.seed(s)


@njit(cache=True)
def spawn(x, rows, cols, p4):
    """Add a 2 (probability 1 - p4) or a 4 (probability p4) in an empty
    cell chosen uniformly at random. Requires at least one empty cell."""
    n_empty = count_empty(x, rows, cols)
    k = int(np.random.random() * n_empty)
    v = np.uint64(2) if np.random.random() < p4 else np.uint64(1)
    i = 0
    for r in range(rows):
        for c in range(cols):
            shift = np.uint64(16 * r + 4 * c)
            if ((x >> shift) & MASK4) == ZERO:
                if i == k:
                    return x | (v << shift)
                i += 1
    return x


@njit(cache=True)
def initial_board(rows, cols, p4):
    """Two tiles in two distinct uniformly chosen cells."""
    n = rows * cols
    i = int(np.random.random() * n)
    j = int(np.random.random() * (n - 1))
    if j >= i:
        j += 1
    x = ZERO
    for cell in (i, j):
        r, c = cell // cols, cell % cols
        v = np.uint64(2) if np.random.random() < p4 else np.uint64(1)
        x |= v << np.uint64(16 * r + 4 * c)
    return x


@njit(cache=True)
def env_step(x, action, rows, cols, target_exp, p4, reward_mode):
    """One step of the game from state x with a legal action.

    reward_mode 0 ('win'):   reward 1 and the episode ends when the move
                             creates the target tile; otherwise reward 0.
    reward_mode 1 ('score'): reward = points of the move; the target is
                             ignored and the episode ends at game over.
    Returns (next_state, reward, terminal, afterstate, target_created),
    where target_created says whether the afterstate contains the target
    (in score mode play goes on, the caller keeps the flag).
    """
    after, points, changed = move(x, action, rows, cols)
    created = max_exp(after, rows, cols) >= target_exp
    if reward_mode == 0 and created:
        return after, 1.0, True, after, True
    nxt = spawn(after, rows, cols, p4)
    terminal = legal_mask(nxt, rows, cols) == 0
    reward = float(points) if reward_mode == 1 else 0.0
    return nxt, reward, terminal, after, created


@njit(cache=True)
def play_random(rows, cols, target_exp, p4, n_games, stop_at_target):
    """Play n_games with uniformly random legal moves (benchmark and
    smoke test). Returns (scores, max_exps, n_moves, won)."""
    scores = np.zeros(n_games, dtype=np.int64)
    max_exps = np.zeros(n_games, dtype=np.int64)
    n_moves = np.zeros(n_games, dtype=np.int64)
    won = np.zeros(n_games, dtype=np.bool_)
    for g in range(n_games):
        x = initial_board(rows, cols, p4)
        score = 0
        moves = 0
        while True:
            mask = legal_mask(x, rows, cols)
            if mask == 0:
                break
            n_legal = 0
            for a in range(4):
                if mask & (1 << a):
                    n_legal += 1
            k = int(np.random.random() * n_legal)
            action = 0
            for a in range(4):
                if mask & (1 << a):
                    if k == 0:
                        action = a
                        break
                    k -= 1
            after, points, _ = move(x, action, rows, cols)
            score += points
            moves += 1
            if max_exp(after, rows, cols) >= target_exp:
                won[g] = True
                if stop_at_target:
                    x = after
                    break
            x = spawn(after, rows, cols, p4)
        scores[g] = score
        max_exps[g] = max_exp(x, rows, cols)
        n_moves[g] = moves
    return scores, max_exps, n_moves, won


# ----------------------------------------------------------------------
# Glue with the pure-Python side
# ----------------------------------------------------------------------
def legal_actions(x, rows, cols):
    """Legal actions as names, in ACTIONS order."""
    mask = legal_mask(x, rows, cols)
    return [ACTIONS[a] for a in range(4) if mask & (1 << a)]


def warm_up():
    """Trigger the compilation of every jitted function (first call)."""
    x = initial_board(2, 2, 0.1)
    seed(0)
    x = spawn(move(x, 1, 2, 2)[0], 2, 2, 0.1) if legal_mask(x, 2, 2) else x
    env_step(x, 0, 2, 2, 4, 0.1, 1)
    is_terminal(x, 2, 2)
    tile_sum(x, 2, 2)
    nibble(x, 0)
    play_random(2, 2, 4, 0.1, 1, True)
