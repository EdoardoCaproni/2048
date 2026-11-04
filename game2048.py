"""2048 game engine, shared by the DP part and the RL part of the project.

Board encoding
--------------
A board is a tuple of R*C integers in row-major order. Each entry is the
base-2 exponent of the tile in that cell: 0 = empty, 1 = tile 2,
2 = tile 4, ..., k = tile 2**k. Tuples are hashable, so boards can be used
directly as states (dictionary keys) in the DP.

A move is split in two parts, as in the problem formulation:
  1. the deterministic slide/merge   x -> y = afterstate(x, u)
  2. the random spawn                y -> x' (a 2 with probability 1 - p4 or
                                     a 4 with probability p4, in an empty
                                     cell chosen uniformly at random)
"""
import random
from functools import lru_cache
from operator import itemgetter

# Fixed action order. It is also the tie-breaking order of the DP and the
# priority order of the "corner" heuristic (keep big tiles bottom-left).
ACTIONS = ('down', 'left', 'right', 'up')


@lru_cache(maxsize=None)
def slide_line(line):
    """Slide and merge one line towards index 0.

    Each tile merges at most once per move, and merges are resolved starting
    from the side the tiles move to: (2, 2, 2, 2) -> (4, 4, _, _).
    Returns (new_line, points), where points is the sum of the new tiles.
    """
    tiles = [e for e in line if e]
    merged, points, i = [], 0, 0
    while i < len(tiles):
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            merged.append(tiles[i] + 1)
            points += 2 ** (tiles[i] + 1)
            i += 2
        else:
            merged.append(tiles[i])
            i += 1
    return tuple(merged) + (0,) * (len(line) - len(merged)), points


class Game2048:
    """Rules of 2048 on an R x C board with target tile `target`.

    p4 is the probability that a spawned tile is a 4 (0.1 in the original
    game). The game is won as soon as a move creates the target tile, and it
    is lost when no move changes the board.
    """

    def __init__(self, rows=4, cols=4, target=2048, p4=0.1):
        if rows < 2 or cols < 2:
            raise ValueError('the board needs at least 2 rows and 2 columns')
        if target < 8 or target & (target - 1):
            raise ValueError('target must be a power of 2, at least 8')
        if not 0.0 <= p4 <= 1.0:
            raise ValueError('p4 must be a probability')
        self.rows, self.cols = rows, cols
        self.n_cells = rows * cols
        self.target = target
        self.target_exp = target.bit_length() - 1
        self.p4 = p4
        row_lines = [tuple(r * cols + c for c in range(cols))
                     for r in range(rows)]
        col_lines = [tuple(r * cols + c for r in range(rows))
                     for c in range(cols)]
        # For each action, the lines of cell indices ordered towards the side
        # the tiles move to (so that slide_line can always work towards 0).
        self._lines = {
            'down': [line[::-1] for line in col_lines],
            'left': row_lines,
            'right': [line[::-1] for line in row_lines],
            'up': col_lines,
        }
        # itemgetter(*line)(board) reads the cells of a line in one call
        self._getters = {action: [(line, itemgetter(*line)) for line in lines]
                         for action, lines in self._lines.items()}
        # Spawned tile exponents with their probabilities (zero ones dropped).
        self._spawn_values = [(e, p) for e, p in ((1, 1.0 - p4), (2, p4))
                              if p > 0]

    def __repr__(self):
        return (f'Game2048(rows={self.rows}, cols={self.cols}, '
                f'target={self.target}, p4={self.p4})')

    # ------------------------------------------------------------------
    # Deterministic part of a move
    # ------------------------------------------------------------------
    def afterstate(self, board, action):
        """Slide/merge the board in a direction: (afterstate, points)."""
        new = list(board)
        points = 0
        for line, get_cells in self._getters[action]:
            moved, p = slide_line(get_cells(board))
            points += p
            for i, e in zip(line, moved):
                new[i] = e
        return tuple(new), points

    def moves(self, board):
        """Legal moves as a list of (action, afterstate, points).

        A move is legal only if it changes the board. The list follows the
        order of ACTIONS and is empty when the game is over.
        """
        out = []
        for action in ACTIONS:
            after, points = self.afterstate(board, action)
            if after != board:
                out.append((action, after, points))
        return out

    def legal_actions(self, board):
        return [action for action, _, _ in self.moves(board)]

    def is_game_over(self, board):
        return not self.moves(board)

    # ------------------------------------------------------------------
    # Random part of a move
    # ------------------------------------------------------------------
    def spawn_distribution(self, afterstate):
        """Distribution of the next state: list of (board, probability).

        After a legal move there is always at least one empty cell.
        """
        empty = [i for i, e in enumerate(afterstate) if e == 0]
        out = []
        for i in empty:
            for e, p in self._spawn_values:
                board = list(afterstate)
                board[i] = e
                out.append((tuple(board), p / len(empty)))
        return out

    def initial_distribution(self):
        """Two tiles in two distinct uniformly chosen cells, each one a 2
        (probability 1 - p4) or a 4 (probability p4)."""
        pairs = [(i, j) for i in range(self.n_cells)
                 for j in range(i + 1, self.n_cells)]
        out = []
        for i, j in pairs:
            for ei, pi in self._spawn_values:
                for ej, pj in self._spawn_values:
                    board = [0] * self.n_cells
                    board[i], board[j] = ei, ej
                    out.append((tuple(board), pi * pj / len(pairs)))
        return out

    def sample_spawn(self, afterstate, rng):
        """Sample the next state from an afterstate (rng: random.Random)."""
        empty = [i for i, e in enumerate(afterstate) if e == 0]
        board = list(afterstate)
        board[rng.choice(empty)] = 2 if rng.random() < self.p4 else 1
        return tuple(board)

    def sample_initial(self, rng):
        board = [0] * self.n_cells
        for i in rng.sample(range(self.n_cells), 2):
            board[i] = 2 if rng.random() < self.p4 else 1
        return tuple(board)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def is_win(self, board):
        """True if the board contains the target tile (or a larger one)."""
        return max(board) >= self.target_exp

    @staticmethod
    def tile_sum(board):
        """Sum of the tile values: it grows by exactly 2 or 4 per move."""
        return sum(2 ** e for e in board if e)

    @staticmethod
    def max_tile(board):
        m = max(board)
        return 2 ** m if m else 0

    def render(self, board):
        """Board as text, one row per line ('.' = empty cell)."""
        cells = [str(2 ** e) if e else '.' for e in board]
        width = max(len(c) for c in cells)
        return '\n'.join(
            ' '.join(c.rjust(width)
                     for c in cells[r * self.cols:(r + 1) * self.cols])
            for r in range(self.rows))


class Env2048:
    """Minimal Gymnasium-style environment (reset/step) for the RL part.

    reward='win':   +1 when a move creates the target tile, which ends the
                    episode; 0 otherwise.
    reward='score': the points of each move (sum of the merged tiles); the
                    target is ignored and the episode ends at game over.

    Only legal actions are allowed: read them from info['legal_actions'] or
    from legal_actions(). info['afterstate'] is the board right after the
    slide/merge, before the random tile (useful for afterstate learning).
    """

    def __init__(self, game, reward='win', seed=None):
        if reward not in ('win', 'score'):
            raise ValueError("reward must be 'win' or 'score'")
        self.game = game
        self.reward = reward
        self.rng = random.Random(seed)
        self.board = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)
        self.board = self.game.sample_initial(self.rng)
        return self.board, {'legal_actions': self.legal_actions()}

    def legal_actions(self):
        return self.game.legal_actions(self.board)

    def step(self, action):
        """Returns (observation, reward, terminated, truncated, info)."""
        after, points = self.game.afterstate(self.board, action)
        if after == self.board:
            raise ValueError(f'illegal action {action!r}')
        if self.reward == 'win' and self.game.is_win(after):
            self.board = after
            return after, 1.0, True, False, {
                'won': True, 'afterstate': after, 'legal_actions': []}
        self.board = self.game.sample_spawn(after, self.rng)
        legal = self.legal_actions()
        reward = float(points) if self.reward == 'score' else 0.0
        return self.board, reward, not legal, False, {
            'won': False, 'afterstate': after, 'legal_actions': legal}
