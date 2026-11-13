"""Graphics for 2048: draw boards and watch a policy play by itself.

In a Jupyter notebook (with src/ on the path):
    from IPython.display import HTML
    from visualize import animate, play_episode
    frames = play_episode(game, dp.table_policy(table), seed=0, V=V)
    HTML(animate(game, frames).to_jshtml())   # player with play/pause/step

From a terminal (a window where the game plays by itself):
    python src/visualize.py                          # optimal policy, 2x3 board, target 64
    python src/visualize.py --rows 3 --cols 3        # 3x3 board (the DP takes ~2 minutes)
    python src/visualize.py --rows 4 --cols 4 --target 2048 --policy corner

Each move is drawn in two frames, as in the formulation: the afterstate
(tiles slid and merged) and the board after the random tile, which is
outlined. With the optimal values V, a bar shows the probability of winning
from the current board under optimal play.
"""
import argparse
import random
import time

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import FancyBboxPatch

import dynamic_programming as dp
from game import Game2048

# Colours of the original game: tile value -> (background, text)
TILE_COLORS = {
    0: ('#cdc1b4', '#776e65'),
    2: ('#eee4da', '#776e65'), 4: ('#ede0c8', '#776e65'),
    8: ('#f2b179', '#f9f6f2'), 16: ('#f59563', '#f9f6f2'),
    32: ('#f67c5f', '#f9f6f2'), 64: ('#f65e3b', '#f9f6f2'),
    128: ('#edcf72', '#f9f6f2'), 256: ('#edcc61', '#f9f6f2'),
    512: ('#edc850', '#f9f6f2'), 1024: ('#edc53f', '#f9f6f2'),
    2048: ('#edc22e', '#f9f6f2'),
}
BIG_TILE = ('#3c3a32', '#f9f6f2')  # tiles above 2048
BOARD_COLOR = '#bbada0'
TEXT_COLOR = '#776e65'
BAR_COLOR = '#8f7a66'
ARROWS = {'down': '↓', 'left': '←', 'right': '→', 'up': '↑'}


# ----------------------------------------------------------------------
# Recording a game
# ----------------------------------------------------------------------
def play_episode(game, policy, seed=0, V=None, max_moves=100_000):
    """Play one game with a policy and record the frames to draw.

    Returns a list of dicts with keys board, phase ('start', 'slide',
    'spawn', 'won', 'lost'), action, move, score, value and new_cell.
    value is the optimal probability of winning (from V, or None): V(x) for
    boards and W(y) for afterstates, so the bar moves with every new tile.
    """
    rng = random.Random(seed)

    def value_of(board, afterstate=False):
        if V is None:
            return None
        if afterstate:
            return dp.afterstate_value(game, board, V)
        return V.get(board, 0.0)

    board = game.sample_initial(rng)
    frames = [dict(board=board, phase='start', action=None, move=0, score=0,
                   value=value_of(board), new_cell=None)]
    score = 0
    for move in range(1, max_moves + 1):
        moves = game.moves(board)
        dist = policy(board, moves)
        if not dist:
            frames.append(dict(frames[-1], phase='lost', new_cell=None))
            break
        actions = list(dist)
        action = rng.choices(actions, [dist[a] for a in actions])[0]
        after, points = next((af, pt) for a, af, pt in moves if a == action)
        score += points
        if game.is_win(after):
            frames.append(dict(board=after, phase='won', action=action,
                               move=move, score=score,
                               value=None if V is None else 1.0, new_cell=None))
            break
        frames.append(dict(board=after, phase='slide', action=action,
                           move=move, score=score,
                           value=value_of(after, afterstate=True),
                           new_cell=None))
        board = game.sample_spawn(after, rng)
        new_cell = next(i for i, (a, b) in enumerate(zip(after, board))
                        if a != b)
        frames.append(dict(board=board, phase='spawn', action=action,
                           move=move, score=score, value=value_of(board),
                           new_cell=new_cell))
    return frames


def caption(game, frame):
    """One-line description of a frame."""
    move, arrow = frame['move'], ARROWS.get(frame['action'], '')
    if frame['phase'] == 'start':
        return 'start'
    if frame['phase'] == 'slide':
        return f"move {move}: {arrow} {frame['action']}   score {frame['score']}"
    if frame['phase'] == 'spawn':
        new_tile = 2 ** frame['board'][frame['new_cell']]
        return f'move {move}: new tile {new_tile}   score {frame["score"]}'
    if frame['phase'] == 'won':
        return f'move {move}: {arrow}  {game.target} reached!'
    return f'game over after {move} moves'


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def draw_board(ax, game, board, new_cell=None, cell=1.0):
    """Draw a board on a matplotlib Axes (cell = cell size in inches, used
    to scale the numbers)."""
    ax.clear()
    ax.set_xlim(-0.1, game.cols + 0.1)
    ax.set_ylim(-0.1, game.rows + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((-0.05, -0.05), game.cols + 0.1,
                                game.rows + 0.1,
                                boxstyle='round,pad=0,rounding_size=0.12',
                                facecolor=BOARD_COLOR, edgecolor='none'))
    for i, e in enumerate(board):
        r, c = divmod(i, game.cols)
        x, y = c, game.rows - 1 - r
        value = 2 ** e if e else 0
        background, text = TILE_COLORS.get(value, BIG_TILE)
        outlined = i == new_cell
        ax.add_patch(FancyBboxPatch((x + 0.07, y + 0.07), 0.86, 0.86,
                                    boxstyle='round,pad=0,rounding_size=0.08',
                                    facecolor=background,
                                    edgecolor=TEXT_COLOR if outlined else 'none',
                                    linewidth=3 if outlined else 0))
        if value:
            digits = len(str(value))
            size = cell * 72 * (0.42 if digits <= 2 else
                                0.32 if digits == 3 else
                                0.25 if digits == 4 else 0.2)
            ax.text(x + 0.5, y + 0.5, str(value), ha='center', va='center',
                    fontsize=size, fontweight='bold', color=text)


def _draw_value(ax, value):
    ax.clear()
    ax.barh([0], [value], height=0.6, color=BAR_COLOR)
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1])
    ax.tick_params(labelsize=7, colors=TEXT_COLOR)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.set_title(f'optimal P(win) from here: {value:.3f}', fontsize=9,
                 color=TEXT_COLOR)


def animate(game, frames, interval=600, cell=1.0, repeat=True,
            notebook=True):
    """Animation of recorded frames (FuncAnimation). In a notebook show it
    with HTML(anim.to_jshtml()); in a script pass notebook=False and call
    plt.show()."""
    with_bar = frames[0]['value'] is not None
    bar_height = 0.8 if with_bar else 0.0
    width = cell * game.cols + 0.6
    height = cell * game.rows + 0.75 + bar_height
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor('#faf8ef')
    ax_board = fig.add_axes([0.3 / width, (bar_height + 0.1) / height,
                             cell * game.cols / width,
                             cell * game.rows / height])
    ax_bar = (fig.add_axes([0.5 / width, 0.25 / height, (width - 1.0) / width,
                            0.15 / height]) if with_bar else None)
    title = fig.suptitle('', fontsize=11, fontweight='bold', color=TEXT_COLOR)

    def update(k):
        frame = frames[k]
        draw_board(ax_board, game, frame['board'], frame['new_cell'], cell)
        title.set_text(caption(game, frame))
        if ax_bar is not None:
            _draw_value(ax_bar, frame['value'])

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=interval, repeat=repeat)
    if notebook:
        plt.close(fig)  # avoid an extra static copy of the figure
    return anim


# ----------------------------------------------------------------------
# Live window
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Watch a policy play 2048 by itself.')
    parser.add_argument('--rows', type=int, default=2)
    parser.add_argument('--cols', type=int, default=3)
    parser.add_argument('--target', type=int, default=64)
    parser.add_argument('--p4', type=float, default=0.1)
    parser.add_argument('--policy', default='optimal',
                        choices=['optimal', 'corner', 'greedy', 'random'])
    parser.add_argument('--games', type=int, default=5,
                        help='number of games played one after the other')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--interval', type=int, default=500,
                        help='milliseconds between frames')
    args = parser.parse_args()

    game = Game2048(args.rows, args.cols, target=args.target, p4=args.p4)
    V = None
    if args.policy == 'optimal':
        if game.n_cells > 9:
            parser.error('the optimal policy is computed by exact DP and is '
                         'available only up to 9 cells: use --policy corner, '
                         'greedy or random on bigger boards')
        print(f'solving {game} by exact DP...', flush=True)
        t0 = time.perf_counter()
        V, table, _ = dp.solve_optimal(game)
        print(f'done in {time.perf_counter() - t0:.1f} s, optimal P(win) = '
              f'{dp.start_value(game, V):.4f}', flush=True)
        policy = dp.table_policy(table)
    else:
        policy = dp.HEURISTICS[args.policy]
        if game.n_cells <= 6:
            V, _, _ = dp.solve_optimal(game)  # instant: show the win bar too

    seed = args.seed if args.seed is not None else random.randrange(10**6)
    frames = []
    for g in range(args.games):
        episode = play_episode(game, policy, seed=seed + g, V=V)
        frames += episode + [episode[-1]] * 3  # pause on the final board
        print(f"game {g + 1}: {episode[-1]['phase']} "
              f"(max tile {game.max_tile(episode[-1]['board'])})", flush=True)
    anim = animate(game, frames, interval=args.interval, cell=1.3,
                   repeat=False, notebook=False)
    plt.show()
    return anim


if __name__ == '__main__':
    main()
