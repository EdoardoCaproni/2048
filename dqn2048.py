"""Deep Q-learning (DQN) for 2048: the final test of the RL part.

The agent follows the PyTorch tutorial "Reinforcement Learning (DQN)
Tutorial" by A. Paszke and M. Towers
(https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html),
an implementation of the DQN of Mnih et al. (2015): Q-learning (Sutton and
Barto, Sec. 6.5) with a neural network approximating the action values
(Sec. 9.7), stabilised by an experience replay memory and a target network
(these two devices are not part of the course). Components and
hyper-parameters are those of the tutorial:

  network      MLP with two hidden layers of 128 ReLU units, one output
               per move
  memory       the last 10 000 transitions; minibatches of 128 sampled
               uniformly at random
  target net   a copy of the network, soft-updated after every step:
               theta' <- tau theta + (1 - tau) theta', tau = 0.005
  loss         Huber (SmoothL1) between Q(s, a) and r + gamma max_a' Q'(s', a')
               (0 in place of the max when s' is terminal)
  optimiser    AdamW, learning rate 3e-4, amsgrad; gradients clipped at 100
  exploration  epsilon-greedy, epsilon = 0.01 + (0.9 - 0.01) exp(-steps / 2500)
  discount     gamma = 0.99; one optimisation step per environment step

Adaptations to 2048 (the tutorial plays CartPole), and only these:
  - the observation is the one-hot encoding of the exponent of every cell
    (R*C*16 inputs);
  - only legal moves are chosen: the Q-values of the illegal moves are
    masked when acting and in the max over the next state's moves;
  - the reward is the score of the move (play goes on past the target until
    game over) divided by the target tile (64 on 2x3, 2048 on 4x4), so that
    the returns are of order one; scaling the rewards scales Q and leaves
    the greedy policy unchanged;
  - the replay memory keeps the transitions in preallocated numpy arrays
    instead of a deque of tensors (same push/sample behaviour, faster).
"""
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from game2048 import ACTIONS

N_VALUES = 16          # exponents 0..15 (one-hot channels per cell)
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4
MEMORY = 10_000


class DQN(nn.Module):
    """The network of the tutorial."""

    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


def encode(boards):
    """One-hot encoding of a list of boards: array (len, cells * 16)."""
    arr = np.minimum(np.asarray(boards, dtype=np.int64), N_VALUES - 1)
    n, cells = arr.shape
    out = np.zeros((n, cells * N_VALUES), dtype=np.float32)
    out[np.arange(n)[:, None], np.arange(cells) * N_VALUES + arr] = 1.0
    return out


def legal_mask(game, board):
    """Boolean array of the legal moves of a board, in ACTIONS order."""
    legal = set(game.legal_actions(board))
    return np.array([a in legal for a in ACTIONS])


class ReplayMemory:
    """Cyclic buffer of the last `capacity` transitions (state, action,
    reward, next state, legal moves of the next state, terminal flag)."""

    def __init__(self, capacity, obs_dim):
        self.s = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a = np.zeros(capacity, dtype=np.int64)
        self.r = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.m2 = np.zeros((capacity, len(ACTIONS)), dtype=bool)
        self.done = np.zeros(capacity, dtype=bool)
        self.capacity, self.size, self.pos = capacity, 0, 0

    def push(self, s, a, r, s2, m2, done):
        i = self.pos
        self.s[i], self.a[i], self.r[i] = s, a, r
        self.s2[i], self.m2[i], self.done[i] = s2, m2, done
        self.pos = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng, device='cpu'):
        idx = np.array(rng.sample(range(self.size), batch_size))
        return tuple(torch.from_numpy(x[idx]).to(device)
                     for x in (self.s, self.a, self.r, self.s2, self.m2, self.done))

    def __len__(self):
        return self.size


class DQNAgent:
    """DQN agent for a Game2048 (any board size), trained with the score
    reward."""

    def __init__(self, game, seed=0, reward_scale=None, device='cpu'):
        self.game = game
        self.reward_scale = float(reward_scale or game.target)
        self.device = torch.device(device)
        n_obs = game.rows * game.cols * N_VALUES
        torch.manual_seed(seed)
        self.rng = random.Random(seed)            # exploration and sampling
        self.env_rng = random.Random(seed + 1)    # random tiles
        self.policy_net = DQN(n_obs, len(ACTIONS)).to(self.device)
        self.target_net = DQN(n_obs, len(ACTIONS)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR,
                                     amsgrad=True)
        self.memory = ReplayMemory(MEMORY, n_obs)
        self.steps_done = 0
        self.episodes = 0
        self.reward = 'score'

    # --- acting ---------------------------------------------------------
    def epsilon(self):
        return EPS_END + (EPS_START - EPS_END) * math.exp(-self.steps_done / EPS_DECAY)

    def select_action(self, obs, mask):
        """Epsilon-greedy among the legal moves (as select_action of the
        tutorial, with the illegal moves masked)."""
        eps = self.epsilon()
        self.steps_done += 1
        if self.rng.random() > eps:
            with torch.no_grad():
                q = self.policy_net(torch.from_numpy(obs)[None].to(self.device))[0].cpu().numpy()
            q[~mask] = -np.inf
            return int(q.argmax())
        return int(self.rng.choice(np.flatnonzero(mask)))

    # --- learning -------------------------------------------------------
    def optimize_model(self):
        """One gradient step on a minibatch from the memory (as
        optimize_model of the tutorial, with the masked max)."""
        if len(self.memory) < BATCH_SIZE:
            return None
        s, a, r, s2, m2, done = self.memory.sample(BATCH_SIZE, self.rng, self.device)
        state_action_values = self.policy_net(s).gather(1, a[:, None])
        with torch.no_grad():
            q2 = self.target_net(s2)
            q2[~m2] = -float('inf')
            next_values = q2.max(1).values
            next_values[done] = 0.0
        expected = next_values * GAMMA + r
        loss = F.smooth_l1_loss(state_action_values, expected[:, None])
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def soft_update(self):
        """theta' <- tau theta + (1 - tau) theta' (target network)."""
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(),
                              self.policy_net.parameters()):
                tp.mul_(1.0 - TAU).add_(pp, alpha=TAU)

    def train_episode(self):
        """One training game. Returns (score, max tile, moves, target
        reached, mean loss)."""
        g = self.game
        board = g.sample_initial(self.env_rng)
        obs, mask = encode([board])[0], legal_mask(g, board)
        score = moves = 0
        won = False
        losses = []
        while True:
            a = self.select_action(obs, mask)
            after, points = g.afterstate(board, ACTIONS[a])
            score += points
            moves += 1
            won = won or g.is_win(after)
            nxt = g.sample_spawn(after, self.env_rng)
            obs2, mask2 = encode([nxt])[0], legal_mask(g, nxt)
            done = not mask2.any()
            self.memory.push(obs, a, points / self.reward_scale, obs2, mask2, done)
            loss = self.optimize_model()
            if loss is not None:
                losses.append(loss)
            self.soft_update()
            board, obs, mask = nxt, obs2, mask2
            if done:
                break
        self.episodes += 1
        return score, g.max_tile(board), moves, won, (float(np.mean(losses)) if losses else float('nan'))

    # --- greedy policy ----------------------------------------------------
    def greedy_batch(self, boards, chunk=8192):
        """Greedy action index (-1 at game over) and max Q of the boards
        (Q rescaled to score units)."""
        actions = np.empty(len(boards), dtype=np.int64)
        values = np.empty(len(boards), dtype=np.float64)
        for i in range(0, len(boards), chunk):
            part = boards[i:i + chunk]
            masks = np.array([legal_mask(self.game, b) for b in part])
            with torch.no_grad():
                q = self.policy_net(torch.from_numpy(encode(part)).to(self.device)).cpu().numpy()
            q[~masks] = -np.inf
            a = q.argmax(1)
            a[~masks.any(1)] = -1
            actions[i:i + len(part)] = a
            values[i:i + len(part)] = np.where(a >= 0, q.max(1), 0.0) * self.reward_scale
        return actions, values

    def play_greedy(self, n_games, seed):
        """Monte Carlo evaluation: n_games played greedily, no learning.
        Returns a dict of arrays (score, max_tile, moves, won)."""
        g = self.game
        rng = random.Random(seed)
        out = {'score': [], 'max_tile': [], 'moves': [], 'won': []}
        for _ in range(n_games):
            board = g.sample_initial(rng)
            score = moves = 0
            won = False
            while True:
                mask = legal_mask(g, board)
                if not mask.any():
                    break
                with torch.no_grad():
                    q = self.policy_net(torch.from_numpy(encode([board])).to(self.device))[0].cpu().numpy()
                q[~mask] = -np.inf
                after, points = g.afterstate(board, ACTIONS[int(q.argmax())])
                score += points
                moves += 1
                won = won or g.is_win(after)
                board = g.sample_spawn(after, rng)
            out['score'].append(score)
            out['max_tile'].append(g.max_tile(board))
            out['moves'].append(moves)
            out['won'].append(won)
        return {k: np.array(v) for k, v in out.items()}

    def save(self, path):
        torch.save({'state_dict': {k: v.cpu() for k, v in self.policy_net.state_dict().items()},
                    'episodes': self.episodes, 'steps': self.steps_done,
                    'board': (self.game.rows, self.game.cols, self.game.target)},
                   path)


def train(agent, n_episodes, eval_every=1000, evaluator=None, verbose=False,
          callback=None):
    """Train for n_episodes games. Every eval_every games a row of training
    statistics is recorded, merged with evaluator(agent) if given;
    callback(agent, history), if given, is called after every row (e.g. to
    save a checkpoint during a long run)."""
    history = []
    block = {'score': [], 'max_tile': [], 'moves': [], 'won': [], 'loss': []}
    t0 = time.perf_counter()
    for ep in range(1, n_episodes + 1):
        score, max_tile, moves, won, loss = agent.train_episode()
        for k, v in zip(block, (score, max_tile, moves, won, loss)):
            block[k].append(v)
        if ep % eval_every == 0 or ep == n_episodes:
            row = {'episode': agent.episodes, 'steps': agent.steps_done,
                   'epsilon': agent.epsilon(),
                   'train_score': float(np.mean(block['score'])),
                   'train_moves': float(np.mean(block['moves'])),
                   'train_target': float(np.mean(block['won'])),
                   'train_max_tile': int(np.max(block['max_tile'])),
                   'loss': float(np.nanmean(block['loss'])),
                   'time_s': time.perf_counter() - t0}
            if evaluator is not None:
                row.update(evaluator(agent))
            history.append(row)
            if verbose:
                extra = ''.join(f'  {k} {row[k]:.4g}' for k in ('p_win', 'score', 'eval_score')
                                if k in row)
                print(f"ep {row['episode']:>7}  steps {row['steps']:>9}  train score "
                      f"{row['train_score']:8.1f}{extra}  ({row['time_s']:.0f} s)",
                      flush=True)
            if callback is not None:
                callback(agent, history)
            block = {k: [] for k in block}
    return history


def exact_evaluator(evaluator):
    """Adapter for a small board: an rl2048.ExactEvaluator applied to the
    greedy policy of the network on every board of both DP layer sets (so
    that no board needs a fallback policy). Returns p_win, score, agree."""
    from ntuple2048 import FrozenPolicy
    boards = sorted({b for layers in (evaluator.layers_win, evaluator.layers_score)
                     for layer in layers.values() for b in layer})

    def f(agent):
        actions, values = agent.greedy_batch(boards)
        res = evaluator.evaluate(FrozenPolicy(boards, actions, values, 'score'))
        return {k: res[k] for k in ('p_win', 'score', 'agree', 'unseen')}
    return f


def mc_evaluator(n_games=200, seed=12345, targets=(512, 1024, 2048, 4096)):
    """Monte Carlo evaluator for the 4x4 board: greedy games with a fixed
    seed. Returns the mean score and the share of games reaching each tile."""
    def f(agent):
        res = agent.play_greedy(n_games, seed)
        row = {'eval_score': float(res['score'].mean()),
               'eval_moves': float(res['moves'].mean()),
               'eval_max_tile': int(res['max_tile'].max())}
        for t in targets:
            row[f'eval_p{t}'] = float((res['max_tile'] >= t).mean())
        return row
    return f
