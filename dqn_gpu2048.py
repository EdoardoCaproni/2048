"""Deep Q-learning on the GPU, with many games played at the same time.

The agent of dqn2048.py (same network, Huber loss, AdamW with learning rate
3e-4 and amsgrad, gradients clipped at 100, target network soft-updated with
tau = 0.005 after every gradient step, the epsilon-greedy schedule of the
PyTorch tutorial, gamma = 0.99, rewards divided by the target tile, illegal
moves masked), trained the way the original DQN of Mnih et al. (2015) uses
its data instead of with the tutorial's CartPole schedule, so that it can
play many more games:

  - n_envs games are played at the same time: the moves of all of them are
    chosen with one forward pass on the GPU, and the rules run in the numba
    engine of fast2048.py (the same rules as game2048.py);
  - the replay memory holds the last 1 000 000 moves, as in the paper, on the
    GPU (each board stored as 16 exponents, one-hot encoded when sampled);
  - every move is replayed on average replay_ratio times (8 in the paper:
    one minibatch of 32 every 4 moves; 128 in the tutorial: one minibatch of
    128 per move), in minibatches of `batch` moves.
"""
import math
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

import torch.nn as nn

import fast2048 as fb
from fast2048 import njit  # numba imported with the cache directory set
from dqn2048 import DQN, EPS_DECAY, EPS_END, EPS_START, GAMMA, LR, N_VALUES, TAU


class PairCNN(nn.Module):
    """Convolutional network over the pairs of neighbouring cells, the
    pairs that can merge. The input is the one-hot board as 16 channels
    (one per exponent) on the R x C grid. A first layer of 1x2 and 2x1
    convolutions reads the horizontal and vertical pairs; a second layer
    applies 1x2 and 2x1 convolutions to each of them (pairs of pairs). All the
    maps are flattened and joined, then a hidden layer of 256 units gives the
    four action values. Same input and output as the perceptron of dqn2048.py,
    so the rest of the agent does not change."""

    def __init__(self, rows, cols, channels=128, hidden=256):
        super().__init__()
        self.rows, self.cols = rows, cols
        k = {'h': (1, 2), 'v': (2, 1)}
        self.first = nn.ModuleDict({d: nn.Conv2d(N_VALUES, channels, k[d]) for d in k})
        size = {'h': (rows, cols - 1), 'v': (rows - 1, cols)}
        self.second = nn.ModuleDict()
        n_feat = sum(r * c for r, c in size.values()) * channels
        for d1 in k:
            for d2 in k:
                r, c = size[d1][0] - k[d2][0] + 1, size[d1][1] - k[d2][1] + 1
                if r > 0 and c > 0:  # e.g. no vertical pair of pairs on a 2-row board
                    self.second[d1 + d2] = nn.Conv2d(channels, channels, k[d2])
                    n_feat += r * c * channels
        self.hidden = nn.Linear(n_feat, hidden)
        self.out = nn.Linear(hidden, 4)

    def forward(self, x):
        b = x.shape[0]
        x = x.view(b, self.rows * self.cols, N_VALUES).transpose(1, 2).reshape(
            b, N_VALUES, self.rows, self.cols)
        maps = {d: F.relu(conv(x)) for d, conv in self.first.items()}
        feats = [m.flatten(1) for m in maps.values()]
        for name, conv in self.second.items():
            feats.append(F.relu(conv(maps[name[0]])).flatten(1))
        return self.out(F.relu(self.hidden(torch.cat(feats, 1))))


# ----------------------------------------------------------------------
# the games, in numba (boards are the uint64 bitboards of fast2048.py)
# ----------------------------------------------------------------------
@njit(cache=True)
def _observe(xs, rows, cols, exps, masks):
    """Exponents (n, rows*cols) and legal moves (n, 4) of the boards."""
    for k in range(xs.shape[0]):
        x = xs[k]
        for r in range(rows):
            for c in range(cols):
                exps[k, r * cols + c] = fb.nibble(x, 4 * r + c)
        m = fb.legal_mask(x, rows, cols)
        for a in range(4):
            masks[k, a] = ((m >> a) & 1) == 1


@njit(cache=True)
def _step(xs, actions, rows, cols, p4, points, done):
    """Play one legal move on every board (slide, then the random tile)."""
    for k in range(xs.shape[0]):
        after, pts, _ = fb.move(xs[k], actions[k], rows, cols)
        nxt = fb.spawn(after, rows, cols, p4)
        xs[k] = nxt
        points[k] = pts
        done[k] = fb.legal_mask(nxt, rows, cols) == 0


@njit(cache=True)
def _new_boards(xs, idx, rows, cols, p4):
    for j in range(idx.shape[0]):
        xs[idx[j]] = fb.initial_board(rows, cols, p4)


class VecDQN:
    def __init__(self, rows=4, cols=4, target=2048, p4=0.1, n_envs=512, batch=1024,
                 replay_ratio=8, memory=1_000_000, seed=0, device='cuda',
                 reward_scale=None, net='mlp'):
        self.rows, self.cols, self.target, self.p4 = rows, cols, target, p4
        self.n_cells = rows * cols
        self.n_envs, self.batch, self.replay_ratio = n_envs, batch, replay_ratio
        self.updates_per_step = max(1, round(replay_ratio * n_envs / batch))
        self.reward_scale = float(reward_scale or target)
        self.seed = seed
        self.dev = torch.device(device)
        torch.manual_seed(seed)
        fb.seed(seed)
        self.gen = torch.Generator(device=self.dev)
        self.gen.manual_seed(seed)
        n_obs = self.n_cells * N_VALUES
        self.net = net

        def make():
            return PairCNN(rows, cols) if net == 'cnn' else DQN(n_obs, 4)
        self.policy_net = make().to(self.dev)
        self.target_net = make().to(self.dev)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        # replay memory on the GPU
        cap = self.capacity = memory
        dev = self.dev
        self.m_s = torch.zeros((cap, self.n_cells), dtype=torch.uint8, device=dev)
        self.m_s2 = torch.zeros((cap, self.n_cells), dtype=torch.uint8, device=dev)
        self.m_a = torch.zeros(cap, dtype=torch.int64, device=dev)
        self.m_r = torch.zeros(cap, dtype=torch.float32, device=dev)
        self.m_m2 = torch.zeros((cap, 4), dtype=torch.bool, device=dev)
        self.m_done = torch.zeros(cap, dtype=torch.bool, device=dev)
        self.size = self.pos = 0
        # the games being played
        self.xs = np.zeros(n_envs, dtype=np.uint64)
        _new_boards(self.xs, np.arange(n_envs, dtype=np.int64), rows, cols, p4)
        self.exps = np.zeros((n_envs, self.n_cells), dtype=np.int8)
        self.masks = np.zeros((n_envs, 4), dtype=np.bool_)
        _observe(self.xs, rows, cols, self.exps, self.masks)
        self.ep_score = np.zeros(n_envs)
        self.ep_moves = np.zeros(n_envs, dtype=np.int64)
        self.finished = []  # (score, max exponent, moves) of the games ended since the last report
        self.steps = self.games = self.updates = 0
        self.points = np.zeros(n_envs, dtype=np.float32)
        self.done = np.zeros(n_envs, dtype=np.bool_)

    def onehot(self, exps):
        return F.one_hot(exps.long().clamp_(max=N_VALUES - 1), N_VALUES).flatten(1).float()

    def epsilon(self):
        return EPS_END + (EPS_START - EPS_END) * math.exp(-self.steps / EPS_DECAY)

    def greedy(self, exps, masks):
        with torch.no_grad():
            q = self.policy_net(self.onehot(exps))
        q[~masks] = -float('inf')
        return q.argmax(1)

    def prefill(self, n_moves):
        """Fill the replay memory with n_moves moves of the current policy,
        without learning and without counting the games (used after
        warm_start, so that training does not restart from an empty memory)."""
        games, finished = self.games, self.finished
        target = min(self.size + n_moves, self.capacity)
        while self.size < target:
            self.step(learn=False)
        self.games, self.finished = games, finished

    def step(self, learn=True):
        """One move in every game, the transitions into the memory, then the
        gradient steps of these moves."""
        n, dev = self.n_envs, self.dev
        s = torch.from_numpy(self.exps).to(dev)
        m = torch.from_numpy(self.masks).to(dev)
        a = self.greedy(s, m)
        explore = torch.rand(n, device=dev, generator=self.gen) < self.epsilon()
        if bool(explore.any()):
            rand = torch.multinomial(m.float(), 1, generator=self.gen).squeeze(1)
            a = torch.where(explore, rand, a)
        _step(self.xs, a.cpu().numpy().astype(np.int64), self.rows, self.cols, self.p4,
              self.points, self.done)
        _observe(self.xs, self.rows, self.cols, self.exps, self.masks)
        idx = (torch.arange(n, device=dev) + self.pos) % self.capacity
        self.m_s[idx] = s.to(torch.uint8)
        self.m_a[idx] = a
        self.m_r[idx] = torch.from_numpy(self.points).to(dev) / self.reward_scale
        self.m_s2[idx] = torch.from_numpy(self.exps).to(dev).to(torch.uint8)
        self.m_m2[idx] = torch.from_numpy(self.masks).to(dev)
        self.m_done[idx] = torch.from_numpy(self.done).to(dev)
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        self.steps += n
        self.ep_score += self.points
        self.ep_moves += 1
        if self.done.any():
            d = np.flatnonzero(self.done)
            for k in d:
                self.finished.append((self.ep_score[k], int(self.exps[k].max()), int(self.ep_moves[k])))
            self.games += len(d)
            self.ep_score[d] = 0.0
            self.ep_moves[d] = 0
            _new_boards(self.xs, d.astype(np.int64), self.rows, self.cols, self.p4)
            _observe(self.xs, self.rows, self.cols, self.exps, self.masks)
        if learn and self.size >= self.batch:
            for _ in range(self.updates_per_step):
                self.update()

    def update(self):
        """One gradient step on a minibatch from the memory (the update of
        dqn2048.DQNAgent.optimize_model) and the soft update of the target."""
        idx = torch.randint(0, self.size, (self.batch,), device=self.dev, generator=self.gen)
        s, a, r = self.onehot(self.m_s[idx]), self.m_a[idx], self.m_r[idx]
        s2, m2, done = self.onehot(self.m_s2[idx]), self.m_m2[idx], self.m_done[idx]
        q = self.policy_net(s).gather(1, a[:, None])
        with torch.no_grad():
            q2 = self.target_net(s2)
            q2[~m2] = -float('inf')
            nxt = q2.max(1).values
            nxt[done] = 0.0
        loss = F.smooth_l1_loss(q, (r + GAMMA * nxt)[:, None])
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
                tp.lerp_(pp, TAU)
        self.updates += 1

    def evaluate(self, n_games, seed):
        """Greedy play (no exploration, no learning) of n_games games at the
        same time; the numba generator is re-seeded afterwards so that the
        training stays reproducible."""
        rows, cols, dev = self.rows, self.cols, self.dev
        fb.seed(seed)
        xs = np.zeros(n_games, dtype=np.uint64)
        _new_boards(xs, np.arange(n_games, dtype=np.int64), rows, cols, self.p4)
        score = np.zeros(n_games)
        moves = np.zeros(n_games, dtype=np.int64)
        max_exp = np.zeros(n_games, dtype=np.int64)
        alive = np.arange(n_games)
        exps = np.zeros((n_games, self.n_cells), dtype=np.int8)
        masks = np.zeros((n_games, 4), dtype=np.bool_)
        while alive.size:
            bx = xs[alive].copy()
            e, mk = exps[:alive.size], masks[:alive.size]
            _observe(bx, rows, cols, e, mk)
            over = ~mk.any(1)
            max_exp[alive] = e.max(1)
            if over.any():
                alive = alive[~over]
                bx = bx[~over]
                e, mk = e[~over], mk[~over]
                if not alive.size:
                    break
            a = self.greedy(torch.from_numpy(e).to(dev), torch.from_numpy(mk).to(dev))
            pts = np.zeros(alive.size, dtype=np.float32)
            dn = np.zeros(alive.size, dtype=np.bool_)
            _step(bx, a.cpu().numpy().astype(np.int64), rows, cols, self.p4, pts, dn)
            xs[alive] = bx
            score[alive] += pts
            moves[alive] += 1
        fb.seed(self.seed * 1_000_003 + self.games)
        return {'score': score, 'moves': moves, 'max_tile': 2 ** max_exp}

    def greedy_boards(self, boards, chunk=65536):
        """Greedy action (-1 at game over) and max Q (score units) for a list
        of board tuples, e.g. every decision state of a small board."""
        xs = np.array([fb.encode(b, self.rows, self.cols) for b in boards], dtype=np.uint64)
        exps = np.zeros((len(xs), self.n_cells), dtype=np.int8)
        masks = np.zeros((len(xs), 4), dtype=np.bool_)
        _observe(xs, self.rows, self.cols, exps, masks)
        actions = np.empty(len(xs), dtype=np.int64)
        values = np.empty(len(xs), dtype=np.float64)
        for i in range(0, len(xs), chunk):
            e = torch.from_numpy(exps[i:i + chunk]).to(self.dev)
            m = torch.from_numpy(masks[i:i + chunk]).to(self.dev)
            with torch.no_grad():
                q = self.policy_net(self.onehot(e))
            q[~m] = -float('inf')
            best = q.max(1)
            a = best.indices.cpu().numpy()
            a[~masks[i:i + chunk].any(1)] = -1
            actions[i:i + chunk] = a
            values[i:i + chunk] = np.where(a >= 0, best.values.cpu().numpy(), 0.0) * self.reward_scale
        return actions, values

    def warm_start(self, path, games):
        """Continue a run of which only the policy network was saved (model.pt):
        the target network starts as a copy of it, the optimiser and the
        replay memory start empty, and the move counter (hence epsilon, at
        its final value) and the game counter continue from the saved run."""
        ck = torch.load(path, map_location=self.dev)
        self.policy_net.load_state_dict(ck['state_dict'])
        self.target_net.load_state_dict(ck['state_dict'])
        self.games = games
        self.steps = int(ck.get('steps', 0))
        self.updates = int(ck.get('updates', 0))
        fb.seed(self.seed * 1_000_003 + games)
        _new_boards(self.xs, np.arange(self.n_envs, dtype=np.int64), self.rows, self.cols, self.p4)
        _observe(self.xs, self.rows, self.cols, self.exps, self.masks)
        self.ep_score[:] = 0.0
        self.ep_moves[:] = 0

    def save_state(self, path):
        """Everything needed to continue exactly: networks, optimiser,
        counters and the replay memory (about 50 MB for 1 000 000 moves)."""
        torch.save({'policy': self.policy_net.state_dict(), 'target': self.target_net.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'games': self.games,
                    'steps': self.steps, 'updates': self.updates, 'pos': self.pos, 'size': self.size,
                    'memory': {k: getattr(self, k).cpu() for k in
                               ('m_s', 'm_s2', 'm_a', 'm_r', 'm_m2', 'm_done')}}, path)

    def load_state(self, path):
        st = torch.load(path, map_location=self.dev)
        self.policy_net.load_state_dict(st['policy'])
        self.target_net.load_state_dict(st['target'])
        self.optimizer.load_state_dict(st['optimizer'])
        self.games, self.steps, self.updates = st['games'], st['steps'], st['updates']
        self.pos, self.size = st['pos'], st['size']
        for k, v in st['memory'].items():
            getattr(self, k).copy_(v.to(self.dev))
        fb.seed(self.seed * 1_000_003 + self.games)
        _new_boards(self.xs, np.arange(self.n_envs, dtype=np.int64), self.rows, self.cols, self.p4)
        _observe(self.xs, self.rows, self.cols, self.exps, self.masks)
        self.ep_score[:] = 0.0
        self.ep_moves[:] = 0

    def save(self, path):
        torch.save({'state_dict': {k: v.cpu() for k, v in self.policy_net.state_dict().items()},
                    'games': self.games, 'steps': self.steps, 'updates': self.updates,
                    'board': (self.rows, self.cols, self.target),
                    'net': self.net, 'n_envs': self.n_envs, 'batch': self.batch,
                    'replay_ratio': self.replay_ratio, 'memory': self.capacity}, path)


def exact_evaluator(evaluator):
    """For a small board: the exact P(win), expected score and share of
    optimal decisions of the greedy policy (an rl2048.ExactEvaluator applied
    to every board of both DP layer sets)."""
    from ntuple2048 import FrozenPolicy
    boards = sorted({b for layers in (evaluator.layers_win, evaluator.layers_score)
                     for layer in layers.values() for b in layer})

    def f(agent):
        actions, values = agent.greedy_boards(boards)
        res = evaluator.evaluate(FrozenPolicy(boards, actions, values, 'score'))
        return {k: res[k] for k in ('p_win', 'score', 'agree')}
    return f


def eval_row(res, prefix='eval_'):
    row = {f'{prefix}score': float(res['score'].mean()), f'{prefix}moves': float(res['moves'].mean()),
           f'{prefix}max_tile': int(res['max_tile'].max())}
    for t in (256, 512, 1024, 2048, 4096, 8192):
        row[f'{prefix}p{t}'] = float((res['max_tile'] >= t).mean())
    return row


def train(agent, n_games, eval_every, eval_games=200, eval_seed=12345, callback=None,
          verbose=True, exact=None):
    """Play until n_games games have ended; every eval_every games record the
    training statistics and a greedy evaluation of eval_games games (and,
    on a small board, the exact evaluation exact(agent))."""
    history = []
    next_eval = (agent.games // eval_every + 1) * eval_every  # a resumed run continues the grid
    t0 = time.perf_counter()
    while agent.games < n_games:
        agent.step()
        if agent.games >= next_eval or agent.games >= n_games:
            fin = np.array(agent.finished) if agent.finished else np.zeros((0, 3))
            agent.finished = []
            row = {'games': agent.games, 'steps': agent.steps, 'updates': agent.updates,
                   'epsilon': agent.epsilon(),
                   'train_score': float(fin[:, 0].mean()) if len(fin) else float('nan'),
                   'train_moves': float(fin[:, 2].mean()) if len(fin) else float('nan'),
                   'train_max_tile': int(2 ** fin[:, 1].max()) if len(fin) else 0,
                   'time_s': time.perf_counter() - t0}
            row.update(eval_row(agent.evaluate(eval_games, eval_seed)))
            if exact is not None:
                row.update(exact(agent))
            history.append(row)
            if verbose:
                extra = f"  exact P(win) {row['p_win']:.4f}" if 'p_win' in row else \
                    f"  P(1024) {row['eval_p1024']:.2f}  P(2048) {row['eval_p2048']:.2f}"
                print(f"{agent.rows}x{agent.cols} s{agent.seed} games {row['games']:>8}  moves "
                      f"{row['steps'] / 1e6:7.2f} M  train score {row['train_score']:8.0f}  "
                      f"eval score {row['eval_score']:8.0f}{extra}  ({row['time_s']:.0f} s)", flush=True)
            if callback is not None:
                callback(agent, history)
            while next_eval <= agent.games:
                next_eval += eval_every
    return history
