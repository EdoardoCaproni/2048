# 2048 as a Markov decision problem

Project work for *Dynamic Programming and Reinforcement Learning* (University of Siena, A.Y. 2024/25), by Edoardo Caproni and Tommaso Quintabà.
The full report is [`report/report.pdf`](report/report.pdf).

**▶ Watch the agents play in the browser: [edoardocaproni.github.io/2048](https://edoardocaproni.github.io/2048/).** In this policy viewer the best policy found by every method plays: on the 2x3 board with the exact optimal win probability of every position and move, on the original 4x4 game with the tabular methods and the deep Q-networks of the final test.

2048 is a puzzle for one player. Every move has two steps: the player slides all the tiles in one direction (deterministic), then a new tile appears in a random empty cell (random). The rules, including the probabilities of the new tile, are completely known, so the game is a Markov decision problem with a known model. Small versions of the game can be solved exactly by dynamic programming; the original 4x4 game cannot, because it has far too many states. The project is built around this gap:

- **Part 1, exact dynamic programming.** On small boards (up to 3x3 cells) we compute the optimal policy exactly, and use it to measure how good the usual rules of thumb are and how much of the game is luck.
- **Part 2, reinforcement learning.** We forget the model and learn from a simulator with the methods of the course. On the small boards every learned policy is measured exactly against the DP solution.

## Contents

1. [The game](#the-game)
2. [Problem formulation](#problem-formulation)
3. [Methods](#methods)
   - [Exact DP by stages](#exact-dp-by-stages)
   - [Value iteration](#value-iteration)
   - [Policy evaluation and benchmark policies](#policy-evaluation-and-benchmark-policies)
   - [Choosing the reward](#choosing-the-reward)
   - [Prediction: TD(0), n-step TD, Monte Carlo](#prediction-td0-n-step-td-monte-carlo)
   - [Tabular control](#tabular-control)
   - [Exploration](#exploration)
   - [Exact learning curves](#exact-learning-curves)
   - [Final test: deep Q-network](#final-test-deep-q-network)
4. [Main results](#main-results)
5. [Repository](#repository)

## The game

The board has $`R\times C`$ cells and the target tile is $`2^K`$ (the original game: $`R=C=4`$, $`K=11`$, tile 2048).

1. A move (down, left, right, up) slides all the tiles as far as possible in that direction. Two equal tiles that collide merge into one tile carrying their sum; a tile created by a merge cannot merge again in the same move. The score grows by the value of every tile created by a merge.
2. A move is legal only if it changes the board.
3. After every legal move a new tile appears in a uniformly random empty cell: a 2 with probability $`1-p_4`$, a 4 with probability $`p_4`$ ($`p_4=0.1`$ in the original game). The game starts with two such tiles.
4. The game is won when a move creates the target tile, and lost when no legal move is left.

## Problem formulation

We use the notation of the DP part of the course (Bertsekas): state $`x`$, control $`u`$, disturbance $`w`$, dynamics $`x_{t+1}=f(x_t,u_t,w_t)`$.

**State.** A board is stored as the exponents of its tiles, cell by cell: $`x=(x_1,\dots,x_{RC})`$, with $`x_i=0`$ for an empty cell and $`x_i=k`$ for the tile $`2^k`$. The state $`x_t`$ is the board the player sees before move $`t`$, after the random tile.

**Control.** The legal moves, the ones that change the board:

```math
U(x)=\{u\in\{\text{down},\text{left},\text{right},\text{up}\} : M(x,u)\neq x\}.
```

The game is over exactly when $`U(x)`$ is empty.

**Afterstate (deterministic part of a move).** $`y=M(x,u)`$ is the board after the slide and the merges, before the new tile. It is the part of the next board that the player controls. The points of the move, $`r(x,u)`$, are the sum of the tiles created by merges.

**Disturbance (random part of a move).** $`w_t=(i_t,v_t)`$: the cell $`i_t`$ of the new tile, uniform over the empty cells $`E(y_t)`$ of the afterstate, and its exponent $`v_t`$, equal to 1 (a 2) with probability $`1-p_4`$ and 2 (a 4) with probability $`p_4`$. Given the afterstate, the new tile does not depend on anything else.

**Dynamics.** Slide, then add the random tile ($`y+v\,e_i`$ is the board $`y`$ with exponent $`v`$ placed in cell $`i`$):

```math
x_{t+1}=f(x_t,u_t,w_t)=M(x_t,u_t)+v_t\,e_{i_t}.
```

**Objective.** Maximise the probability of creating the target tile:

```math
V^*(x)=\max_{\pi}\ P^{\pi}\left(\text{the target tile is created}\mid x_0=x\right).
```

In the cost form of the course this is a problem with no cost per stage and a terminal cost of 1 for a lost game and 0 for a won game; the optimal cost is $`J^*=1-V^*`$, the smallest probability of losing. For comparison we also study the expected score, with the points $`r(x,u)`$ as the reward of each stage.

**Bellman equation.** Each move is split into its two steps. $`V^*(x)`$ is the best win probability from the board $`x`$, before the move; $`W^*(y)`$ is the best win probability from the afterstate $`y`$, before the new tile:

```math
V^*(x)=\begin{cases}0 & \text{if } U(x)=\emptyset,\\ \max_{u\in U(x)} W^*\big(M(x,u)\big) & \text{otherwise,}\end{cases}
```

```math
W^*(y)=\begin{cases}1 & \text{if } \max_i y_i=K,\\ \displaystyle\sum_{i\in E(y)}\frac{1}{|E(y)|}\Big[(1-p_4)\,V^*(y+e_i)+p_4\,V^*(y+2e_i)\Big] & \text{otherwise.}\end{cases}
```

In words: the player picks the move whose afterstate is worth most; an afterstate holding the target is a win; otherwise its value is the average over where the new tile lands and whether it is a 2 or a 4. For the expected score the move is chosen by $`\max_u [r(x,u)+W(M(x,u))]`$ and the game goes on past the target.

**The key property: the tile sum is a clock.** Let $`\sigma(x)=\sum_{i:x_i>0}2^{x_i}`$ be the sum of the tiles. A slide never changes it (a merge turns $`2^k+2^k`$ into $`2^{k+1}`$), and the new tile adds 2 or 4:

```math
\sigma\big(M(x,u)\big)=\sigma(x),\qquad \sigma(x_{t+1})\in\{\sigma(x_t)+2,\ \sigma(x_t)+4\}.
```

Consequences:
- no board is visited twice in a game, and every game ends within $`RC\cdot2^{K-2}`$ moves, whatever the policy. So every policy is proper, and the Bellman equation has a unique solution (the total-cost assumptions of the course hold);
- grouping the boards by tile sum into stages $`X_\sigma`$, every move goes from $`X_\sigma`$ to $`X_{\sigma+2}`$ or $`X_{\sigma+4}`$: the problem can be solved by a single backward pass, like a finite-horizon problem.

## Methods

### Exact DP by stages

1. **Forward pass:** starting from the two initial tiles, generate the reachable boards stage by stage, for increasing tile sum.
2. **Backward pass:** for $`\sigma`$ from the largest to the smallest, and for every board $`x\in X_\sigma`$, compute $`W^*`$ of each afterstate with the Bellman equation and set $`V^*(x)=\max_u W^*(M(x,u))`$.

When a board is processed, every board it can lead to has a larger tile sum and already has its final value, so one pass is enough. The cost is proportional to (number of states) x (moves) x (possible new tiles). This solves the 2x3 board with target 64 (27,022 states) in a fraction of a second and the 3x3 board with target 64 (4.8 million states) in four minutes, in pure Python.

### Value iteration

The algorithm of the course for total-cost problems: start from $`V^{(0)}=0`$ and apply the Bellman equation to all the states at every iteration,

```math
V^{(i+1)}(x)=\max_{u\in U(x)} W^{(i)}\big(M(x,u)\big),
```

where $`W^{(i)}`$ is the afterstate equation computed with $`V^{(i)}`$. Started from zero, $`V^{(i)}(x)`$ is the best probability of winning within $`i`$ moves. Since every game is bounded, value iteration is exact after finitely many iterations (46 on the 2x3 board with target 64), not only in the limit. The backward pass above is the same iteration with the states updated in place (Gauss-Seidel) in decreasing tile sum: its first sweep already gives $`V^*`$.

### Policy evaluation and benchmark policies

Replacing the maximisation with the move chosen by a fixed policy $`\mu`$ gives the exact value of that policy, with the same backward pass:

```math
V_\mu(x)=W_\mu\big(M(x,\mu(x))\big).
```

We use it to evaluate three rules of thumb exactly: the **corner rule** (the first legal move in the order down, left, right, up: keep the big tiles in a corner), the **greedy rule** (the move with the most points) and **random play**. In the RL part, the same evaluation measures every learned policy.

### Choosing the reward

A learning agent only knows what we want through its reward. Our first idea was to reward what good players do. For a move $`(x,u)`$ with afterstate $`y=M(x,u)`$:

```math
r_{\text{naive}}(x,u)=\underbrace{2^{\max_i y_i}\,\mathbf{1}\big[\max_i y_i>\max_i x_i\big]}_{\text{new largest tile}}+\underbrace{(2+2p_4)}_{\text{growth of the tile sum}}+\underbrace{\big(|E(y)|-|E(x)|-1\big)}_{\text{change in free cells}}.
```

Adding up each part over a game shows what it really asks for:
- **tile sum:** by the clock property the move never changes the sum, so this part pays the same for every move: it only rewards surviving;
- **free cells:** the changes add up to $`-|E(x_0)|`$, because a lost game ends on a full board. It is a constant, so it cannot change the best policy (it shifts every value $`V(x)`$ by exactly $`-|E(x)|`$);
- **new largest tile:** the bonuses add up to about twice the largest tile reached, a graded version of the win.

The exact DP confirms it on the 2x3 board: the optimal policy of the naive reward wins 0.9387 of the games, exactly as the score-optimal policy, against 0.9393 for the win-optimal one; with or without the free-cell part it picks the same move on all 68,340 boards. Trained with the naive reward, Sarsa on afterstates learns as fast as with the score (0.797 against 0.796 after 200,000 episodes). The naive reward works, but no better than the score, which the game provides and which has no weights to tune. We therefore use two rewards:
- **win:** $`+1`$ when a move creates the target (the episode ends), 0 otherwise. The value of a state is its win probability, so $`v_*=V^*`$. It is the goal itself, but sparse: random play wins 2 games in 10,000 on the 2x3 board.
- **score:** the points $`r(x,u)`$ of each move. It is dense and almost aligned with the goal: the score-optimal policy makes the win-optimal move in 97.9% of the decisions.

Every policy is measured by its exact win probability, whatever reward it learned from.

### Prediction: TD(0), n-step TD, Monte Carlo

Prediction estimates the value $`v_\pi`$ of a fixed policy from its episodes alone. We take the optimal policy and the win reward, so the true answer $`v_\pi=V^*`$ is known exactly. All methods use a constant step size $`\alpha`$ and $`\gamma=1`$.

- **TD(0)** moves the estimate of a state towards the reward plus the estimate of the next state:

  ```math
  V(S_t)\leftarrow V(S_t)+\alpha\big[R_{t+1}+V(S_{t+1})-V(S_t)\big].
  ```

- **n-step TD** looks $`n`$ steps ahead before using an estimate ($`n=2,4,8,16`$):

  ```math
  G_{t:t+n}=R_{t+1}+\dots+R_{t+n}+V(S_{t+n}),\qquad V(S_t)\leftarrow V(S_t)+\alpha\big[G_{t:t+n}-V(S_t)\big].
  ```

- **Constant-$`\alpha`$ Monte Carlo** uses the full return of the game, $`G_t`$, as target: the limit $`n\to\infty`$.

The error is the root mean square error against $`V^*`$, weighted by how often the policy visits each state. On this game the ranking of the textbook random walk is reversed at the start: Monte Carlo is faster than TD(0), because the only reward comes at the end of a 33-move game and TD(0) passes it back one state per visit. With a longer budget an intermediate $`n`$ wins (8-step TD).

### Tabular control

Six control methods of the course, each keeping a table of values and improving the policy as the values change.

- **Monte Carlo control** (on-policy, first visit, $`\epsilon`$-soft): after each game, $`Q(S_t,A_t)`$ moves to the average of the observed returns $`G_t`$. Since no state repeats within a game, first-visit and every-visit coincide.
- **Sarsa** (on-policy TD): uses the next move actually played,

  ```math
  Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\big].
  ```

- **Q-learning** (off-policy TD): uses the best next move, so it learns the optimal values $`q_*`$ whatever the exploration,

  ```math
  Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\max_a Q(S_{t+1},a)-Q(S_t,A_t)\big].
  ```

- **Expected Sarsa:** uses the average over the next moves of the $`\epsilon`$-greedy policy, a target with less variance than Sarsa's,

  ```math
  Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\Big[R_{t+1}+\sum_a \pi(a\mid S_{t+1})\,Q(S_{t+1},a)-Q(S_t,A_t)\Big].
  ```

- **n-step Sarsa** ($`n=3`$): Sarsa with the target $`R_{t+1}+R_{t+2}+R_{t+3}+Q(S_{t+3},A_{t+3})`$.
- **Sarsa on afterstates.** Two different moves from two different boards can lead to the same afterstate, and from there the future is the same. So we keep one value per afterstate and write the value of a move as

  ```math
  Q(x,u)=r(x,u)+V\big(M(x,u)\big).
  ```

  This is exact, not an approximation: the true value of a move depends on $`(x,u)`$ only through its points and its afterstate. The agent uses the slide rule $`M`$ but never the probabilities of the new tile. With this $`Q`$ the reward cancels in the Sarsa error, and the update becomes an update of one afterstate value ($`Y_t=M(S_t,A_t)`$, target 0 at game over):

  ```math
  V(Y_t)\leftarrow V(Y_t)+\alpha\big[r(S_{t+1},A_{t+1})+V(Y_{t+1})-V(Y_t)\big].
  ```

  Every update improves all the moves, from all the boards, that lead to the same afterstate, which is why this method learns fastest. The same update with a linear approximation of $`V`$ is the core of the strongest published 2048 players (Szubert & Jaśkowski 2014 and their successors; see the report, Section 7).

### Exploration

- **$`\epsilon`$-greedy:** with probability $`\epsilon`$ a uniformly random legal move, otherwise the move with the largest $`Q`$ (ties broken in the order down, left, right, up, so an agent with all-zero values plays the corner rule).
- **Optimistic initial values:** start every $`Q`$ at 1, the largest possible win probability. Every untried move looks promising until it has been tried, so the agent explores every move once and then follows the best. With the sparse win reward this removes the dead start of a learner that has never seen a win.
- **Step-size and exploration schedules:** constant $`\alpha`$ and $`\epsilon`$ against linearly decaying ones ($`\epsilon`$ from 0.3 to 0.01, $`\alpha`$ from 0.2 to 0.02), and a grid over constant values.

### Exact learning curves

On the small boards the DP solution turns every learning curve into an exact measurement. At regular checkpoints the greedy policy of the agent is evaluated exactly by policy evaluation (win probability and expected score), and the learned values are compared with $`V^*`$:

```math
\text{RMSE}=\sqrt{\sum_x \mu^*(x)\big(\hat V(x)-V^*(x)\big)^2\Big/\sum_x \mu^*(x)},
```

where $`\mu^*(x)`$ is the expected number of visits of board $`x`$ under the optimal policy (computed by a forward pass). The curves have no Monte Carlo noise: all their variation comes from learning.

### Final test: deep Q-network

Outside the methods of the course, the RL part ends with a deep Q-network built as in the PyTorch tutorial: Q-learning with a neural network $`Q(x,u;\theta)`$, trained on random minibatches of past transitions (experience replay) towards targets computed by a slowly updated copy of the network (target network $`\theta'`$):

```math
\text{loss}=\text{Huber}\Big(r+\gamma\max_{u'}Q(x',u';\theta')-Q(x,u;\theta)\Big).
```

The input is the one-hot encoding of the exponent of every cell; illegal moves are masked. It is tested on the 2x3 board, where it is evaluated exactly, and on the original 4x4 game. With the tutorial's schedule (one gradient step per move) it is slow: 2,314 points after 10,000 games of the 4x4 game, below the rules of thumb. Playing 1,024 games at once on a GPU, with a replay memory of a million moves (as in the original DQN), it reaches 0.871 of games won on 2x3 and 8,862 points on 4x4 after 100,000 games; with a convolutional network over the pairs of neighbouring cells, 17,485 points and the 2048 tile in 36% of the games. Neither curve is still rising at 100,000 games, and the n-tuple players of the literature, which are given the slide rule through the afterstates, stay far ahead (2048 in over 97% of the games).

## Main results

| | 2x3 board, target 64 |
|---|---|
| Reachable states | 27,022 in 55 stages |
| Optimal policy (exact DP) | wins 93.93% of the games; maximum expected score 480.3 |
| Corner / greedy / random rule | 0.07% / 0.13% / 0.02% |
| Best tabular learner, win reward with optimism | Q-learning 0.875; Sarsa on afterstates 0.848 |
| Best tabular learner, score reward | Sarsa on afterstates 0.796 (0.85 with decaying schedules) |
| DQN (final test), score reward | 0.812 with the tutorial's schedule (20,000 games); 0.871 with many games on the GPU (200,000) |
| Prediction after 20,000 games (error against $`V^*`$) | 8-step TD 0.080, Monte Carlo 0.085, TD(0) 0.114 |

| | 4x4 board, target 2048: mean score, tiles reached |
|---|---|
| Corner / greedy rule | 2,564 / 3,027 points; 512 in 5% / 12% of the games |
| Tabular methods, settings of the 2x3 runs (200,000 episodes) | from 2,163 (Sarsa on afterstates) to 2,390 points (Q-learning), below the corner rule; 50-74% of the moves on boards never met in training |
| DQN, tutorial's schedule (10,000 games) | 2,314 points; 512 in 4%, never 1024 |
| DQN on the GPU, perceptron (100,000 games) | 8,862 points; 1024 in 41%, 2048 in 3% |
| DQN on the GPU, convolutional network (100,000 games) | 17,485 points; 1024 in 73%, 2048 in 36%, 4096 in 1.3% |

Win probabilities on 2x3 are exact and averaged over the seeds; scores on 4x4 are means over 1,000 games (10,000 for the rules) and over the seeds (five for the tabular methods, three for the DQN, two for the convolutional network).

**The tabular methods on the 4x4 game** (report, Section 6.5). The six methods run on the original game unchanged, with the settings of the 2x3 runs (score reward, alpha 0.1, MC control with sample averages, epsilon 0.1, 200,000 episodes, 5 seeds). `rlfast2048.py` holds the same algorithms as `rl2048.py` with the table in a hash table keyed by the 64-bit board; on 2x3, with the settings of Table 7 of the report, its exact win probabilities agree with the report within the spread of the seeds (`results/tabular_4x4/port_check_2x3.csv`). After 200,000 episodes each table holds 23-28 million boards, nearly all met only once, and in a new game 69-74% of the moves fall on boards never met in training (50% for Sarsa on afterstates, which needs one known afterstate out of four). There the table has no value and the corner rule plays: all six methods stay below the corner rule itself (2,564 points), and more training does not raise the score. More exploration makes it worse: with epsilon 0.2 and 0.3 every method scores less and meets fewer of the boards of its own greedy play, because the random moves scatter the training games over boards that the greedy policy never meets. A table cannot generalise from one board to another: this is the curse of dimensionality that the report measures with the state counts, and the reason why the final test needs a network that generalises.

| Tabular method on 4x4 | epsilon 0.1 | epsilon 0.2 | epsilon 0.3 |
|---|---|---|---|
| Monte Carlo control | 2,287 (74%) | 2,120 (82%) | 2,129 (86%) |
| Sarsa | 2,372 (69%) | 2,319 (80%) | 2,279 (84%) |
| Expected Sarsa | 2,367 (69%) | 2,292 (79%) | 2,289 (84%) |
| Q-learning | 2,390 (69%) | 2,329 (80%) | 2,319 (84%) |
| 3-step Sarsa | 2,355 (72%) | 2,207 (81%) | 2,190 (85%) |
| Sarsa on afterstates | 2,163 (50%) | 1,958 (57%) | 1,901 (62%) |

Mean score of the greedy policy (1,000 games, mean over 5 seeds) and, in brackets, the share of its moves on boards never met in training; the corner rule scores 2,564 points, the greedy rule 3,027.

Exact DP solves boards up to 3x3 (4.8 million states); the 4x4 game has up to $`11^{16}\approx4.6\cdot10^{16}`$ states. On the small boards the corner rule is far from optimal (0.07% against 93.9% on 2x3/64, 43% against 100% on 3x3/64): on a cramped board what matters is whether a merge survives the next random tile, not where the largest tile sits. Luck matters too: with only 2s the 2x3 game is always won, and at $`p_4=0.5`$ even the optimal player wins only 72.7%.

## Repository

### Setup

Python 3.10 or later. From this folder:

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
python test_2048.py             # about one minute: rules, DP, engines, learning, DQN smoke test
```

### Code

| File | Content |
|---|---|
| `game2048.py` | Rules of the game on an R x C board (pure Python, the reference implementation), `Env2048` with the Gymnasium `reset`/`step` interface |
| `dp2048.py` | Exact DP: reachable states by stage, backward recursion, value iteration (total cost; Jacobi or Gauss-Seidel by stage), exact policy evaluation, occupancy, Monte Carlo simulation |
| `fast2048.py` | The same rules on a 64-bit bitboard compiled with numba (about 50x faster); `test_2048.py` checks it against `game2048.py` move by move |
| `predict2048.py` | Tabular prediction of a fixed policy: TD(0), n-step TD and constant-alpha Monte Carlo, with the error against the exact values |
| `shaping2048.py` | The rules-of-thumb ("naive") reward: a bonus for each new largest tile, the growth of the tile sum, the change of free cells; usable as a reward name wherever `'win'`/`'score'` are |
| `rl2048.py` | Tabular control agents on `Env2048`: MC control, Sarsa, Q-learning, Expected Sarsa, n-step Sarsa, Sarsa on afterstates; `ExactEvaluator` computes the exact value of a learned policy with the DP |
| `rlfast2048.py` | The same six tabular methods on the bitboard engine, with the table in a hash table keyed by the 64-bit board, for the 4x4 game (over a million moves per second) |
| `dqn2048.py` | Final test: DQN as in the PyTorch tutorial (replay memory, target network, Huber loss, AdamW), with legal-move masking; needs PyTorch |
| `dqn_gpu2048.py` | The same agent on the GPU with 1,024 games at once (rules in the numba engine), a replay memory of a million moves, the perceptron or a convolutional network (`PairCNN`); needs PyTorch with CUDA |
| `ntuple2048.py` | Not part of the report: a re-implementation of the n-tuple players of the literature (semi-gradient Sarsa on afterstates with binary n-tuple features), kept for exploration and used by the 4x4 policy viewer |
| `viz2048.py` | Animations of the policies playing (`python viz2048.py -h`) |
| `viewer/` | Policy viewer, online at https://edoardocaproni.github.io/2048/ or opened locally (`viewer/index.html`): the best policy of every method plays. On the 2x3 board every policy is a table over the decision states, shown with the exact optimal win probability of every position and move; on the 4x4 board the page plays the rules of thumb, the six tabular methods (the 50,000 boards each table met most often in training, the corner rule elsewhere), the three DQNs of the final test (the convolutional network with its weights in half precision) and an n-tuple player that is not part of the report. Data written by `experiments/export_viewer.py`; the page loads the 4x4 agents (8 MB) only when that board is chosen, and the tables (5 MB) only when one of them is |
| `.github/workflows/pages.yml` | Publishes `viewer/` as the GitHub page at every push to `main` that changes it. One-time setup by the owner of the repository: Settings > Pages > Build and deployment > Source: GitHub Actions |
| `2048_DP.ipynb` | Notebook of Part 1 |

### Reproducing the results

All the scripts write to `results/`; run them from this folder with the venv active.

```
python experiments/dp_figures.py            # Part 1 figures and tables (add 'big' for the 3x3 board, 5 min)
python experiments/dp_value_iteration.py    # value iteration against the backward recursion (2 min)
python experiments/bench_engine.py          # throughput of the two engines
python experiments/exp_reward.py dp         # optimal policy of each reward, exact (a few minutes)
python experiments/exp_reward.py tabular    # Sarsa on afterstates with the naive reward, 5 seeds (15 min)
python experiments/plot_reward.py
python experiments/exp_prediction.py        # TD(0), n-step TD and MC prediction, 50 runs (under a minute)
python experiments/exp_tabular_2x3.py       # tabular methods, 90 runs x 200k episodes (about 1 h on 26 cores)
python experiments/plot_tabular.py
python experiments/exp_tabular_grid.py      # alpha x epsilon grid and decaying schedules (130 runs)
python experiments/plot_schedule.py
python experiments/exp_tabular_4x4.py check # the fast tabular code on 2x3, exact P(win) against the report (a minute)
python experiments/exp_tabular_4x4.py       # the six tabular methods on 4x4, 5 seeds x 200k episodes (3 min, 3 GB of RAM per process)
python experiments/exp_tabular_4x4.py eps=0.2,0.3   # the same with more exploration (6 min)
python experiments/plot_tabular_4x4.py
python experiments/eval_4x4.py              # the rules of thumb on the 4x4 board (and any n-tuple checkpoints present)
python experiments/exp_dqn.py               # DQN final test, tutorial's schedule: 3 seeds on 2x3 and on 4x4, on the CPU (about 1 h)
python experiments/exp_dqn_gpu.py           # DQN on the GPU, perceptron: 3 seeds on 2x3 and on 4x4 (about 20 min)
python experiments/exp_dqn_gpu.py 4x4 net=cnn games=100000 eval_every=5000 seeds=0,1   # convolutional network (about 1.5 h)
python experiments/plot_dqn_gpu.py
python experiments/export_viewer.py         # data of the viewer: retrains the best tabular policies, exports the DQN networks (a few minutes; the 4x4 n-tuple player trains for about 1 h the first time)
```

Not part of the report (exploration of the n-tuple players of the literature):

```
python experiments/exp_ntuple_small.py      # semi-gradient Sarsa, n-tuple features, on 2x3, 2x4, 3x3 with exact validation
python experiments/plot_ntuple_small.py
python experiments/exp_4x4.py               # 4x4 training grid, 300k games per run (hours; '100000 rcs' for a short run)
python experiments/plot_4x4.py
```

The report is compiled with `pdflatex` and `bibtex` in `report/` after copying the figures from `results/` (`python experiments/collect_figures.py`).
