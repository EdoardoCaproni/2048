# 2048 as a Markov decision problem

Project work for *Dynamic Programming and Reinforcement Learning* (University of Siena, A.Y. 2024/25), by Edoardo Caproni and Tommaso Quintabà.

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
   - [Extra: Sarsa with n-tuple features](#extra-sarsa-with-n-tuple-features)
4. [Main results](#main-results)
5. [Repository](#repository)

## The game

The board has $`R\times C`$ cells and the target tile is $`2^K`$ (the original game: $`R=C=4`$, $`K=11`$, tile 2048).

1. A move (down, left, right, up) slides all the tiles as far as possible in that direction. Two equal tiles that collide merge into one tile carrying their sum; a tile created by a merge cannot merge again in the same move. The score grows by the value of every tile created by a merge.
2. A move is legal only if it changes the board.
3. After every legal move a new tile appears in a uniformly random empty cell: a 2 with probability $`1-p_4`$, a 4 with probability $`p_4`$ ($`p_4=0.1`$ in the original game). The game starts with two such tiles.
4. The game is won when a move creates the target tile, and lost when no legal move is left.

## Problem formulation

We use the notation of the DP part of the course (Bertsekas): state $`x`$, control $`u`$, disturbance $`w`$, dynamics $`x_{t+1}=f(x_t,u_t,w_t)`$. The RL part switches to the notation of Sutton & Barto, as the course does. Every symbol is explained where it first appears.

**State.** A board is stored as the exponents of its tiles, cell by cell: $`x=(x_1,\dots,x_{RC})`$, with $`x_i=0`$ for an empty cell and $`x_i=k`$ for the tile $`2^k`$. The index $`t=0,1,2,\dots`$ counts the moves, and the state $`x_t`$ is the board the player sees before move $`t`$, after the random tile.

**Control.** The control $`u`$ is the move. $`U(x)`$ is the set of legal moves of the board $`x`$, the ones that change it:

```math
U(x)=\{u\in\{\text{down},\text{left},\text{right},\text{up}\} : M(x,u)\neq x\}.
```

The game is over exactly when $`U(x)`$ is empty.

**Afterstate (deterministic part of a move).** $`y=M(x,u)`$ is the board after the slide and the merges, before the new tile. It is the part of the next board that the player controls. The points of the move, $`r(x,u)`$, are the sum of the tiles created by merges.

**Disturbance (random part of a move).** $`w_t=(i_t,v_t)`$: the cell $`i_t`$ of the new tile, uniform over the set $`E(y_t)`$ of the empty cells of the afterstate (there are $`|E(y_t)|`$ of them), and its exponent $`v_t`$, equal to 1 (a 2) with probability $`1-p_4`$ and 2 (a 4) with probability $`p_4`$. Given the afterstate, the new tile does not depend on anything else.

**Dynamics.** Slide, then add the random tile. Here $`e_i`$ is the board with a 1 in cell $`i`$ and 0 elsewhere, so $`y+v\,e_i`$ is the board $`y`$ with the exponent $`v`$ placed in the empty cell $`i`$:

```math
x_{t+1}=f(x_t,u_t,w_t)=M(x_t,u_t)+v_t\,e_{i_t}.
```

**Objective.** Maximise the probability of creating the target tile:

```math
V^*(x)=\max_{\pi}\ P^{\pi}\left(\text{the target tile is created}\mid x_0=x\right).
```

Read it from the inside out. A policy $`\pi`$ is a rule that picks the move on every board. $`P^{\pi}(\dots\mid x_0=x)`$ is the probability of winning when the game starts from the board $`x`$ and the player chooses every move with the policy $`\pi`$: the moves follow the rule, the new tiles are random, and the probability is taken over the new tiles. The maximum is over all the policies, so $`V^*(x)`$ is the win probability of the best possible player starting from $`x`$. Throughout, a star marks a value of the best policy ("optimal").

In cost form this is a problem with no cost per stage and a terminal cost of 1 for a lost game and 0 for a won game; the optimal cost is $`J^*=1-V^*`$, the smallest probability of losing. For comparison we also study the expected score, with the points $`r(x,u)`$ as the reward of each stage.

**Bellman equation.** Each move has two steps: first the player chooses a slide, then chance places the new tile. We give a value to the board at both moments:
- $`V^*(x)`$ is the win probability of the best player from the board $`x`$, the moment *before the player chooses* a move;
- $`W^*(y)`$ is the win probability of the best player from the afterstate $`y`$, the moment *after the slide and before the new tile*. The player has already moved, so here nothing is chosen: the next thing that happens is the random tile, and after it the player sees a new board $`x'`$ and plays its best from there. So $`W^*(y)`$ is the average of $`V^*`$ over the boards $`x'`$ that the new tile can produce from $`y`$.

$`W^*`$ is not a new quantity of the game, only a name for this intermediate average. With it the two steps become two short equations, one for the choice (a maximum) and one for the chance (an average):

```math
V^*(x)=\begin{cases}0 & \text{if } U(x)=\emptyset,\\ \max_{u\in U(x)} W^*\big(M(x,u)\big) & \text{otherwise,}\end{cases}
```

```math
W^*(y)=\begin{cases}1 & \text{if } \max_i y_i=K,\\ \displaystyle\sum_{i\in E(y)}\frac{1}{|E(y)|}\Big[(1-p_4)\,V^*(y+e_i)+p_4\,V^*(y+2e_i)\Big] & \text{otherwise.}\end{cases}
```

In words: the player picks the move whose afterstate is worth most, and a board with no legal move is lost. An afterstate holding the target ($`\max_i y_i=K`$: the largest exponent on the board is $`K`$, the tile $`2^K`$) is a win. Otherwise the new tile goes to each of the $`|E(y)|`$ empty cells with probability $`1/|E(y)|`$, and is a 2 (exponent 1, board $`y+e_i`$) with probability $`1-p_4`$ or a 4 (exponent 2, board $`y+2e_i`$) with probability $`p_4`$; $`W^*(y)`$ is the average of $`V^*`$ over these boards, each weighted by its probability.

**Bellman equation for the expected score.** The same two steps, with values that count points instead of probabilities: $`V^*_{\text{score}}(x)`$ is the largest expected number of points still to come from the board $`x`$, before the move, and $`W^*_{\text{score}}(y)`$ the same from the afterstate $`y`$, before the new tile:

```math
V^*_{\text{score}}(x)=\begin{cases}0 & \text{if } U(x)=\emptyset,\\ \max_{u\in U(x)} \Big[r(x,u)+W^*_{\text{score}}\big(M(x,u)\big)\Big] & \text{otherwise,}\end{cases}
```

```math
W^*_{\text{score}}(y)=\sum_{i\in E(y)}\frac{1}{|E(y)|}\Big[(1-p_4)\,V^*_{\text{score}}(y+e_i)+p_4\,V^*_{\text{score}}(y+2e_i)\Big].
```

Two things change. The player now picks the move with the most points now plus the most expected points later, $`r(x,u)+W^*_{\text{score}}`$, instead of the move with the best afterstate alone. And the case "the afterstate holds the target" is gone: creating the target does not end the game, which goes on until no legal move is left. A game over still gives 0, because no more points can be made. The tile sum is still a clock, so the same backward pass solves this equation too.

**The key property: the tile sum is a clock.** Let $`\sigma(x)=\sum_{i:x_i>0}2^{x_i}`$ be the sum of the tiles. A slide never changes it (a merge turns $`2^k+2^k`$ into $`2^{k+1}`$), and the new tile adds 2 or 4:

```math
\sigma\big(M(x,u)\big)=\sigma(x),\qquad \sigma(x_{t+1})\in\{\sigma(x_t)+2,\ \sigma(x_t)+4\}.
```

Consequences:
- no board is visited twice in a game, and every game ends within $`RC\cdot2^{K-2}`$ moves, whatever the policy. So every policy is proper (it ends the game for sure), and the Bellman equation has a unique solution;
- grouping the boards by tile sum into stages $`X_\sigma`$, every move goes from $`X_\sigma`$ to $`X_{\sigma+2}`$ or $`X_{\sigma+4}`$: the problem can be solved by a single backward pass, like a finite-horizon problem.

## Methods

### Exact DP by stages

1. **Forward pass:** starting from the two initial tiles, generate the reachable boards stage by stage, for increasing tile sum.
2. **Backward pass:** for $`\sigma`$ from the largest to the smallest, and for every board $`x\in X_\sigma`$, compute $`W^*`$ of each afterstate with the Bellman equation and set $`V^*(x)=\max_u W^*(M(x,u))`$.

When a board is processed, every board it can lead to has a larger tile sum and already has its final value, so one pass is enough. The cost is proportional to (number of states) x (moves) x (possible new tiles). This solves the 2x3 board with target 64 (27,022 states) in a fraction of a second and the 3x3 board with target 64 (4.8 million states) in four minutes, in pure Python.

### Value iteration

Start from $`V^{(0)}=0`$ and apply the Bellman equation to all the states at every iteration,

```math
V^{(i+1)}(x)=\max_{u\in U(x)} W^{(i)}\big(M(x,u)\big),
```

where the superscript $`(i)`$ is the iteration number and $`W^{(i)}`$ is the afterstate equation computed with $`V^{(i)}`$. Started from zero, $`V^{(i)}(x)`$ is the best probability of winning within $`i`$ moves. Since every game is bounded, value iteration is exact after finitely many iterations (46 on the 2x3 board with target 64), not only in the limit. The backward pass above is the same iteration with the states updated in place (Gauss-Seidel) in decreasing tile sum: its first sweep already gives $`V^*`$.

### Policy evaluation and benchmark policies

Replacing the maximisation with the move chosen by a fixed policy $`\mu`$ (the DP part of the course writes $`\mu`$ for a policy; the RL part writes $`\pi`$) gives the exact value of that policy, with the same backward pass:

```math
V_\mu(x)=W_\mu\big(M(x,\mu(x))\big).
```

Here $`\mu(x)`$ is the move the policy plays on the board $`x`$, and $`W_\mu`$ is the afterstate equation with $`V_\mu`$ in place of $`V^*`$.

We use it to evaluate three rules of thumb exactly: the **corner rule** (the first legal move in the order down, left, right, up: keep the big tiles in a corner), the **greedy rule** (the move with the most points) and **random play**. In the RL part, the same evaluation measures every learned policy.

### Choosing the reward

A learning agent only knows what we want through its reward. Our first idea was to reward what good players do. For a move $`(x,u)`$ with afterstate $`y=M(x,u)`$:

```math
r_{\text{naive}}(x,u)=\underbrace{2^{\max_i y_i}\,\mathbf{1}\big[\max_i y_i>\max_i x_i\big]}_{\text{new largest tile}}+\underbrace{(2+2p_4)}_{\text{growth of the tile sum}}+\underbrace{\big(|E(y)|-|E(x)|-1\big)}_{\text{change in free cells}}.
```

The first part pays the value of the largest tile when the move creates a new largest tile ($`\mathbf 1[\cdot]`$ is 1 if the condition holds, 0 otherwise); the second is the average growth of the tile sum, 2 or 4; the third is the change in the number of empty cells from $`x`$ to the next board (the new tile fills one, hence the $`-1`$). Adding up each part over a game shows what it really asks for:
- **tile sum:** by the clock property the move never changes the sum, so this part pays the same for every move: it only rewards surviving;
- **free cells:** the changes add up to $`-|E(x_0)|`$, because a lost game ends on a full board. It is a constant, so it cannot change the best policy (it shifts every value $`V(x)`$ by exactly $`-|E(x)|`$);
- **new largest tile:** the bonuses add up to about twice the largest tile reached, a graded version of the win.

The exact DP confirms it on the 2x3 board: the optimal policy of the naive reward wins 0.9387 of the games, exactly as the score-optimal policy, against 0.9393 for the win-optimal one; with or without the free-cell part it picks the same move on all 68,340 boards. Trained with the naive reward, Sarsa on afterstates learns as fast as with the score (0.797 against 0.796 after 200,000 episodes). The naive reward works, but no better than the score, which the game provides and which has no weights to tune. We therefore use two rewards:
- **win:** $`+1`$ for the move that creates the target (which ends the game), 0 for every other move. In RL the *return* of a game is the sum of the rewards collected until it ends, and the *value* $`v_\pi(x)`$ of a board is the expected return when the game continues from $`x`$ with the policy $`\pi`$. With this reward the return is 1 for a won game and 0 for a lost one, and the expected value of something that is either 1 or 0 is the probability that it is 1. So $`v_\pi(x)`$ is the win probability of $`\pi`$ from $`x`$, and the best value $`v_*(x)=\max_\pi v_\pi(x)`$ is the best win probability: exactly the $`V^*(x)`$ computed by the DP. The learners therefore aim at the very numbers of Part 1, which is why we can measure them exactly. This reward is the goal itself, but sparse: random play wins 2 games in 10,000 on the 2x3 board.
- **score:** the points $`r(x,u)`$ of each move. The return is the points scored from the board $`x`$ to the end of the game (the whole score, for a game counted from its start), so $`v_*`$ is the $`V^*_{\text{score}}`$ of the DP. It is dense and almost aligned with the goal: the score-optimal policy makes the win-optimal move in 97.9% of the decisions.

Every policy is measured by its exact win probability, whatever reward it learned from.

### Prediction: TD(0), n-step TD, Monte Carlo

From here on we use the RL notation: $`S_t`$ is the board before move $`t`$ (the $`x_t`$ of Part 1), $`A_t`$ the move played, $`R_{t+1}`$ its reward (the index is $`t+1`$ because the reward arrives with the next board), and an episode is one game. Prediction estimates the value $`v_\pi`$ of a fixed policy $`\pi`$ (the expected return from each board) from its episodes alone. We take the optimal policy and the win reward, so the true answer $`v_\pi=V^*`$ is known exactly. All methods use a constant step size $`\alpha`$ (how far each update moves the estimate) and $`\gamma=1`$ (no discount).

**TD(0)** keeps a table $`V`$ of estimates, one per board, and moves the estimate of a state towards the reward plus the estimate of the next state:

```math
V(S_t)\leftarrow V(S_t)+\alpha\big[R_{t+1}+V(S_{t+1})-V(S_t)\big].
```

**n-step TD** looks $`n`$ steps ahead before using an estimate ($`n=2,4,8,16`$):

```math
G_{t:t+n}=R_{t+1}+\dots+R_{t+n}+V(S_{t+n}),\qquad V(S_t)\leftarrow V(S_t)+\alpha\big[G_{t:t+n}-V(S_t)\big].
```

**Constant-α Monte Carlo** uses the full return of the game, $`G_t=R_{t+1}+R_{t+2}+\dots`$ up to the end of the game, as target: the limit $`n\to\infty`$.

The error is the root mean square error against $`V^*`$, weighted by how often the policy visits each state. On this game the ranking of the textbook random walk is reversed at the start: Monte Carlo is faster than TD(0), because the only reward comes at the end of a 33-move game and TD(0) passes it back one state per visit. With a longer budget an intermediate $`n`$ wins (8-step TD).

### Tabular control

Six control methods of the course, each keeping a table of values and improving the policy as the values change. $`Q(S_t,A_t)`$ is the table's estimate of the value of playing $`A_t`$ on $`S_t`$: the expected return of that move and of the moves after it. All of them choose their moves ε-greedily: the move with the largest $`Q`$, except that with probability $`\epsilon`$ they play a random legal move (see [Exploration](#exploration)).

**Monte Carlo control** (on-policy, first visit, ε-soft): after each game, $`Q(S_t,A_t)`$ moves to the average of the observed returns $`G_t`$. Since no state repeats within a game, first-visit and every-visit coincide.

**Sarsa** (on-policy TD) uses the next move actually played:

```math
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+Q(S_{t+1},A_{t+1})-Q(S_t,A_t)\big].
```

**Q-learning** (off-policy TD) uses the best next move, so it learns the optimal values $`q_*`$ (the value of each move when the best policy plays afterwards) whatever the exploration:

```math
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\big[R_{t+1}+\max_a Q(S_{t+1},a)-Q(S_t,A_t)\big].
```

**Expected Sarsa** uses the average over the next moves of the ε-greedy policy ($`\pi(a\mid S_{t+1})`$ is the probability that the policy plays $`a`$ on $`S_{t+1}`$), a target with less variance than Sarsa's:

```math
Q(S_t,A_t)\leftarrow Q(S_t,A_t)+\alpha\Big[R_{t+1}+\sum_a \pi(a\mid S_{t+1})\,Q(S_{t+1},a)-Q(S_t,A_t)\Big].
```

**n-step Sarsa** ($`n=3`$) is Sarsa with the target $`R_{t+1}+R_{t+2}+R_{t+3}+Q(S_{t+3},A_{t+3})`$.

**Sarsa on afterstates.** Two different moves from two different boards can lead to the same afterstate, and from there the future is the same. So we keep one value per afterstate and write the value of a move as

```math
Q(x,u)=r(x,u)+V\big(M(x,u)\big).
```

This is exact, not an approximation: the true value of a move depends on $`(x,u)`$ only through its points and its afterstate. The agent uses the slide rule $`M`$ but never the probabilities of the new tile. With this $`Q`$ the reward cancels in the Sarsa error, and the update becomes an update of one afterstate value ($`Y_t=M(S_t,A_t)`$, target 0 at game over):

```math
V(Y_t)\leftarrow V(Y_t)+\alpha\big[r(S_{t+1},A_{t+1})+V(Y_{t+1})-V(Y_t)\big].
```

Every update improves all the moves, from all the boards, that lead to the same afterstate, which is why this method learns fastest. The same update with a linear approximation of $`V`$ is the core of the strongest published 2048 players (Szubert & Jaśkowski 2014 and their successors).

### Exploration

- **ε-greedy:** the agent plays a uniformly random legal move with probability $`\epsilon`$, and otherwise the move with the largest $`Q`$ (ties are broken in the order down, left, right, up, so an agent with all-zero values plays the corner rule).
- **Optimistic initial values:** start every $`Q`$ at 1, the largest possible win probability. Every untried move looks promising until it has been tried, so the agent explores every move once and then follows the best. With the sparse win reward this removes the dead start of a learner that has never seen a win.
- **Step-size and exploration schedules:** we compare constant α and ε with values that decrease linearly during training (ε from 0.3 to 0.01, α from 0.2 to 0.02), and try a grid of constant values.

### Exact learning curves

This is how every RL result on the small boards is measured. A learning curve shows how good an agent is as its training goes on. It is usually drawn by stopping every so often, letting the agent play some test games and averaging their results, so the curve is noisy: a lucky batch of test games looks like progress. On the small boards no test games are needed, because the DP evaluates any policy exactly. So at regular checkpoints we take the agent's greedy policy (on every board the move with the largest $`Q`$, without the random moves of exploration) and measure two things exactly.

**How well it plays:** the win probability and the expected score of the greedy policy, computed by the policy evaluation above. These are the numbers in the tables of results.

**How well it knows the values:** on a board $`x`$ the agent's estimate of the value is its best move value, $`\hat V(x)=\max_u Q(x,u)`$ (the hat marks an estimate), and the true value is the $`V^*(x)`$ of the DP ($`V^*_{\text{score}}(x)`$ for an agent trained on the score). The errors $`\hat V(x)-V^*(x)`$ on all the boards are summarised in one number, a weighted root mean square error:

```math
\text{RMSE}=\sqrt{\sum_x d(x)\,\big(\hat V(x)-V^*(x)\big)^2}.
```

Read it from the inside out: square each error, so that errors up and down do not cancel; average the squares over the boards, each board $`x`$ with weight $`d(x)`$; take the square root, to come back to the units of the values. An RMSE of 0.02 on the win probability means estimates off by about 2 percentage points. The weight $`d(x)`$ is the share of the optimal player's decisions taken on the board $`x`$ (computed exactly with the DP), and the weights add up to 1. Most boards are almost never reached by a good player, and an error there costs nothing in play; the weights make the error count where the game is actually played. The sum runs over the boards the agent has met, since it has no estimate for the others.

Neither measurement uses random games, so the curves have no measurement noise: every change in them is a change in what the agent has learned.

### Final test: deep Q-network

Outside the methods of the course, the RL part ends with a deep Q-network built as in the PyTorch tutorial: Q-learning with a neural network $`Q(x,u;\theta)`$, trained on random minibatches of past transitions (experience replay) towards targets computed by a slowly updated copy of the network (target network $`\theta'`$):

```math
\text{loss}=\text{Huber}\Big(r+\gamma\max_{u'}Q(x',u';\theta')-Q(x,u;\theta)\Big).
```

Here $`(x,u,r,x')`$ is a stored transition (board, move, points, next board), $`\gamma<1`$ is a discount that weighs later rewards less, and the Huber loss is the squared error for small errors and the absolute error for large ones, so a few large errors do not dominate the gradient. The input is the one-hot encoding of the exponent of every cell; illegal moves are masked. It is tested on the 2x3 board, where it is evaluated exactly, and on the original 4x4 game. With the tutorial's schedule (one gradient step per move) it is slow: 2,314 points after 10,000 games of the 4x4 game, below the rules of thumb. Playing 1,024 games at once on a GPU, with a replay memory of a million moves (as in the original DQN), it reaches 0.871 of games won on 2x3 and 8,862 points on 4x4 after 100,000 games; with a convolutional network over the pairs of neighbouring cells, 17,485 points and the 2048 tile in 36% of the games. Neither curve is still rising at 100,000 games, and the n-tuple players of the literature, which are given the slide rule through the afterstates, stay far ahead (2048 in over 97% of the games).

### Extra: Sarsa with n-tuple features

Also outside the methods of the course, and not part of the report: we re-implemented the n-tuple players of the literature (Szubert & Jaśkowski 2014), the method behind the strongest published 2048 players, to see how it works. It is Sarsa on afterstates with the table over whole boards replaced by a sum of small tables:

```math
V(y;w)=\sum_{m=1}^{M} w_m\big[\text{index}_m(y)\big].
```

An n-tuple is a fixed group of $`n`$ cells, for example a row. For each of the $`M`$ tuples, $`\text{index}_m(y)`$ reads the exponents found in its cells as a single number, and $`w_m`$ is the tuple's own table of weights, one weight for every possible content of those cells. The value of an afterstate is the sum of the $`M`$ weights its tuples point to. Two boards with the same content in a row share the weight of that row, so what is learned on one board carries over to boards never seen: the generalisation that the tabular methods lack on the 4x4 game. The update is the one of Sarsa on afterstates, applied to the $`M`$ weights that $`Y_t`$ points to, each moved by $`\alpha/M`$ times the error (semi-gradient Sarsa, Sutton & Barto §10.1):

```math
w_m\big[\text{index}_m(Y_t)\big]\leftarrow w_m\big[\text{index}_m(Y_t)\big]+\frac{\alpha}{M}\big[r(S_{t+1},A_{t+1})+V(Y_{t+1};w)-V(Y_t;w)\big],\qquad m=1,\dots,M.
```

We ran it on the 2x3, 2x4 and 3x3 boards, where the DP measures every learned policy exactly, and on the 4x4 game, where it is the extra n-tuple player of the policy viewer. The code is in `src/ntuple_network.py`, the scripts under "Extra" in [Reproducing the results](#reproducing-the-results).

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

**The tabular methods on the 4x4 game.** The six methods run on the original game unchanged, with the settings of the 2x3 runs (score reward, alpha 0.1, MC control with sample averages, epsilon 0.1, 200,000 episodes, 5 seeds). `src/tabular_control_fast.py` holds the same algorithms as `src/tabular_control.py` with the table in a hash table keyed by the 64-bit board; on 2x3, with the same settings, its exact win probabilities agree with those of `src/tabular_control.py` within the spread of the seeds (`results/tabular_4x4/port_check_2x3.csv`). After 200,000 episodes each table holds 23-28 million boards, nearly all met only once, and in a new game 69-74% of the moves fall on boards never met in training (50% for Sarsa on afterstates, which needs one known afterstate out of four). There the table has no value and the corner rule plays: all six methods stay below the corner rule itself (2,564 points), and more training does not raise the score. More exploration makes it worse: with epsilon 0.2 and 0.3 every method scores less and meets fewer of the boards of its own greedy play, because the random moves scatter the training games over boards that the greedy policy never meets. A table cannot generalise from one board to another: this is the curse of dimensionality already visible in the state counts of Part 1, and the reason why the final test needs a network that generalises.

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
python src/tests.py             # about one minute: rules, DP, engines, learning, DQN smoke test
```

### Code

The modules are in `src/`, the scripts that produce the results in `experiments/` (they import the modules from `src/` by themselves).

| File | Content |
|---|---|
| `src/game.py` | Rules of the game on an R x C board (pure Python, the reference implementation), `Env2048` with the Gymnasium `reset`/`step` interface |
| `src/dynamic_programming.py` | Exact DP: reachable states by stage, backward recursion, value iteration (total cost; Jacobi or Gauss-Seidel by stage), exact policy evaluation, occupancy, Monte Carlo simulation |
| `src/game_fast.py` | The same rules on a 64-bit bitboard compiled with numba (about 50x faster); `src/tests.py` checks it against `src/game.py` move by move |
| `src/tabular_prediction.py` | Tabular prediction of a fixed policy: TD(0), n-step TD and constant-alpha Monte Carlo, with the error against the exact values |
| `src/shaped_rewards.py` | The rules-of-thumb ("naive") reward: a bonus for each new largest tile, the growth of the tile sum, the change of free cells; usable as a reward name wherever `'win'`/`'score'` are |
| `src/tabular_control.py` | Tabular control agents on `Env2048`: MC control, Sarsa, Q-learning, Expected Sarsa, n-step Sarsa, Sarsa on afterstates; `ExactEvaluator` computes the exact value of a learned policy with the DP |
| `src/tabular_control_fast.py` | The same six tabular methods on the bitboard engine, with the table in a hash table keyed by the 64-bit board, for the 4x4 game (over a million moves per second) |
| `src/dqn_tutorial.py` | Final test: DQN as in the PyTorch tutorial (replay memory, target network, Huber loss, AdamW), with legal-move masking; needs PyTorch |
| `src/dqn_gpu.py` | The same agent on the GPU with 1,024 games at once (rules in the numba engine), a replay memory of a million moves, the perceptron or a convolutional network (`PairCNN`); needs PyTorch with CUDA |
| `src/ntuple_network.py` | Extra, outside the main line of the project: a re-implementation of the n-tuple players of the literature (semi-gradient Sarsa on afterstates with binary n-tuple features), kept for exploration and used by the 4x4 policy viewer |
| `src/visualize.py` | Animations of the policies playing (`python src/visualize.py -h`) |
| `src/tests.py` | Checks of the rules, the DP, the two engines and the learning methods (`python src/tests.py`) |
| `viewer/` | Policy viewer, online at https://edoardocaproni.github.io/2048/ or opened locally (`viewer/index.html`): the best policy of every method plays. On the 2x3 board every policy is a table over the decision states, shown with the exact optimal win probability of every position and move; on the 4x4 board the page plays the rules of thumb, the six tabular methods (the 50,000 boards each table met most often in training, the corner rule elsewhere), the three DQNs of the final test (the convolutional network with its weights in half precision) and an extra n-tuple player. Data written by `experiments/export_viewer.py`; the page loads the 4x4 agents (8 MB) only when that board is chosen, and the tables (5 MB) only when one of them is |
| `.github/workflows/pages.yml` | Publishes `viewer/` as the GitHub page at every push to `main` that changes it. One-time setup by the owner of the repository: Settings > Pages > Build and deployment > Source: GitHub Actions |

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
python experiments/exp_tabular_4x4.py check # the fast tabular code on 2x3, exact P(win) against tabular_control.py (a minute)
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

Extra (exploration of the n-tuple players of the literature):

```
python experiments/exp_ntuple_small.py      # semi-gradient Sarsa, n-tuple features, on 2x3, 2x4, 3x3 with exact validation
python experiments/plot_ntuple_small.py
python experiments/exp_4x4.py               # 4x4 training grid, 300k games per run (hours; '100000 rcs' for a short run)
python experiments/plot_4x4.py
```
