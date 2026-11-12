# 2048 as a Markov decision problem

Project work for *Dynamic Programming and Reinforcement Learning* (University of Siena, A.Y. 2024/25).
Repository: https://github.com/EdoardoCaproni/2048

- **Part 1, exact DP**: small versions of 2048 (up to 3x3 cells) are solved exactly. The game is a total-cost problem with termination states: value iteration reaches the exact solution in a finite number of iterations, and ordering the states by the tile sum, which is a clock of the game, turns it into a single backward recursion. The optimal policy is compared with the usual heuristics, and the role of chance is measured.
- **Part 2, RL** (only methods of the course): the model is forgotten and the game is learned from a simulator. The reward is chosen first: a reward built on the players' rules of thumb (free cells, tile sum, a bonus for each new largest tile) is compared exactly with the score and the win, and the score is kept. On the 2x3 board TD(0), n-step TD and Monte Carlo prediction and six tabular control methods (MC control, Sarsa, Q-learning, Expected Sarsa, n-step Sarsa, Sarsa on afterstates) are measured against the exact values. A deep Q-network built as in the PyTorch DQN tutorial is the final test on the original 4x4 game; what others achieved there with learned n-tuple evaluations is reported from the literature.
- **Report**: `report/report.tex` (problem formulation, DP, RL, and what is known about the game).

## Setup

Python 3.10 or later. From this folder:

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
python test_2048.py             # about one minute: rules, DP, engines, learning, DQN smoke test
```

## Code

| File | Content |
|---|---|
| `game2048.py` | Rules of the game on an R x C board (pure Python, the reference implementation), `Env2048` with the Gymnasium `reset`/`step` interface |
| `dp2048.py` | Exact DP: reachable states by stage, backward recursion, value iteration (total cost; Jacobi or Gauss-Seidel by stage), exact policy evaluation, occupancy, Monte Carlo simulation |
| `fast2048.py` | The same rules on a 64-bit bitboard compiled with numba (about 50x faster), used for learning on big boards; `test_2048.py` checks it against `game2048.py` move by move |
| `predict2048.py` | Tabular prediction of a fixed policy: TD(0), n-step TD and constant-alpha Monte Carlo, with the error against the exact values |
| `shaping2048.py` | The rules-of-thumb ("naive") reward: a bonus for each new largest tile, the growth of the tile sum, the change of free cells; usable as a reward name wherever `'win'`/`'score'` are |
| `rl2048.py` | Tabular control agents on `Env2048`: MC control, Sarsa, Q-learning, Expected Sarsa, n-step Sarsa, Sarsa on afterstates (q(x,u) = r(x,u) + V(M(x,u))); `ExactEvaluator` computes the exact value of a learned policy with the DP |
| `ntuple2048.py` | Not part of the report: a re-implementation of the n-tuple players of the literature (semi-gradient Sarsa on afterstates with binary n-tuple features), kept for exploration and used by the 4x4 policy viewer |
| `dqn2048.py` | Final test: DQN as in the PyTorch tutorial (replay memory, target network, Huber loss, AdamW), with legal-move masking; needs PyTorch |
| `viz2048.py` | Animations of the policies playing (`python viz2048.py -h`) |
| `viewer/` | Policy viewer: open `viewer/index.html` in a browser and watch the best policy of every method play, on the 2x3 board with the exact optimal win probability of every position and move (data written by `experiments/export_viewer.py`) |
| `2048_DP.ipynb` | Notebook of Part 1 |

## Reproducing the results

All the scripts write to `results/`; run them from this folder with the venv active.

```
python experiments/dp_figures.py            # Part 1 figures and tables (add 'big' for the 3x3 board, 5 min)
python experiments/dp_value_iteration.py    # value iteration against the backward recursion (2 min)
python experiments/bench_engine.py          # throughput of the two engines
python experiments/exp_prediction.py        # TD(0), n-step TD and MC prediction, 50 runs (under a minute)
python experiments/exp_tabular_2x3.py       # tabular methods, 90 runs x 200k episodes (about 1 h on 26 cores)
python experiments/plot_tabular.py
python experiments/exp_reward.py dp         # optimal policy of each reward, exact (a few minutes)
python experiments/exp_reward.py tabular    # Sarsa on afterstates with the naive reward, 5 seeds (15 min)
python experiments/plot_reward.py
python experiments/exp_tabular_grid.py      # alpha x epsilon grid and decaying schedules (130 runs)
python experiments/plot_schedule.py
python experiments/eval_4x4.py              # the rules of thumb on the 4x4 board (and any n-tuple checkpoints present)
python experiments/exp_dqn.py               # DQN final test: 3 seeds on 2x3 and on 4x4, in parallel (about 1 h)
python experiments/plot_dqn.py
python experiments/export_viewer.py         # retrains the best policy of every method and writes viewer/data_*.js (a few minutes)
```

Not part of the report (exploration of the n-tuple players of the literature):

```
python experiments/exp_ntuple_small.py      # semi-gradient Sarsa, n-tuple features, on 2x3, 2x4, 3x3 with exact validation
python experiments/plot_ntuple_small.py
python experiments/exp_4x4.py               # 4x4 training grid, 300k games per run (hours; '100000 rcs' for a short run)
python experiments/plot_4x4.py
```

The report is compiled with `pdflatex` and `bibtex` in `report/` after copying the figures from `results/` (`python experiments/collect_figures.py`).
