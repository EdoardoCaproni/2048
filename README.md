# 2048 as a Markov decision problem

Project work for *Dynamic Programming and Reinforcement Learning* (University of Siena, A.Y. 2024/25).

- **Part 1, exact DP**: small versions of 2048 (up to 3x3 cells) are solved exactly by a backward recursion over the tile sum, which is a clock of the game. The optimal policy is compared with the usual heuristics, and the role of chance is measured.
- **Part 2, RL**: the model is forgotten and the game is learned from a simulator. Tabular Monte Carlo and TD control on the 2x3 board are validated against the exact values; an n-tuple network trained by TD(0) on afterstates plays the original 4x4 game.
- **Report**: `report/report.tex` (problem formulation, DP, RL, and what is known about the game).

## Setup

Python 3.11 or later. From this folder:

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
python test_2048.py             # about one minute: rules, DP, engines, learning
```

## Code

| File | Content |
|---|---|
| `game2048.py` | Rules of the game on an R x C board (pure Python, the reference implementation), `Env2048` with the Gymnasium `reset`/`step` interface |
| `dp2048.py` | Exact DP: reachable states by stage, backward recursion, exact policy evaluation, occupancy, Monte Carlo simulation |
| `fast2048.py` | The same rules on a 64-bit bitboard compiled with numba (about 50x faster), used for learning on big boards; `test_2048.py` checks it against `game2048.py` move by move |
| `rl2048.py` | Tabular agents on `Env2048`: MC control, Sarsa, Q-learning, Expected Sarsa, n-step Sarsa, TD(0) on afterstates; `ExactEvaluator` computes the exact value of a learned policy with the DP |
| `ntuple2048.py` | n-tuple network (linear function approximation) and TD(0) on afterstates, training loop, exact comparison with the DP on small boards |
| `viz2048.py` | Animations of the policies playing (`python viz2048.py -h`) |
| `2048_DP.ipynb` | Notebook of Part 1 |

## Reproducing the results

All the scripts write to `results/`; run them from this folder with the venv active.

```
python experiments/dp_figures.py            # Part 1 figures and tables (add 'big' for the 3x3 board, 5 min)
python experiments/bench_engine.py          # throughput of the two engines
python experiments/exp_tabular_2x3.py       # tabular methods, 90 runs x 200k episodes (about 1 h on 26 cores)
python experiments/plot_tabular.py
python experiments/exp_tabular_grid.py      # alpha x epsilon grid and decaying schedules (130 runs)
python experiments/plot_schedule.py
python experiments/exp_ntuple_small.py      # n-tuple TD(0) on 2x3, 2x4, 3x3 with exact validation
python experiments/plot_ntuple_small.py
python experiments/exp_4x4.py               # 4x4 training grid, 300k games per run (hours; '100000 rcs' for a short run)
python experiments/exp_4x4.py 1000000 rect 0.1 2,3   # the two 1M-game runs of the best configuration
python experiments/eval_4x4.py              # Monte Carlo evaluation of the checkpoints and of the heuristics
python experiments/plot_4x4.py
```

The report is compiled with `pdflatex` and `bibtex` in `report/` after copying the figures from `results/` (`python experiments/collect_figures.py`).
