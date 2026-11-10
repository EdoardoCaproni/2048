# Study index: course topics, depth, progress

Private, not part of the submission. One row per topic in the order the
course taught it (lectures page, A.Y. 2024/25). Sources per row:

- **Lecture**: number and date from the course page, with the slide file
  the page links (`private/*.pdf`, password-protected: to be integrated
  once the slides are available; rows marked ⚠ depend on slides only).
- **RLOC** = Bertsekas, *Reinforcement Learning and Optimal Control*;
  **RLI** = Sutton & Barto 2nd ed.; **PSES** = Ross, probability.
- **Form.** = page of `Materiali DPRL/Formulario RL.pdf` (20 pages).
- **Notes** = page of `Materiali DPRL/RL Book Notes.pdf` (39 pages,
  follows RLI chapters 2–7, 9, 10; nothing on chapter 13).
- **Report** = where the 2048 report uses it (section numbers of
  `report/report.tex`), because the oral starts from the project.

## Depth legend

| Depth | Meaning | What to be able to do |
|---|---|---|
| **A** | core, used in the project | write the formula or algorithm on the board from memory, explain every symbol, say what it converges to and under which conditions, connect it to 2048 |
| **B** | taught, not used in the project | state the idea in two sentences, write the update rule / Bellman equation, know when it is preferred over the A methods |
| **C** | taught as example or overview | know it exists, what problem it solves, one sentence on how |

Rule of thumb for the oral (exam page: "discussion of the project work"
+ "theoretical questions on the basic notions of DP and RL"): every A
topic will come up through the project; B topics come up as "why did you
not use X?" or as a stand-alone basic question; C topics only if the
professor digresses.

## Progress

Mark with the date when done. **E** = Edoardo, **T** = Tommaso. "Read"
= read the sources; "Can explain" = explained it aloud without notes
(use `notes/oral_questions.md`).

### Part 1: dynamic programming (RLOC ch. 1 and 4, slides)

| # | Topic | Lecture | Sources | Form. | Report | Depth | E read | E explain | T read | T explain |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | What the course is about: DP vs RL, model known vs unknown, terminology (state, control, disturbance, cost vs reward) | 01 (02/10), 12 (07/11) `RLOC-Preface`, `RLI-Intro`, `RLOC-Terminology` | RLOC preface; RLI ch. 1 | — | §1, §6.1 | B | [ ] | [ ] | [ ] | [ ] |
| 2 | Probability review: random variables, expectation, conditional expectation, law of large numbers, independence | 02 (03/10) `03-Probability-Basics` | PSES ch. 3–4 | — | §5 (MC check) | B | [ ] | [ ] | [ ] | [ ] |
| 3 | Simulation of stochastic systems: sampling from a distribution, Monte Carlo estimate of an expectation, confidence interval $\hat p\pm1.96\sqrt{\hat p(1-\hat p)/n}$ | 03 (09/10) `04-Simulation-Tools`, homework | formulary p. 1 "Why Monte Carlo simulation" | 1 | §5 Monte Carlo validation | A | [ ] | [ ] | [ ] | [ ] |
| 4 | Python, notebooks, numpy | 04 (10/10) | — | — | appendix | C | [ ] | [ ] | [ ] | [ ] |
| 5 | Discrete-time Markov chains: definition, transition matrix $P_{ij}$, state transition graph, homogeneity, $n$-step transitions, expectation of a function of the trajectory | 05 (16/10) `06-Markov-Chains-Basics`, homework | formulary p. 1 | 1 | §5 occupancy (forward pass) | A | [ ] | [ ] | [ ] | [ ] |
| 6 | Dynamic system $x_{k+1}=f(x_k,w_k)$ as a Markov chain; sequential decision problems; MDP: $x_{k+1}=f(x_k,u_k,w_k)$, stage cost $g_k$, terminal cost $g_T$, admissible controls $U(x)$, policies $\mu_k$ | 06 (17/10) `07-MDP-Formulation`; 02 `02-MDP-Intro` | RLOC §1.1–1.2 | 1–2 | §3 formulation | **A** | [ ] | [ ] | [ ] | [ ] |
| 7 | Finite-horizon problem: $J^*=\min_\pi E[\sum_k g_k + g_T]$, tail subproblems, cost-to-go $V_k(s)$, **Bellman equation** $V_k(s)=\min_u E[g_k+V_{k+1}(f)]$, **DP algorithm** (backward recursion), principle of optimality | 07 (23/10) `08-Deterministic-DP`; 08 (24/10) `09-Stochastic-DP` | RLOC §1.1–1.3 | 3 | §3 eq. (1)–(2), Alg. 1 | **A** | [ ] | [ ] | [ ] | [ ] |
| 8 | Expected values in the DP recursion: $E[g(s,u,w)]=\sum_w g\,p(w)$, $E[V_{k+1}(f(s,u,w))]=\sum_w V_{k+1}(f(s,u,w))p(w)$; deterministic DP as the special case | 07, 08 | RLOC §1.1 | 3 | §3 eq. (2) | **A** | [ ] | [ ] | [ ] | [ ] |
| 9 | Worked examples: HEMS (home energy), shortest path, inventory control (IC) with Python | 07 `08b-HEMS`, `08c-Shortest-Path`; 08 `09b-IC`, `09b-Python-IC` ⚠ | RLOC §1.2 examples | — | — | C | [ ] | [ ] | [ ] | [ ] |
| 10 | Optimal stopping: stop/continue, threshold policies; gambling example | 09 (30/10) `10-Optimal-Stopping`, `10b-Gambling` ⚠ | RLOC §1.3 (examples) | — | — | B | [ ] | [ ] | [ ] | [ ] |
| 11 | Infinite horizon: stationarity assumptions; **discounted cost** $V^*(s)=\min_\mu E[\sum_k\alpha^k g]$, Bellman equation, **value iteration** and its error bound $\|V^{(i)}-V^*\|\le\frac{\alpha}{1-\alpha}\delta$ | 10 (31/10) `11-Infinite-DP` | RLOC ch. 4 (§4.1–4.3) | 4 | §3 Cor. 1 (why not needed) | **A** for Bellman/VI, B for the bound | [ ] | [ ] | [ ] | [ ] |
| 12 | Infinite horizon with **total cost**: terminal absorbing state, proper policies ("reach the terminal state with probability $>0$ from every state"), Bellman equation, value iteration from $V^{(0)}=0$ | 10 | RLOC §4.1, 4.4 (stochastic shortest path) | 4–5 | §3 Cor. 1 (all policies proper) | **A** | [ ] | [ ] | [ ] | [ ] |
| 13 | Minimum-time problems (Spider-Fly), component replacement | 11 (06/11) `12-Component-Replacement`, `12b-Spider-Fly` ⚠ | RLOC §4.4 examples | — | — | B | [ ] | [ ] | [ ] | [ ] |
| 14 | DP summary: what DP needs (model, small state space), curse of dimensionality, bridge to RL | 12 (07/11) `13-DP-Summary` | RLOC §1.4; RLI ch. 1 | — | §5.2 Table 1, §8 | A | [ ] | [ ] | [ ] | [ ] |

### Part 2: reinforcement learning (RLI)

| # | Topic | Lecture | Sources | Form. / Notes | Report | Depth | E read | E explain | T read | T explain |
|---|---|---|---|---|---|---|---|---|---|---|
| 15 | $k$-armed bandit: action value $q_*(a)$, sample average, incremental update $Q\leftarrow Q+\frac1n[R-Q]$, $\epsilon$-greedy, constant step size (non-stationary), convergence conditions $\sum\alpha_n=\infty,\sum\alpha_n^2<\infty$, **optimistic initial values** | 13 (13/11) | RLI §2.1–2.6 | F 6 / N 2–4 | §6.2 exploration, §6.3 init | **A** | [ ] | [ ] | [ ] | [ ] |
| 16 | MDP in RL notation: $p(s',r\mid s,a)$, returns $G_t$ (episodic, discounted, unified), policies, $v_\pi$, $q_\pi$, Bellman expectation equations, $v_*$, $q_*$, Bellman optimality equations; relation to RLOC notation (cost = $-$reward, $\alpha=\gamma$) | 14 (14/11) | RLI ch. 3 (§3.1–3.6) | F 7 / N 4–7 | §6.1 | **A** | [ ] | [ ] | [ ] | [ ] |
| 17 | DP in RLI form: iterative policy evaluation, policy improvement theorem, **policy iteration**, **value iteration**, generalised policy iteration (GPI), bootstrapping | 14 | RLI ch. 4 (§4.1–4.7) | F 8–9 / N 7–12 | §5 (exact policy evaluation), §6.2 "GPI" | **A** | [ ] | [ ] | [ ] | [ ] |
| 18 | **Monte Carlo prediction** (first-visit, every-visit) and **MC control**: exploring starts, on-policy $\epsilon$-soft control, why action values are needed without a model | 15 (20/11) | RLI §5.1–5.4 | F 9–10 / N 12–16 | §6.2 MC control | **A** | [ ] | [ ] | [ ] | [ ] |
| 19 | Off-policy MC: target vs behaviour policy, coverage, **importance sampling** ratio $\rho_{t:T-1}$, ordinary vs weighted IS (bias/variance), incremental implementation, off-policy MC prediction and control; gambler's problem example | 16 (21/11) `18-Python-MC` ⚠ | RLI §5.5–5.7; §4.4 Ex. 4.3 | F 10–12 / N 16–19 | not used (say why: long episodes, huge variance of $\rho$) | B | [ ] | [ ] | [ ] | [ ] |
| 20 | **TD(0) prediction**: update $V(S)\leftarrow V(S)+\alpha[R+\gamma V(S')-V(S)]$, TD error, bootstrapping + sampling, advantages over MC, batch/certainty-equivalence optimality | 17 (26/11) | RLI §6.1–6.3 | F 12 / N 20–22 | §6.2 TD(0) on afterstates | **A** | [ ] | [ ] | [ ] | [ ] |
| 21 | **Sarsa** (on-policy), **Q-learning** (off-policy), **Expected Sarsa**: update rules, what each converges to under $\epsilon$-greedy behaviour, variance comparison | 18 (27/11) | RLI §6.4–6.6 | F 13 / N 23–24 | §6.2 Table 3 | **A** | [ ] | [ ] | [ ] | [ ] |
| 22 | Afterstates (value of the position after the agent's deterministic move) | — (not a lecture; in the assigned chapter) | RLI §6.8 | — | §3 remark, §6.2, §6.3 | **A** (we must defend it) | [ ] | [ ] | [ ] | [ ] |
| 23 | TD on the gambler's problem; Gymnasium Frozen Lake (`reset`/`step` interface) | 19 (28/11) `21-Python-TD` ⚠ | RLI Ex. 4.3 | — | §6.1 (`Env2048` has the same interface) | C | [ ] | [ ] | [ ] | [ ] |
| 24 | **$n$-step TD prediction**: $n$-step return $G_{t:t+n}$, update, error-reduction property, random walk example | 20 (04/12) `22-Random-walk-prediction` ⚠ | RLI §7.1 | F 13–14 / N 25–27 | — | B | [ ] | [ ] | [ ] | [ ] |
| 25 | **$n$-step Sarsa**, $n$-step Expected Sarsa, off-policy $n$-step with IS, **tree backup** (off-policy without IS) | 21 (05/12) `23-Random-walk-control` ⚠ | RLI §7.2, 7.3, 7.5 | F 14–17 / N 27–32 | §6.2 3-step Sarsa | A for $n$-step Sarsa, B for the rest | [ ] | [ ] | [ ] | [ ] |
| 26 | Function approximation, prediction: $\hat v(s,\mathbf w)$, objective $\overline{VE}$ with on-policy distribution $\mu$, SGD, gradient MC, **semi-gradient TD(0)** (why "semi"), linear case $\hat v=\mathbf w^\top\mathbf x(s)$, feature construction (§9.5 as background for the $n$-tuple network) | 22 (10/12) | RLI §9.1–9.4, 9.7 (9.5 for features) | F 17–18 / N 32–37 | §6.3 | **A** | [ ] | [ ] | [ ] | [ ] |
| 27 | Function approximation, control: episodic **semi-gradient Sarsa**, semi-gradient $n$-step Sarsa | 22 | RLI §10.1–10.2 | F 19–20 / N 37–39 | mentioned (why afterstate TD instead) | B | [ ] | [ ] | [ ] | [ ] |
| 28 | Policy gradient: parametrised policy $\pi(a\mid s,\theta)$, policy gradient theorem, **REINFORCE** $\theta\leftarrow\theta+\alpha\gamma^tG_t\nabla\ln\pi$, baseline | 23 (11/12) | RLI §13.1–13.3 | — (not in formulary or notes) | not used (why: deterministic move model, long episodes) | B | [ ] | [ ] | [ ] | [ ] |
| 29 | Actor–critic, policy parametrisation for continuous actions | 24 (12/12) | RLI §13.5, 13.7 | — | — | C | [ ] | [ ] | [ ] | [ ] |
| 30 | Overview only: eligibility traces, deep RL | contents page "overview of further topics" | RLI ch. 12 | — | — | C | [ ] | [ ] | [ ] | [ ] |

### Project-specific items (not course topics, but the oral starts here)

| # | Item | Where | Depth | E | T |
|---|---|---|---|---|---|
| P1 | 2048 as an MDP: state, control, disturbance $(i,v)$, afterstate $y=M(x,u)$, objectives (win probability as terminal cost, score) | report §3 | A | [ ] | [ ] |
| P2 | Tile sum is a clock ⇒ finite horizon, stage decomposition, one backward pass, no VI/PI, all policies proper | report §4 Prop. 1, Cor. 1 | A | [ ] | [ ] |
| P3 | Exact results: state counts, optimal vs corner/greedy/random, exceptions table, role of $p_4$ | report §5 | A | [ ] | [ ] |
| P4 | Exact evaluation of a learned policy, weighted RMSE, agreement; why learning curves have no simulation noise | report §6.1 | A | [ ] | [ ] |
| P5 | Which tabular method did what and why (dense vs sparse reward, optimism, afterstates) | report §6.2 | A | [ ] | [ ] |
| P6 | $n$-tuple network = linear FA with one-hot tables; update = semi-gradient TD(0); symmetric sampling; approximation error measured exactly | report §6.3 | A | [ ] | [ ] |
| P7 | 4x4 results and comparison with the literature; what is known about 2048 (open problems, NP-hardness, strong players) | report §6.3, §7 | A | [ ] | [ ] |
| P8 | Code: be able to write a small DP recursion or a Sarsa loop in Python on the spot | `dp2048.py`, `rl2048.py` | A | [ ] | [ ] |

## Suggested order (9 days, 10–18 Nov; oral around 19 Nov)

| Day | Topics (row numbers) | Check |
|---|---|---|
| 1 | 2, 3, 5, 6: probability, simulation, Markov chains, MDP formulation; redo P1 in the course notation | [ ] |
| 2 | 7, 8, 14: finite-horizon Bellman equation and DP algorithm on a toy example by hand; P2, P3 | [ ] |
| 3 | 11, 12, 10, 13: infinite horizon (discounted, total cost, VI), stopping, minimum time; P2 again (why one pass suffices) | [ ] |
| 4 | 15, 16, 17: bandits, RLI notation, policy/value iteration, GPI; P4 | [ ] |
| 5 | 18, 19: Monte Carlo on- and off-policy, importance sampling | [ ] |
| 6 | 20, 21, 22, 23: TD(0), Sarsa, Q-learning, Expected Sarsa, afterstates; P5 | [ ] |
| 7 | 24, 25, 26, 27: $n$-step methods, function approximation; P6 | [ ] |
| 8 | 28, 29, 30: policy gradient (B level), overview topics; P7, P8 | [ ] |
| 9 | Mock oral: `notes/oral_questions.md` questions 1–40, one explains and the other asks | [ ] |

Rows marked ⚠ have no textbook source and rely on the slides: do them
when the slides are available, at depth C unless the slides show the
professor spent a full lecture on them (then B).
