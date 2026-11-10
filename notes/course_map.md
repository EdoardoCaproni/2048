# Course map: what the project uses versus what the course taught

Private study notes, not part of the submission. See `study_index.md`
for the full topic list with depth and progress; the professor's slides
are in `Materiali DPRL/slides/` (git-ignored). Lecture numbers and
dates are from the course page (`lectures.html`, A.Y. 2024/25); chapters
are RLOC = Bertsekas, *Reinforcement Learning and Optimal Control* (DP
half, cost-minimisation notation) and RLI = Sutton & Barto,
*Reinforcement Learning: An Introduction*, 2nd ed. (RL half). "Formulary"
refers to `Materiali DPRL/Formulario RL.pdf`.

Status legend: **TAUGHT** = a lecture covered it; **BOOK** = in the
textbook sections that were assigned, but not an explicit lecture topic;
**EXT** = an extension or a technique from the 2048 literature, not in
the course. At the oral, present EXT items as "we used X, which is Y of
the course plus Z"; do not present them as course material.

## Part 1: dynamic programming

| Concept in the report | Where | Lecture | Reference | Status | What to say at the oral |
|---|---|---|---|---|---|
| Formulating the game as an MDP: state, control, disturbance, dynamics $x_{t+1}=f(x_t,u_t,w_t)$, cost | Sec. 3 | 06 (17/10) Formulating MDPs; 02 (03/10) | RLOC ch. 1; formulary "Markov decision process" | TAUGHT | Our $f$ = slide/merge then random tile; $w_t=(i_t,v_t)$ independent of the past given the afterstate. |
| Objective as a cost: terminal cost 1 if lost, 0 if won, no stage cost; $J^* = 1 - V^*$ | Sec. 3 | 08 (24/10) Stochastic DP | RLOC ch. 1 | TAUGHT | The course minimises expected cost; we maximise $P(\text{win})$; the two are the same problem. Be ready to rewrite the Bellman equation as a min. |
| Bellman equation and backward recursion (finite horizon) | Sec. 3, Alg. 1 | 07 (23/10) Deterministic DP; 08 (24/10) Stochastic DP | RLOC ch. 1; formulary "Bellman's equation", "Dynamic programming algorithm" | TAUGHT | Stage index = tile sum / 2 (Prop. 1); the recursion runs over stages like $V_t(s)=\min_u E[g + V_{t+1}(f)]$. |
| Why no value iteration / policy iteration is needed; all policies proper; unique solution | Cor. 1 | 10 (31/10) Infinite horizon DP; 11 (06/11) minimum time | RLOC ch. 4 (total cost, proper policies) | TAUGHT (the concepts) + EXT (the clock argument) | The game is a total-cost problem with a terminal state reached in bounded time: the stage decomposition makes it a finite-horizon problem, so one backward pass is exact. Know what value iteration would do here (converge in at most max-stages iterations). |
| Expected-value computation over the disturbance $\sum_i \frac{1}{|E(y)|}[(1-p_4)V(y+e_i)+p_4 V(y+2e_i)]$ | Eq. (2) | 02 (03/10) probability review; 03 (09/10) simulation | RLOC ch. 1; PSES ch. 3-4 | TAUGHT | |
| Afterstates (value of the board after the deterministic part of the move) | Remark, Sec. 6 | not a lecture topic | RLI §6.8 | BOOK | "Afterstate value function" of Sutton & Barto §6.8: the move factors into a controlled deterministic part and an uncontrolled random part. The Bellman equation on afterstates is the same equation, written in two steps. |
| Policy evaluation by the same recursion (exact value of the heuristics and of learned policies) | Sec. 5, Sec. 6 | 14 (14/11) MDPs and DP revisited; 08 | RLI ch. 4 (iterative policy evaluation); RLOC ch. 1 | TAUGHT | Here it is not iterative: one backward pass, because of the stage structure. |
| Monte Carlo simulation to validate the exact value (confidence interval) | Sec. 5 | 03 (09/10) Tools for simulating stochastic systems | formulary "Why Monte Carlo simulation" | TAUGHT | $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$. |
| Occupancy / expected visits (forward pass) to weight the decisions | Sec. 5 | 05 (16/10) Markov chains | formulary "Markov chains" | TAUGHT (Markov chain forward computation) | It is the distribution of the Markov chain induced by the optimal policy, propagated stage by stage. |
| Curse of dimensionality, state counts | Sec. 5 | 01, 12 (07/11) Towards RL | RLOC ch. 1, RLI ch. 1 | TAUGHT | This is the motivation of the RL half. |
| Symmetries as a test | Remark | -- | -- | EXT | |

## Part 2: reinforcement learning

| Concept in the report | Where | Lecture | Reference | Status | What to say at the oral |
|---|---|---|---|---|---|
| Simulator interface (`reset`/`step`), Gymnasium | Sec. 6.1 | 19 (28/11) Gymnasium Frozen Lake | -- | TAUGHT | |
| Returns, $\gamma=1$ allowed because episodes are bounded | Sec. 6.1 | 14 (14/11) | RLI ch. 3 | TAUGHT | Episodic task, finite horizon (Cor. 1). |
| Monte Carlo control, on-policy, first visit, $\epsilon$-soft | Sec. 6.2 | 15 (20/11) MC methods I | RLI §5.4; formulary "Monte Carlo E-soft" | TAUGHT | First-visit = every-visit here because no state repeats in an episode (tile sum clock). |
| Off-policy MC, importance sampling | not used | 16 (21/11) MC methods II | RLI §5.5-5.7 | TAUGHT, not used | Be ready to say why not: episodes are long (30 moves) and the behaviour/target policies differ in many decisions, so the importance ratios have huge variance. |
| Sarsa | Sec. 6.2 | 18 (27/11) TD control | RLI §6.4 | TAUGHT | On-policy: learns $q_{\pi_\epsilon}$, the value of the exploring policy. |
| Q-learning | Sec. 6.2 | 18 (27/11) | RLI §6.5 | TAUGHT | Off-policy: learns $q_*$ while behaving $\epsilon$-greedily; this is why its RMSE against $V^*$ is the smallest. |
| Expected Sarsa | Sec. 6.2 | 18 (27/11) | RLI §6.6 | TAUGHT | Replaces the sampled $Q(s',a')$ by its expectation under the policy: lower variance than Sarsa, same fixed point. |
| $n$-step Sarsa | Sec. 6.2 | 20-21 (04-05/12) n-step bootstrapping | RLI §7.2; formulary "n-step Sarsa" | TAUGHT | Intermediate between MC ($n=\infty$) and TD(0) ($n=1$). |
| Tree backup, off-policy n-step | not used | 21 (05/12) | RLI §7.3, 7.5 | TAUGHT, not used | |
| TD(0) on afterstates (tabular) | Sec. 6.2 | 17 (26/11) TD prediction (TD(0)); afterstates: -- | RLI §6.1-6.3 for TD(0); §6.8 for afterstates | TAUGHT (TD(0)) + BOOK (afterstates) | It is TD(0) prediction applied to the afterstate chain, with the greedy/$\epsilon$-greedy policy improving as the values change (generalised policy iteration, RLI §4.6). |
| $\epsilon$-greedy exploration, optimistic initial values | Sec. 6.2, 6.3 | 13 (13/11) Intuitive RL (bandits) | RLI §2.2, §2.6; formulary "K-armed bandit" | TAUGHT | Optimistic initialisation = the bandit trick of §2.6: unvisited actions look good, so they get tried. Needed because the win reward is sparse. |
| Dense (score) reward as a surrogate of the sparse (win) reward | Sec. 6.1 | -- | -- | EXT (justified by our DP result: 97.9% agreement) | Reward design is a modelling choice; our DP measures exactly what it costs (0.9387 vs 0.9393). |
| Exact evaluation of a learned policy (exact learning curves) | Sec. 6.1 | -- | policy evaluation, RLOC ch. 1 | EXT (use of a taught tool) | Nobody does this on real problems; we can because the small boards are solved. |
| Linear function approximation, semi-gradient TD(0) | Sec. 6.3 | 22 (10/12) Approximate methods | RLI §9.1-9.4 (and §9.7); formulary "On-policy prediction with approximation" | TAUGHT | $\hat v(s,\mathbf w)=\mathbf w^\top \mathbf x(s)$; update $\mathbf w \leftarrow \mathbf w + \alpha[R+\gamma \hat v(S')-\hat v(S)]\nabla \hat v(S)$; "semi-gradient" because the target also depends on $\mathbf w$ but is not differentiated. |
| Semi-gradient Sarsa (control with approximation) | mentioned | 22 (10/12) | RLI §10.1-10.2 | TAUGHT, not used | We do control on afterstates instead: the greedy move needs only $\hat V$ of the afterstates, not $\hat q(s,a)$. |
| $n$-tuple network = binary features from lookup tables (like tile coding with one tiling per tuple) | Sec. 6.3 | -- | RLI §9.5 (coarse coding, tile coding) is the closest | EXT (Szubert & Jaśkowski 2014) | Say: "a linear approximator with binary features, one active entry per table; the update is exactly semi-gradient TD(0)." The tuples are the feature construction, which the course leaves to the designer (§9.5). |
| Symmetric sampling (shared weights over the 8 symmetries) | Sec. 6.3 | -- | -- | EXT | A way of building prior knowledge (invariance of $V^*$) into the features. |
| Step size $\alpha/M$ spread over the active features | Sec. 6.3 | 22 | RLI §9.5 (step size with tile coding: $\alpha = 1/(\tau \cdot \text{tilings})$ rule) | BOOK | Same reasoning as the tile-coding step size. |
| Greedy policy without $\epsilon$ on 4x4 (exploration from the random tile only) | Sec. 6.3 | -- | -- | EXT (Szubert & Jaśkowski) | The spawn randomness already makes the trajectories diverse; on the small boards we show when it is not enough (tabular case). |
| Bitboard engine, numba | Appendix | 04 (10/10) Python, numpy | -- | EXT (engineering) | Only speed; correctness is checked against the pure-Python rules. |
| Policy gradient / REINFORCE / actor-critic | not used | 23-24 (11-12/12) | RLI ch. 13 | TAUGHT, not used | Possible question: "why not policy gradient?" Answer: with a known deterministic move model, afterstate values give the greedy policy directly; policy gradient would need a parametrised stochastic policy and has high variance on 1000-move episodes. |
| Eligibility traces, deep RL (DQN) | not used | overview only | RLI ch. 12, 16 | not taught | Mention only if asked; TD($\lambda$) is the natural extension of our TD(0). |

## Things the professor will likely probe

- Write the Bellman equation of the game in the course's cost notation (min over $u$ of expected cost-to-go), and explain the terminal costs.
- Why is the problem finite-horizon although the number of moves is random? (The tile sum is a clock; stage = tile sum; horizon bounded by $RC\,2^{K-2}$ moves.)
- Difference between the value of a state and of an afterstate; write both Bellman equations.
- On-policy vs off-policy, with Sarsa vs Q-learning as the examples; what each converges to under $\epsilon$-greedy behaviour.
- Why Expected Sarsa has lower variance than Sarsa.
- What "semi-gradient" means and why the full gradient is not used.
- Why the score reward was used for learning and how we measured the cost of that choice.
- How the Monte Carlo confidence intervals were computed and why the exact evaluation is better.
- The role of exploration: why the zero-initialised greedy learner fails with the win reward on 2x3 but not on 2x4/3x3 (the corner-like initial policy already wins 36-43% there).
