# Study index: course topics, depth, progress

Private, not part of the submission. One row per topic in the order the
course taught it (lectures page, A.Y. 2024/25), checked against the
professor's own slides (downloaded in `Materiali DPRL/slides/`, text
extracts in `slides/txt/`, notebooks unzipped in `slides/nb/`; the folder
is git-ignored, each of us keeps a copy).

Sources per row:

- **Lecture**: number and date from the course page.
- **Slides**: the professor's deck (`slides/<file>.pdf`, page numbers
  `p.`). The DP half (lectures 02–12) has the professor's own decks; the
  RL half (lectures 13–24) was taught from **Sutton's slides** (Google
  shared drive, not downloaded) and the book, with Python notebooks.
- **RLOC** = Bertsekas, *Reinforcement Learning and Optimal Control*;
  **RLI** = Sutton & Barto 2nd ed.; **PSES** = Ross, probability.
- **Form.** = page of `Materiali DPRL/Formulario RL.pdf` (20 handwritten
  pages); **Notes** = page of `Materiali DPRL/RL Book Notes.pdf` (39
  pages, follows RLI ch. 2–7, 9, 10; nothing on ch. 13).
- **Report** = where the 2048 report uses it (section of
  `report/report.tex`), because the oral starts from the project.

## Depth legend

| Depth | Meaning | What to be able to do |
|---|---|---|
| **A** | core, used in the project | write the formula or algorithm on the board from memory in the **professor's notation** (see the cheat-sheet at the end), explain every symbol, say what it converges to and under which assumptions, connect it to 2048 |
| **B** | taught, not used in the project, or a worked example of the course | state the idea in two sentences, write the update rule / Bellman equation, formulate the example as an MDP (state, input, uncertainty, dynamics, costs) |
| **C** | mentioned, overview | know it exists, what problem it solves, one sentence on how |

Rule of thumb for the oral (exam page: "discussion of the project work"
+ "theoretical questions on the basic notions of DP and RL"): every A
topic will come up through the project; B topics come up as "why did you
not use X?", as "formulate this toy problem" (the professor's examples
are the obvious candidates), or as a stand-alone basic question; C topics
only if the professor digresses.

## Progress

Mark with the date when done. **E** = Edoardo, **T** = Tommaso. "Read"
= read the slides and the book pages; "Explain" = explained it aloud
without notes (use `notes/oral_questions.md`).

### Part 1: dynamic programming (professor's slides, RLOC ch. 1 and 4)

| # | Topic (what the slides actually contain) | Lecture | Slides | Book | Form. | Report | Depth | E read | E explain | T read | T explain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | What the course is about: sequential decision problems, MDP = Markov chain + input + costs + policy; DP exact, RL approximate; **terminology table** RL ↔ DP (environment = system, agent = controller, reward = −cost, value = −cost function, Q-factor, planning = DP with known model, learning = DP without explicit model, GPI = optimistic policy iteration, **afterstate = post-decision state**) | 01 (02/10), 12 (07/11) | `02-MDP-Intro` p. 2–6, 23–25; `RLOC-Terminology` (RLOC §1.4, pp. 43–45); `RLOC-Preface`; `RLI-Intro` (= RLI ch. 1) | RLOC §1.4; RLI ch. 1 | — | §1, §6.1 | B | [ ] | [ ] | [ ] | [ ] |
| 2 | Probability review: classical and frequentist definitions, Kolmogorov axioms A1–A3 and consequences, conditional probability, independence, **Bayes rule**, **total probability theorem**, random variables, CDF, PMF (discrete) and PDF (continuous), mean, variance, expected value operator $E[g(x)]=\sum g(x_i)f_x(x_i)$ | 02 (03/10) | `03-Probability-Basics` (15 handwritten pages) | PSES ch. 3–4 | — | §3 eq. (2), §5 | B | [ ] | [ ] | [ ] | [ ] |
| 3 | Simulation tools: estimating pmf/pdf/cdf from samples (histogram, ecdf), simulating $x_{t+1}=f_t(x_t,u_t,w_t)$, **inversion sampling** $\bar x=F_x^{-1}(\bar u)$ and its discrete version (`np.random.choice`), **law of large numbers** (proof via Chebyshev), **Monte Carlo estimate** of a probability $\hat p_A=N_A/N$ and of an expectation $\hat\mu_g=\frac1N\sum g(x^{(k)})$: unbiased, error std $\propto1/\sqrt N$ | 03 (09/10) | `04-Simulation-Tools` p. 9–15; homework `04b-Homework` (birthday, $\pi$, Buffon, IC cost distribution, wind energy P90) and solution notebook | — | 1 | §5 Monte Carlo validation | **A** | [ ] | [ ] | [ ] | [ ] |
| 4 | Python, notebooks, numpy (functions listed on the slides: `np.random.choice`, `np.histogram`, `np.cumsum`, `plt.hist`...) | 04 (10/10) | `02-MDP-Intro` p. 22; `04-Simulation-Tools` p. 16 | — | — | appendix | C | [ ] | [ ] | [ ] | [ ] |
| 5 | Discrete-time Markov chains: Markov property, **transition probability matrix** $P_{ij}(t)$ (stochastic matrix, product of stochastic matrices), state transition graph, homogeneous chains, **dynamic-system representation** $x_{t+1}=f(x_t,w_t)$ with $x_0,w_0,w_1,\dots$ independent, three equivalent representations; **distribution propagation** $\pi_{t+1}=\pi_tP$, $\pi_t=\pi_0P^t$, $N$-step probabilities; **steady state** $\pi_{ss}=\pi_{ss}P$ for finite irreducible aperiodic chains; Monte Carlo simulation of a chain ($n^T$ terms otherwise); inventory control as a chain | 05 (16/10) | `06-Markov-Chains-Basics` p. 5–23; homework `06b-Homework` (S/I/R epidemic on a grid) | RLOC §1.1 | 1 | §5 occupancy = distribution propagation stage by stage | **A** | [ ] | [ ] | [ ] | [ ] |
| 6 | **MDP formulation**: Markov chain + input $u_t\in U_t(x_t)$ + stage cost $g_t(x_t,u_t,w_t)$ (or $g_t(x_t,u_t,x_{t+1})$) + terminal cost $g_T(x_T)$ (used to enforce constraints, $=\infty$); the two specifications (dynamic system $f_t$ with $\Pr(w_t)$, or transition matrices $P^k_{ij}(t)=\Pr(x_{t+1}=j\mid x_t=i,u_t=k)$); policy $\mu=(\mu_0,\dots,\mu_{T-1})$, $u_t=\mu_t(x_t)$; closed-loop system = chain + costs; special cases (time-invariant, deterministic cost, stationary policy); the problem $\min_\mu E[\sum_t g_t+g_T]$; **open-loop vs closed-loop**; $m^{nT}$ possible policies | 06 (17/10) | `07-MDP-Formulation` p. 2–12 | RLOC §1.1–1.2 | 2 | §3 formulation (state, control, disturbance $(i,v)$, dynamics, objective) | **A** | [ ] | [ ] | [ ] | [ ] |
| 7 | **Deterministic DP**: principle of optimality (Bellman quote, Milan–Rome–Naples), tail subproblems, cost-to-go $V_t(s)$, $V_T=g_T$, $V_0=J^*$, Bellman's backward recursion $V_t(s)=\min_{u}[g_t(s,u)+V_{t+1}(f_t(s,u))]$, **DP algorithm** (initialisation, loop $t=T-1,\dots,0$, for all $s$: $\mu^*_t(s)=\arg\min$, $V_t(s)$), cost $nmT$ operations vs $m^{nT}$ policies | 07 (23/10) | `08-Deterministic-DP` p. 3–9; `08c-Shortest-Path` (hand-made example with the $U$ and $V$ matrices) | RLOC §1.1 | 3 | §3 Alg. 1 | **A** | [ ] | [ ] | [ ] | [ ] |
| 8 | **Stochastic DP**: same with expectations, $V_t(s)=\min_u E[g_t(s,u,w_t)+V_{t+1}(f_t(s,u,w_t))]$, DP algorithm with $\arg\min E[\cdot]$, cost $nmpT$; **how to compute the expected values** in both specifications ($\sum_w(\cdot)\Pr(w_t)$, or $\sum_{x_{t+1}}(\cdot)P^{u}_{s x_{t+1}}$), special cases; definition of the **greedy policy** ($\arg\min$ of the expected immediate cost only, neglects the future) | 08 (24/10) | `09-Stochastic-DP` p. 2–12 | RLOC §1.2–1.3 | 3 | §3 eq. (1)–(2); our "greedy heuristic" is exactly the slides' greedy policy | **A** | [ ] | [ ] | [ ] | [ ] |
| 9 | Worked finite-horizon examples, formulated step by step (horizon, state, input and $U(x_t)$, uncertainty, dynamics, stage cost, terminal cost, DP, comparison with heuristics by Monte Carlo): **HEMS** dishwasher (state = next cycle, $g_T=\infty$ if not finished; homework versions 2–3 redefine the state), **inventory control** ((s,S) optimal policy, two heuristics, cost distributions, Python notebook `DP_IC.ipynb`) | 07, 08 | `08b-HEMS`; `09b-IC`; `nb/09b-Python-IC` | RLOC §1.2 | — | §5 benchmark policies (same structure: optimal vs heuristics, MC) | B (practice formulating) | [ ] | [ ] | [ ] | [ ] |
| 10 | **Optimal stopping**: "purchasing with a deadline" (state with several components $(i,j,k)$, chain given by transition matrices, "already purchased" flag, $g_T=\infty$ if not purchased, 4-D arrays for $V$ and $\mu^*$, **time-varying threshold** policy; homework: constant threshold by MC); **gambling** (max probability that $x_T\ge6$ → stage reward 0, **terminal reward 1 on the event**, DP with max; homework: heuristic vs optimal by MC, "max expected final capital") | 09 (30/10) | `10-Optimal-Stopping`; `10b-Gambling`; `nb/10-Python-Optimal-Stopping` | RLOC §1.3 (examples); Bertsekas DPOC vol. I ch. 4 | — | **§3 objective**: our win probability is formulated exactly like the gambling slide (reward 1 at the end on the event) | B; the gambling formulation **A** | [ ] | [ ] | [ ] | [ ] |
| 11 | **Infinite horizon, discounted cost**: why (stationary optimal policies, cost independent of $T$); stationarity assumptions (homogeneous chain, stationary cost, no terminal cost, i.i.d. $w_t$, stationary policy); $J=E[\sum_t\alpha^tg]$, $0<\alpha<1$, assumption A1 bounded cost; one value function $V^*(s)$; **Bellman's equation** $V^*(s)=\min_{u\in U(s)}E[g(s,u,w)+\alpha V^*(f(s,u,w))]$ has a unique solution, $\mu^*$ from the argmin; **value iteration** from $V^{(0)}=0$, convergence from any $V^{(0)}$, exponential rate, stop when $\|V^{(i+1)}-V^{(i)}\|_\infty<\delta$, **error bound** $\|V^{(i+1)}-V^*\|_\infty\le\frac{\alpha}{1-\alpha}\delta$ ($\delta=\frac{1-\alpha}{\alpha}\varepsilon$) | 10 (31/10) | `11-Infinite-DP` p. 2–11, 17 | RLOC §4.1–4.3 | 3–4 | §3 Cor. 1 (why we need no iteration) | **A** (the bound: B) | [ ] | [ ] | [ ] | [ ] |
| 12 | **Infinite horizon, total cost**: $J=E[\sum_tg]$ with assumptions **A2** (a termination state $S$, absorbing and cost-free) and **A3** (positive probability of reaching $S$ under every policy from every $x_0$); Bellman's equation without $\alpha$; value iteration, converges from any $V^{(0)}$ but **no error bound**; "α = 1? try yourself"; applications: optimal stopping, minimum time | 10 | `11-Infinite-DP` p. 12–17 | RLOC §4.1, 4.4 (stochastic shortest path) | 4–5 | §3 Cor. 1: 2048 satisfies A2–A3 (game over / win are absorbing, every policy reaches them in bounded time), and the clock makes the problem finite-horizon on top | **A** | [ ] | [ ] | [ ] | [ ] |
| 13 | Infinite-horizon examples: **component replacement** (wear level, replace or keep, discounted VI with $\alpha=0.97$, implementation with growing matrices $U_{aux},V_{aux}$, optimal threshold "replace above 3" vs greedy (threshold 7) vs "replace at once", compared by MC; homework: both chain specifications, vary $\alpha$); **spider and fly** (minimum time: state = (fly, spider), stage cost 1 until caught, termination states absorbing + cost-free, total-cost VI) | 11 (06/11) | `12-Component-Replacement`; `12b-Spider-Fly` | RLOC §4.4 examples; Bertsekas DPOC vol. I ex. 7.2.2, vol. II ex. 1.2.1 | — | — | B (practice formulating) | [ ] | [ ] | [ ] | [ ] |
| 14 | **DP summary and limitations**: the list of problems that are MDPs (min cost, discounted, optimal stopping, minimum time, **min/max probability of an event**, constraints via costs); the two DP algorithms side by side; towards RL: continuous spaces → discretisation → **curse of dimensionality** (states grow exponentially with the dimension); **unknown model** (no $f$ or $\Pr(x_{t+1}\mid x_t,u_t)$, only a simulator) → approximate DP: rollout, neuro-dynamic programming, reinforcement learning | 12 (07/11) | `13-DP-Summary` p. 3–17 | RLOC §1.4, ch. 2 (overview) | — | §5.2 Table 1, §6.1, §8 | **A** | [ ] | [ ] | [ ] | [ ] |

### Part 2: reinforcement learning (Sutton's slides + RLI + notebooks)

| # | Topic | Lecture | Material | Book | Form. / Notes | Report | Depth | E read | E explain | T read | T explain |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 | $k$-armed bandit: action value $q_*(a)$, sample average, incremental update $Q\leftarrow Q+\frac1n[R-Q]$, $\epsilon$-greedy, constant step size for non-stationary problems (exponential recency weighting), convergence conditions $\sum\alpha_n=\infty,\sum\alpha_n^2<\infty$, **optimistic initial values** | 13 (13/11) | Sutton's slides ch. 2 | RLI §2.1–2.6 | F 6 / N 2–4 | §6.2 exploration, §6.3 initial values | **A** | [ ] | [ ] | [ ] | [ ] |
| 16 | MDP in RL notation: $p(s',r\mid s,a)$, $p(s'\mid s,a)$, $r(s,a)$, returns $G_t$ (episodic, discounted, unified), policies $\pi(a\mid s)$, $v_\pi$, $q_\pi$, Bellman expectation equations, $v_*$, $q_*$, Bellman optimality equations; **dictionary to the professor's DP notation** (cost = −reward, $\gamma=\alpha$, $V_t\leftrightarrow v_*$, $\mu\leftrightarrow\pi$, Q-factor) | 14 (14/11) | Sutton's slides ch. 3; `RLOC-Terminology` | RLI §3.1–3.6 | F 7 / N 4–7 | §6.1 | **A** | [ ] | [ ] | [ ] | [ ] |
| 17 | DP in RLI form: iterative policy evaluation, policy improvement theorem, **policy iteration**, **value iteration**, generalised policy iteration (GPI), bootstrapping; the **gambler's problem** (RLI Ex. 4.3) solved by VI = the professor's running RL example (`gambler_utils.py` in the notebooks) | 14, 16, 19 | Sutton's slides ch. 4; `nb/18-Python-MC`, `nb/21-Python-TD/gambler` | RLI §4.1–4.7, Ex. 4.3 | F 8–9 / N 7–12 | §5 exact policy evaluation; §6.2 GPI | **A** | [ ] | [ ] | [ ] | [ ] |
| 18 | **Monte Carlo prediction** (first-visit, every-visit) of $V$ and $Q$, exploring starts, on-policy **$\epsilon$-soft MC control** (GPI episode by episode); notebook `Gambler-MC-on-policy`: MC estimates of $V$ and $Q$ for min/average/max/optimal betting policies compared with the DP values, then $\epsilon$-greedy MC control | 15 (20/11) | Sutton's slides ch. 5; `nb/18-Python-MC/Gambler-MC-on-policy.ipynb` | RLI §5.1–5.4 | F 9–10 / N 12–16 | §6.2 MC control | **A** | [ ] | [ ] | [ ] | [ ] |
| 19 | **Off-policy MC**: target vs behaviour policy, coverage, **importance sampling** ratio $\rho_{t:T-1}$, ordinary (unbiased, high variance) vs weighted (biased, lower variance), incremental implementation, off-policy MC prediction and control; notebook `Gambler-MC-off-policy-plot`: behaviour mostly random ($\epsilon=0.9$), $10^6$–$10^7$ episodes, very slow, high variance, central states never learned because the ratio is 0 most of the time | 16 (21/11) | Sutton's slides ch. 5; `nb/18-Python-MC/Gambler-MC-off-policy-plot.ipynb` | RLI §5.5–5.7 | F 10–12 / N 16–19 | not used: say why (long episodes, ratios with huge variance; the notebook showed it) | B | [ ] | [ ] | [ ] | [ ] |
| 20 | **TD(0) prediction**: $V(S)\leftarrow V(S)+\alpha[R+\gamma V(S')-V(S)]$, TD error, bootstrapping + sampling, MC error as a sum of TD errors, advantages (online, incremental), convergence, **batch updating** (TD = certainty-equivalence estimate, MC = minimum squared error); homework of the TD notebook: batch MC vs batch TD on the gambler, compared with the DP value | 17 (26/11) | Sutton's slides ch. 6; `nb/21-Python-TD/gambler/Gambler-TD.ipynb` (homework part) | RLI §6.1–6.3 | F 12 / N 20–22 | §6.2 TD(0) on afterstates | **A** | [ ] | [ ] | [ ] | [ ] |
| 21 | **Sarsa** (on-policy), **Q-learning** (off-policy), **Expected Sarsa**: update rules, what each converges to under $\epsilon$-greedy behaviour, variance comparison; notebook `Gambler-TD`: the three methods with $\epsilon$-greedy, $V$ recovered from $Q$ and compared with the DP value of the greedy policy; homework: vary $\alpha$, $\epsilon$ (also decreasing), number of episodes | 18 (27/11) | Sutton's slides ch. 6; `nb/21-Python-TD/gambler/Gambler-TD.ipynb` | RLI §6.4–6.6 | F 13 / N 23–24 | §6.2 Table 3 | **A** | [ ] | [ ] | [ ] | [ ] |
| 22 | **Afterstates** (value of the position after the agent's deterministic move; "post-decision state" in Bertsekas' terminology table) | — (in the assigned chapter, not a lecture) | `RLOC-Terminology` (y) | RLI §6.8 | — | §3 remark, §6.2, §6.3 | **A** (we must defend it) | [ ] | [ ] | [ ] | [ ] |
| 23 | **Gymnasium**: `reset`, `step`, `render`, `close`; Frozen Lake (deterministic and slippery), heuristic policy simulated for the success probability; homework: Sarsa, Q-learning, Expected Sarsa on Frozen Lake and Cliff Walking | 19 (28/11) | `nb/21-Python-TD/frozen-lake/frozen-lake.ipynb` | — | — | §6.1 (`Env2048` has the same interface) | C | [ ] | [ ] | [ ] | [ ] |
| 24 | **$n$-step TD prediction**: $n$-step return $G_{t:t+n}$, the algorithm with $\tau=t-n+1$ (RLI p. 144), error-reduction property; notebook `random-walk-prediction`: 19-state random walk, RMS error vs $n$ and $\alpha$ (RLI Fig. 7.2), true values from Bellman's linear system (homework) | 20 (04/12) | Sutton's slides ch. 7; `nb/22-Random-walk-prediction` | RLI §7.1 | F 13–14 / N 25–27 | — | B | [ ] | [ ] | [ ] | [ ] |
| 25 | **$n$-step Sarsa** (RLI p. 147), $n$-step Expected Sarsa, off-policy $n$-step with importance sampling, **tree backup** (off-policy without IS); notebook `random-walk-control`: $n$-step Sarsa, percentage of optimal actions vs $n$ and $\alpha$, then $n=6,\alpha=0.2$; homework: $n$-step Expected Sarsa and tree backup | 21 (05/12) | Sutton's slides ch. 7; `nb/random-walk-control` | RLI §7.2, 7.3, 7.5 | F 14–17 / N 27–32 | §6.2 3-step Sarsa | A for $n$-step Sarsa, B for the rest | [ ] | [ ] | [ ] | [ ] |
| 26 | Function approximation, prediction: $\hat v(s,\mathbf w)$, objective $\overline{VE}$ with the on-policy distribution $\mu(s)$, SGD, gradient MC, **semi-gradient TD(0)** (why "semi"), linear case $\hat v=\mathbf w^\top\mathbf x(s)$, feature construction (§9.5, background for the $n$-tuple network), neural networks (§9.7, overview) | 22 (10/12) | Sutton's slides ch. 9 | RLI §9.1–9.4, 9.7 (9.5 for features) | F 17–18 / N 32–37 | §6.3 | **A** | [ ] | [ ] | [ ] | [ ] |
| 27 | Function approximation, control: episodic **semi-gradient Sarsa**, semi-gradient $n$-step Sarsa | 22 | Sutton's slides ch. 10 | RLI §10.1–10.2 | F 19–20 / N 37–39 | mentioned (why afterstate TD instead) | B | [ ] | [ ] | [ ] | [ ] |
| 28 | Policy gradient: parametrised policy $\pi(a\mid s,\boldsymbol\theta)$, policy gradient theorem, **REINFORCE** $\boldsymbol\theta\leftarrow\boldsymbol\theta+\alpha\gamma^tG_t\nabla\ln\pi$, baseline | 23 (11/12) | Sutton's slides ch. 13 | RLI §13.1–13.3 | — (not in the formulary or notes) | not used (why: deterministic move model, long episodes) | B | [ ] | [ ] | [ ] | [ ] |
| 29 | Actor–critic, policy parametrisation for continuous actions | 24 (12/12) | Sutton's slides ch. 13 | RLI §13.5, 13.7 | — | — | C | [ ] | [ ] | [ ] | [ ] |
| 30 | Overview only: eligibility traces, deep RL, rollout and neuro-dynamic programming (named in the DP summary) | contents page; `13-DP-Summary` p. 17 | — | RLI ch. 12; RLOC ch. 2 | — | — | C | [ ] | [ ] | [ ] | [ ] |

### Project-specific items (not course topics, but the oral starts here)

| # | Item | Where | Depth | E | T |
|---|---|---|---|---|---|
| P1 | 2048 as an MDP in the professor's template: horizon, state $x_t$, input $u_t\in U(x_t)$, uncertainty $w_t=(i_t,v_t)$ with its distribution, dynamics $f$, stage cost, terminal cost; both objectives (win probability = "max probability of an event" as in the gambling slide; score = reward per stage) | report §3; `10b-Gambling` | A | [ ] | [ ] |
| P2 | Tile sum is a clock ⇒ finite horizon, stage decomposition, one backward pass; why no value iteration (compare with `11-Infinite-DP`: A2–A3 hold, but the clock makes it finite-horizon) | report §4 Prop. 1, Cor. 1 | A | [ ] | [ ] |
| P3 | Exact results: state counts, optimal vs corner/greedy/random (the "greedy policy" of `09-Stochastic-DP` p. 8), exceptions table, role of $p_4$ | report §5 | A | [ ] | [ ] |
| P4 | Exact evaluation of a learned policy, weighted RMSE, agreement; why the learning curves have no simulation noise (the notebooks compare learned $V$ with the DP value of the gambler in the same way) | report §6.1 | A | [ ] | [ ] |
| P5 | Which tabular method did what and why (dense vs sparse reward, optimism, afterstates) | report §6.2 | A | [ ] | [ ] |
| P6 | $n$-tuple network = linear FA with one-hot tables; update = semi-gradient TD(0); symmetric sampling; approximation error measured exactly | report §6.3 | A | [ ] | [ ] |
| P7 | 4x4 results and the literature; what is known about 2048 (open problems, NP-hardness, strong players) | report §6.3, §7 | A | [ ] | [ ] |
| P8 | Code on the spot: the DP loop of `09b-Python-IC` style (init $V_T$, loop $t$, loop $s$, loop $u$, expectation over $w$), a Sarsa loop like `Gambler-TD`, a Monte Carlo estimate with its confidence interval | `dp2048.py`, `rl2048.py`, `nb/` | A | [ ] | [ ] |

## The professor's notation (use it at the board)

| Object | Professor (DP half, slides) | RLI (RL half) | Our report |
|---|---|---|---|
| time, horizon | $t=0,\dots,T$ | $t$, $T$ | stage $k$ = tile sum $/2$ |
| state, input, uncertainty | $x_t\in X_t$, $u_t\in U_t(x_t)$, $w_t\in W_t$ | $S_t$, $A_t$, (inside $p$) | $x$, $u$, $w=(i,v)$ |
| dynamics | $x_{t+1}=f_t(x_t,u_t,w_t)$ or $P^k_{ij}(t)$ | $p(s',r\mid s,a)$ | $x'=M(x,u)+$ tile |
| costs | stage $g_t(x_t,u_t,w_t)$, terminal $g_T(x_T)$ | reward $R_{t+1}$ | $r$, terminal 1 on win |
| policy | $\mu=(\mu_0,\dots,\mu_{T-1})$, $u_t=\mu_t(x_t)$ | $\pi(a\mid s)$ | $\pi(x)$ |
| objective | $J^*(x_0)=\min_\mu E[\sum_tg_t+g_T]$ | $\max E[G_t]$ | $\max P(\text{win})$, $\max E[\text{score}]$ |
| value function | $V_t(s)$ cost-to-go, $V_T=g_T$, $V_0=J^*$; infinite horizon $V^*(s)$ | $v_\pi$, $v_*$, $q_\pi$, $q_*$ | $V_k(x)$, afterstate $W(y)$ |
| Bellman | $V_t(s)=\min_{u}E[g_t+V_{t+1}(f_t)]$ | $v_*(s)=\max_a\sum p[r+\gamma v_*(s')]$ | eq. (1)–(2) |
| discount | $\alpha\in(0,1)$ | $\gamma$ | none ($\gamma=1$, finite horizon) |
| assumptions | A1 bounded cost; A2 termination state absorbing and cost-free; A3 reached with positive probability under every policy | episodic task; proper policies | Cor. 1 |
| algorithms | DP algorithm (backward), value iteration $V^{(i)}$ | policy evaluation, PI, VI, GPI | one backward pass |
| greedy policy | $\arg\min_uE[g_t(s,u,w)]$ (immediate cost only) | greedy w.r.t. $Q$ | "greedy heuristic" |

## Suggested order (9 days, 10–18 Nov; oral around 19 Nov)

| Day | Topics (row numbers) | Check |
|---|---|---|
| 1 | 2, 3, 5: probability, simulation tools, Markov chains (slides 03, 04, 06); redo the inventory-control chain by hand | [ ] |
| 2 | 6, 7, 8: MDP formulation, deterministic and stochastic DP (slides 07, 08, 08c, 09); P1, P2; solve the shortest path by hand as in `08c` | [ ] |
| 3 | 9, 10: HEMS, inventory control, purchasing with a deadline, gambling (slides 08b, 09b, 10, 10b); formulate each one aloud; P3 | [ ] |
| 4 | 11, 12, 13, 14: infinite horizon, VI, A1–A3, component replacement, spider–fly, DP summary (slides 11, 12, 12b, 13); P2 again | [ ] |
| 5 | 1, 15, 16, 17: terminology table, bandits, RLI notation, PI/VI/GPI, gambler's problem; P4 | [ ] |
| 6 | 18, 19: Monte Carlo on- and off-policy, importance sampling (run the two gambler notebooks) | [ ] |
| 7 | 20, 21, 22, 23: TD(0), Sarsa, Q-learning, Expected Sarsa, afterstates, Gymnasium (run `Gambler-TD`); P5 | [ ] |
| 8 | 24, 25, 26, 27: $n$-step methods (run the random-walk notebooks), function approximation; P6, P7 | [ ] |
| 9 | 28, 29, 30 at B/C level; P8; mock oral with `notes/oral_questions.md`, one asks and the other answers | [ ] |

Still missing from the materials: Sutton's slides for lectures 13–24
(Google drive linked from the teaching-material page). They follow the
book chapter by chapter, so the book sections above are the reference;
download them if the drive is accessible to check which sections the
professor skipped.
