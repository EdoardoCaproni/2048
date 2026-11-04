# Oral preparation: likely questions and short answers

Private notes, not part of the submission. Notation: the professor's DP
half uses Bertsekas (costs, minimisation, $J$ or $V$ for cost-to-go); the
RL half uses Sutton & Barto (rewards, maximisation, $v_\pi$, $q_\pi$).
Be able to switch between the two.

## A. Dynamic programming (Bertsekas, RLOC ch. 1 and 4)

**1. State the finite-horizon stochastic DP problem.**
System $x_{t+1}=f_t(x_t,u_t,w_t)$, $t=0,\dots,T-1$, $w_t$ with known
distribution depending on $(x_t,u_t)$ only; policy $\pi=(\mu_0,\dots,\mu_{T-1})$,
$u_t=\mu_t(x_t)\in U_t(x_t)$; cost $J_\pi(x_0)=E[g_T(x_T)+\sum_t g_t(x_t,\mu_t(x_t),w_t)]$;
find $\pi^*$ minimising it.

**2. Bellman's equation / the DP algorithm.** $J_T(x)=g_T(x)$;
$J_t(x)=\min_{u\in U_t(x)} E_{w}[g_t(x,u,w)+J_{t+1}(f_t(x,u,w))]$, backward
in $t$. $\mu_t^*(x)$ = a minimiser. Principle of optimality: the tail of an
optimal policy is optimal for the tail subproblem.

**3. How does 2048 fit?** State = board (exponents), control = legal move,
disturbance = (cell, value) of the new tile, dynamics = slide/merge then
tile. Cost: terminal 1 if lost, 0 if won, no stage cost, so
$J^*(x)=\min_\pi P^\pi(\text{lose})=1-V^*(x)$. For the score objective: stage
reward $r(x,u)$ (= negative stage cost), no terminal cost.

**4. Why is it finite-horizon if the number of moves is random?** The tile
sum $\sigma$ grows by exactly 2 or 4 per move (Prop. 1), so $\sigma/2$ is a
clock: stage $k$ = states with $\sigma=2k$. Every game ends within
$RC\cdot2^{K-2}$ moves. The backward recursion over $\sigma$ is the DP
algorithm with stage index $\sigma/2$ and transitions that skip at most one
stage.

**5. Why not value iteration or policy iteration?** They are for the
infinite-horizon problems (discounted or total cost, stationary). Our
problem can be seen as total cost with terminal states, where every policy
is proper (terminates), so VI would converge in at most (number of
stages) iterations; but the stage structure gives the exact solution in
one pass, with no iteration. Know VI anyway: $V^{k+1}(s)=\min_u E[g+\alpha V^k(f)]$,
contraction with modulus $\alpha$ in the discounted case.

**6. Proper policy, total cost.** A policy is proper if it reaches the
terminal (absorbing, cost-free) state with probability 1 from every state.
Assumption in the course: at least one proper policy exists and improper
policies have infinite cost. In 2048 every policy is proper (bounded
horizon).

**7. Afterstate Bellman equation.** $V^*(x)=\max_u W^*(M(x,u))$,
$W^*(y)=E_w[V^*(y+w)]$ (and $=1$ if $y$ contains the target). Two half-steps:
controlled deterministic, then uncontrolled random.

**8. Policy evaluation.** Same recursion with $\mu(x)$ in place of the
max. Exact. Used for corner/greedy/random and for the learned policies.

**9. Expected visits (occupancy).** Forward pass: $\mu(x_0)=p_0(x_0)$,
$\mu(x')=\sum_x \mu(x)\,\pi(u|x)\,P(x'|x,u)$. It is the marginal of the Markov
chain induced by the policy, summed over time; it weights "where the
decisions matter".

**10. Monte Carlo check.** $\hat p=\frac1n\sum \mathbb 1_{\text{win}}$,
$SE=\sqrt{\hat p(1-\hat p)/n}$, 95% interval $\hat p\pm1.96\,SE$; 20k games give
$\pm0.0033$.

**11. Curse of dimensionality.** States $\le K^{RC}$; 2x3/64 has 27k, 3x3/64
4.8M, 4x4/2048 $\approx 4.6\cdot10^{16}$. Time of the backward pass is
proportional to $|X|\cdot|U|\cdot$(spawns).

**12. Optimal stopping / minimum time (other DP topics of the course).**
Be able to state them: stopping = at each stage choose stop (terminal
cost) or continue; minimum time = total cost with unit stage cost until
the target set is reached. 2048 with the win objective is a
"maximum probability of reaching a target set" problem, a cousin of
minimum time.

## B. Reinforcement learning (Sutton & Barto)

**13. MDP in RL notation; returns; value functions.** $p(s',r|s,a)$,
$G_t=\sum_k\gamma^k R_{t+k+1}$, $v_\pi(s)=E_\pi[G_t|S_t=s]$,
$q_\pi(s,a)=E_\pi[G_t|S_t=s,A_t=a]$, Bellman equations and optimality
equations (write them). $\gamma=1$ is fine for episodic tasks with bounded
length (ours).

**14. Why $v_*=V^*$ of Part 1 with the win reward?** Reward 1 only at the
win, $\gamma=1$: the return is the win indicator, its expectation the win
probability.

**15. GPI.** Generalised policy iteration: evaluation and improvement
interleaved; all control methods (MC control, Sarsa, Q-learning,...) are
instances.

**16. MC control, on-policy, $\epsilon$-soft.** Generate an episode with
$\pi$ ($\epsilon$-greedy), compute returns backwards, update $Q(S_t,A_t)$
towards the return (first visit), improve $\pi$ to be $\epsilon$-greedy wrt
$Q$. Converges to the best $\epsilon$-soft policy. In 2048 first-visit =
every-visit (no state repeats in an episode).

**17. Off-policy MC and importance sampling.** Target $\pi$, behaviour $b$
with coverage; ratio $\rho_{t:T-1}=\prod \pi(A_k|S_k)/b(A_k|S_k)$; ordinary
(unbiased, high variance) vs weighted (biased, lower variance). Not used
because the ratios of 30-move episodes have enormous variance.

**18. TD(0) prediction.** $V(S_t)\leftarrow V(S_t)+\alpha[R_{t+1}+\gamma V(S_{t+1})-V(S_t)]$.
Bootstraps; biased but lower variance than MC; converges for decreasing
$\alpha$. TD error $\delta_t$.

**19. Sarsa.** $Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma Q(S',A')-Q(S,A)]$ with $A'$
from the behaviour policy: on-policy, converges to $q_{\pi}$ of the
$\epsilon$-greedy policy, and to $q_*$ if $\epsilon\to0$ (GLIE).

**20. Q-learning.** $Q(S,A)\leftarrow Q(S,A)+\alpha[R+\gamma\max_a Q(S',a)-Q(S,A)]$:
off-policy, learns $q_*$ regardless of the behaviour (as long as it keeps
exploring). This is why its values have the smallest error against $V^*$
in our tables, while Sarsa/Expected Sarsa estimate the value of the
exploring policy, which is lower.

**21. Expected Sarsa.** Target $R+\gamma\sum_a\pi(a|S')Q(S',a)$: the expectation
removes the sampling variance of $A'$; same fixed point as Sarsa; with a
greedy target policy it is Q-learning.

**22. n-step Sarsa.** $G_{t:t+n}=R_{t+1}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n Q(S_{t+n},A_{t+n})$;
$n=1$ is Sarsa, $n\to\infty$ is MC; intermediate $n$ often best. Tree
backup: the off-policy n-step method without importance sampling.

**23. Afterstates (6.8).** When the environment has a deterministic part
followed by a random part, learn the value of the position after the
deterministic part. Fewer values to learn (one per afterstate instead of
one per state-action), and different (state, action) pairs leading to the
same afterstate share it. Our TD(0) on afterstates: choose
$u=\arg\max r(x,u)+V(M(x,u))$, update $V(y)$ towards $r'+V(y')$.

**24. Exploration.** $\epsilon$-greedy; optimistic initial values (bandits
§2.6): unvisited actions look better than they are, so the greedy agent
tries them; works once, at the start. We used $Q_0=1$ (win reward) and
$V_0=480$ (score reward). On 2x3 with the win reward the zero-initialised
greedy learner never sees a reward (random play wins 0.02%); on 2x4 and
3x3 the initial corner-like policy wins 36-43%, so no help is needed.

**25. Function approximation, semi-gradient TD(0).**
$\hat v(s,\mathbf w)$; objective $\overline{VE}=\sum_s\mu(s)[v_\pi(s)-\hat v(s,\mathbf w)]^2$;
SGD on the MC target gives gradient MC; replacing the target by the
bootstrapped $R+\gamma\hat v(S',\mathbf w)$ and ignoring its dependence on
$\mathbf w$ gives the semi-gradient method
$\mathbf w\leftarrow\mathbf w+\alpha[R+\gamma\hat v(S',\mathbf w)-\hat v(S,\mathbf w)]\nabla\hat v(S,\mathbf w)$.
Linear case $\hat v=\mathbf w^\top\mathbf x(s)$, $\nabla\hat v=\mathbf x(s)$;
converges to the TD fixed point, within $1/(1-\gamma)$ of the best
approximation error.

**26. n-tuple network as linear FA.** Features = one-hot encoding of the
pattern in each tuple: $\mathbf x(y)$ has exactly $M$ ones (one per tuple),
$\hat V=\mathbf w^\top\mathbf x$; the semi-gradient update adds
$\alpha\,\delta$ to the $M$ active weights (we spread it: $\alpha/M$ each,
like the step-size rule of tile coding, §9.5). Tuple choice = feature
engineering (coarse coding).

**27. Why not semi-gradient Sarsa / Q with FA on the 4x4?** Afterstates
make action values unnecessary: the greedy move only needs $\hat V$ of the
4 afterstates, which are computed with the known slide rule. Also
Q-learning with FA (deadly triad: off-policy + bootstrapping + FA) can
diverge; TD(0) on-policy with linear FA converges.

**28. Policy gradient (course topic, not used).** REINFORCE:
$\theta\leftarrow\theta+\alpha G_t\nabla\ln\pi(A_t|S_t,\theta)$; actor-critic adds a
learned baseline/critic. Why not here: stochastic policy not needed, high
variance over 1000-move episodes.

**29. How was the learned policy evaluated?** Exactly, with the DP of Part
1 (policy evaluation), on the small boards; by Monte Carlo (1000 games,
95% CI) on the 4x4 board. The exact learning curves have no simulation
noise.

**30. Limits/what did not work.** Sparse reward without exploration;
approximation error of small tuples (measured exactly on 2x3); constant
$\alpha$ and $\epsilon$ leave the tabular methods short of the optimum at
200k episodes (would need decaying schedules); the 4x4 optimum is unknown.
