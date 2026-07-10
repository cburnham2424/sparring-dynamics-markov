# Sparring Dynamics Markov

Traditional Taekwondo point sparring is a sequential decision-making process in which fighters continuously adapt their tactics based on previous exchanges and opponent behavior. As a second-degree black belt and multiple-time world champion in point sparring, I have experienced this adaptive process firsthand, where matches are often decided by pattern recognition and a single well-timed tactical adjustment.
This project translates those adaptive dynamics into a data-driven mathematical framework. Using Markov chains, evolutionary game theory, and adaptive state-transition dynamics, it models a sparring match as a stochastic system in which fighters evolve their tactical decisions over repeated exchanges. Rather than relying on fixed assumptions, the framework is designed to learn fighter-specific transition probabilities and scoring effectiveness from annotated sparring footage, allowing simulations to become increasingly representative of real competitive behavior as additional match data is incorporated.
The model combines learned transition matrices, payoff estimation, adaptive memory, Monte Carlo simulation, and statistical validation to study how strategies emerge, adapt, and succeed against different opponent styles. While developed using Taekwondo as an observable competitive system, the underlying mathematics is applicable to a broad class of adaptive stochastic systems. The same concepts of state transitions, evolutionary selection, and adaptive dynamics appear in fields ranging from behavioral modeling to mathematical oncology, where similar frameworks are used to study tumor evolution, treatment resistance, and population dynamics. This repository represents both an applied sports analytics project and the foundation for future research in adaptive mathematical modeling.

## Data Status

Two matches annotated (23 exchanges total):
- MATCH_20260613_M1: 14 exchanges, 6-3 final score
- MATCH_20260613_M2: 9 exchanges, 7-2 final score

Empirically grounded cells (>= 5 observations):
- Transition matrix: Attack and Defend rows for
  both fighters (data-driven)
- Payoff matrix: (Attack, Defend) and (Defend, Attack)
  cells (data-driven)
- Remaining 14/16 payoff cells and Disengage/Feint
  transition rows: hand-crafted defaults pending
  additional match annotation

Next milestone: 3 additional matches targeting
Feint and Disengage state observations.
