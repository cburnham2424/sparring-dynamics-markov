# Sparring Dynamics Markov

Traditional Taekwondo point sparring is a sequential decision-making process in which fighters continuously adapt their tactics based on previous exchanges and opponent behavior. As a second-degree black belt and multiple-time world champion in point sparring, I have experienced this adaptive process firsthand, where matches are often decided by pattern recognition and a single well-timed tactical adjustment.
This project translates those adaptive dynamics into a data-driven mathematical framework. Using Markov chains, evolutionary game theory, and adaptive state-transition dynamics, it models a sparring match as a stochastic system in which fighters evolve their tactical decisions over repeated exchanges. Rather than relying on fixed assumptions, the framework is designed to learn fighter-specific transition probabilities and scoring effectiveness from annotated sparring footage, allowing simulations to become increasingly representative of real competitive behavior as additional match data is incorporated.
The model combines learned transition matrices, payoff estimation, adaptive memory, Monte Carlo simulation, and statistical validation to study how strategies emerge, adapt, and succeed against different opponent styles. While developed using Taekwondo as an observable competitive system, the underlying mathematics is applicable to a broad class of adaptive stochastic systems. The same concepts of state transitions, evolutionary selection, and adaptive dynamics appear in fields ranging from behavioral modeling to mathematical oncology, where similar frameworks are used to study tumor evolution, treatment resistance, and population dynamics. This repository represents both an applied sports analytics project and the foundation for future research in adaptive mathematical modeling.

## Data Status — Current

Five matches annotated, 64 exchanges total:
- MATCH_20260613_M1: 14 exchanges, 6-3 (post-break)
- MATCH_20260613_M2: 9 exchanges, 7-2 (post-break)
- MATCH_20230700_M1: 8 exchanges, 6-3 (pre-break)
- MATCH_20230700_M2: 20 exchanges, 4-3 (pre-break,
  same opponent as M1-2026 confirmed by athlete)
- MATCH_20230700_M3: 13 exchanges, 5-2 (pre-break,
  different opponent)

Empirically grounded transition rows (F1):
- Attack: 25 observations — data-driven
- Defend: 12 observations — data-driven
- Disengage: 16 observations — data-driven
- Feint: 6 observations — directional signal only,
  3 of 6 are consolidated multi-event windows
  due to frame rate limitations, treat with caution

Empirically grounded payoff cells (3 of 16):
- (Attack, Defend): 29 observations
- (Defend, Attack): 12 observations
- (Disengage, Disengage): 15 observations
- Remaining 13 cells: hand-crafted defaults

Key longitudinal finding:
Attack→Disengage transition probability differs
by 0.43 between pre-break (0.60) and post-break
(0.17) footage — CJ reset distance far more
frequently after attacking in 2023 peak training
than in 2026 three months post-break return.

Methodological limitations documented:
- Frame rate too coarse for individual Feint
  resolution in match 5 (3s/frame vs 2-2.5s)
- Round field used as match boundary marker
  in combined dataset
- Opponent identity in cross-year comparison
  confirmed by athlete not by footage
- Sample size insufficient for statistical
  inference — dataset is a research prototype

Next data collection targets:
- High frame rate footage of Feint-heavy exchanges
- Additional opponents to fill 13 sparse payoff cells
- Matches covering Feint vs Attack and Feint vs
  Disengage state combinations currently unobserved
