# Taekwondo Sparring Annotation Guide
## sparring_dynamics project — Version 1.0

---

## Overview

This guide explains how to annotate taekwondo sparring
footage for use in the sparring dynamics simulation pipeline.

Each row in your annotation CSV represents one discrete
exchange — a single identifiable tactical interaction
between the two fighters.

---

## What is an Exchange?

An exchange is a tactical unit of sparring. It begins
when one or both fighters commit to a tactical action
and ends when the action resolves (point scored,
contact missed, referee stops, or fighters reset).

**Examples of single exchanges:**
- Fighter throws a roundhouse kick, opponent blocks → one exchange
- Fighter feints, opponent reacts, fighter scores → one exchange
- Both fighters circle without engaging → one Disengage exchange
- Referee stops action for boundary → one None exchange

**Ambiguous cases:**
- A combination (cut kick + roundhouse) should be coded
  as one exchange if it results in one scoring event
- A multi-kick sequence that results in multiple scores
  should be split into separate exchanges

---

## State Coding Definitions

### Attack
Fighter commits to a scoring technique with intent to score.
- Throwing a roundhouse kick to trunk or head
- Throwing a back kick
- Executing a spinning heel kick
- Throwing a punch to trunk

**Key criterion:** Intent to score. If the kick is clearly
a feint and not committed, code as Feint instead.

### Defend
Fighter's primary action is stopping an incoming technique.
- Blocking with arms or leg shield
- Absorbing a hit in a shell stance (Philly shell, guard)
- Using footwork specifically to avoid an incoming technique
- Turtle position

**Key criterion:** Reactivity. The fighter is responding
to opponent's attack rather than initiating.

### Disengage
Fighter creates distance or resets position without
attacking or defending.
- Circling or lateral movement without commitment
- Backing away to reset after an exchange
- Bouncing/footwork in neutral position
- Out-of-bounds reset

**Key criterion:** Neither attacking nor defending —
purely positional or resetting.

### Feint
Fighter performs deceptive movement to provoke a
reaction without committing to a full scoring technique.
- Pump kick (push kick used as probe/feint)
- Level switch (shift weight head-to-body or body-to-head)
- Shoulder fake
- Step-in without committing
- Half-thrown kick pulled back

**Key criterion:** Deception intent. The movement is
designed to elicit a response, not score directly.

---

## Field Reference

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| match_id | Yes | str | Unique match identifier |
| round | Yes | int | Round number (1-3, 4+ for overtime) |
| timestamp_s | Yes | float | Seconds from round start |
| exchange_id | Yes | int | Sequential exchange number |
| fighter_id | Yes | str | Primary fighter code |
| opponent_id | Yes | str | Opponent fighter code |
| fighter_state | Yes | str | Attack/Defend/Disengage/Feint |
| opponent_state | Yes | str | Attack/Defend/Disengage/Feint |
| winner | Yes | str | F1/F2/Double/None |
| f1_points | Yes | int | Points scored by fighter (0-4) |
| f2_points | Yes | int | Points scored by opponent (0-4) |
| technique | No | str | Scoring technique used |
| body_target | No | str | head or trunk |
| penalty | No | str | Gam-jeom if assessed |
| notes | No | str | Free text annotation |

---

## Point Values (WT Rules)

| Technique | Target | Points |
|-----------|--------|--------|
| Punch | Trunk | 1 |
| Kick | Trunk | 2 |
| Kick | Head | 3 |
| Turning kick | Head | 4 |

---

## Annotation Workflow

1. Open match video in VLC or similar player
2. Set playback speed to 0.5× for detailed exchanges
3. Open your annotation CSV template
4. Watch one exchange, pause
5. Code: What state was each fighter in? Who won? Points?
6. Fill one row in the CSV
7. Resume and repeat

**Recommended session length:** 15-20 minutes of annotation
at a time. Annotating 3 minutes of footage typically
produces 20-40 exchanges.

---

## Common Coding Challenges

**Q: Fighter throws a kick that misses entirely.**
A: Code fighter_state=Attack, winner=None, both points=0.

**Q: I can't tell if it was a feint or a committed attack.**
A: If in doubt, code Attack. Reserve Feint for clear
deceptive intent where the technique is visibly pulled.

**Q: Referee stops for boundary mid-exchange.**
A: Code as None. No points, Disengage for both if
no technique was in progress, Attack/Defend if a
technique was already committed.

**Q: Penalty (gam-jeom) during an exchange.**
A: Fill the penalty field. f1_points and f2_points
should reflect technique points only, not penalty points.

---

## Loading Your Annotations

```python
from sparring_dynamics.data.annotation_format import (
    load_annotation_csv, print_annotation_summary,
    annotations_to_exchanges, annotations_to_sequences
)

annotations, metadata = load_annotation_csv('my_match.csv')
print_annotation_summary(metadata)

# For payoff matrix estimation
exchanges = annotations_to_exchanges(annotations)

# For transition matrix estimation
f1_seqs, f2_seqs = annotations_to_sequences(annotations)
```

---

## Reliability Notes

Inter-rater reliability is a real concern in behavioral
annotation. If multiple people annotate the same footage:
- Agree on state definitions before starting
- Code 10 exchanges together to calibrate
- Expect ~80-85% agreement on state labels as a baseline
- Discuss and resolve systematic disagreements

For a thesis, consider reporting Cohen's kappa on a
subset of exchanges coded by two annotators independently.
