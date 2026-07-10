"""
CSV loading and parsing for annotation data.

Supports two CSV formats:
1. Sequence format: fighter, sequence
   (a comma-joined list of states per row, for estimating transition
   matrices from observed state sequences)

2. Exchange format: f1_state, f2_state, winner, f1_points, f2_points
   (for estimating payoff matrices from annotated exchanges)

These are two different files/use cases — a single annotated match
CSV in exchange format does not on its own provide sequence data,
since it records win/loss/points, not fighter-by-fighter state order.
"""
import csv
import os

from sparring_dynamics.config import STATE_INDEX, STATES

VALID_WINNERS = {'F1', 'F2', 'Double', 'None'}
VALID_FIGHTERS = {'F1', 'F2'}


def load_exchange_csv(filepath):
    """
    Load annotated exchange data from CSV.

    Expected columns:
    f1_state, f2_state, winner, f1_points, f2_points

    Returns list of dicts. Raises ValueError on any validation failure
    with a message indicating the row number and specific problem.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Annotation file not found: {filepath}\n"
            f"Run pipeline.py --template to create a template."
        )

    exchanges = []
    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        required = {'f1_state', 'f2_state', 'winner',
                    'f1_points', 'f2_points'}
        if not required.issubset(set(reader.fieldnames or [])):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(
                f"CSV missing required columns: {missing}\n"
                f"Required: {required}"
            )

        for row_num, row in enumerate(reader, start=2):
            # Skip comment rows
            if row['f1_state'].startswith('#'):
                continue

            # Skip fully blank rows (e.g. from an un-annotated template)
            if not row['f1_state'].strip():
                continue

            # Validate states
            for field in ('f1_state', 'f2_state'):
                if row[field] not in STATE_INDEX:
                    raise ValueError(
                        f"Row {row_num}: Invalid {field} "
                        f"'{row[field]}'. Must be one of {STATES}"
                    )

            # Validate winner
            if row['winner'] not in VALID_WINNERS:
                raise ValueError(
                    f"Row {row_num}: Invalid winner '{row['winner']}'. "
                    f"Must be one of {VALID_WINNERS}"
                )

            # Validate points
            try:
                f1_pts = int(row['f1_points'])
                f2_pts = int(row['f2_points'])
            except ValueError:
                raise ValueError(
                    f"Row {row_num}: Points must be integers, "
                    f"got f1_points='{row['f1_points']}', "
                    f"f2_points='{row['f2_points']}'"
                )

            if f1_pts < 0 or f2_pts < 0:
                raise ValueError(
                    f"Row {row_num}: Points must be non-negative."
                )

            exchanges.append({
                'f1_state':  row['f1_state'],
                'f2_state':  row['f2_state'],
                'winner':    row['winner'],
                'f1_points': f1_pts,
                'f2_points': f2_pts
            })

    if len(exchanges) == 0:
        raise ValueError(
            f"No valid exchanges found in {filepath}. "
            f"Check that rows are not all commented out."
        )

    return exchanges


def load_sequence_csv(filepath):
    """
    Load state sequence data from CSV for transition estimation.

    Expected columns: fighter, sequence
    Where sequence is a comma-separated list of states, quoted so it
    survives as a single CSV field:

    fighter,sequence
    F1,"Attack,Feint,Attack,Disengage,Feint"
    F2,"Defend,Attack,Defend,Feint,Defend"

    Returns:
    f1_sequences: list of lists of state strings
    f2_sequences: list of lists of state strings
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Sequence file not found: {filepath}"
        )

    f1_sequences = []
    f2_sequences = []

    with open(filepath, newline='') as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):
            if row.get('fighter', '').startswith('#'):
                continue

            fighter = row.get('fighter', '').strip()
            if fighter not in VALID_FIGHTERS:
                raise ValueError(
                    f"Row {row_num}: Invalid fighter '{fighter}'. "
                    f"Must be F1 or F2."
                )

            raw_seq = row.get('sequence', '').strip()
            sequence = [s.strip() for s in raw_seq.split(',')]

            for i, state in enumerate(sequence):
                if state not in STATE_INDEX:
                    raise ValueError(
                        f"Row {row_num}, position {i}: "
                        f"Invalid state '{state}'. Must be one of {STATES}"
                    )

            if fighter == 'F1':
                f1_sequences.append(sequence)
            else:
                f2_sequences.append(sequence)

    return f1_sequences, f2_sequences


def create_annotation_template(filepath, n_rows=20):
    """
    Generate a blank annotation CSV template ready for manual coding
    of real match footage.

    The real CSV header is the file's first line (required for
    csv.DictReader to parse it, and for load_exchange_csv to work on
    it once filled in) — instructions, one worked example, and blank
    placeholder rows all follow as '#'-commented lines that
    load_exchange_csv already knows to skip.
    """
    lines = [
        "f1_state,f2_state,winner,f1_points,f2_points",
        "# SPARRING ANNOTATION TEMPLATE — Instructions:",
        "# f1_state / f2_state: Attack, Defend, Disengage, or Feint",
        "# winner: F1, F2, Double (both score), or None (no score)",
        "# f1_points / f2_points: points scored (WT rules: head=4, body=3, punch=1)",
        "# Double: both fighters score on same exchange — fill both points",
        "# None: no points — set both points to 0",
        "# Delete the '#' from a row below (or add new ones) to record real data.",
        "Attack,Defend,F1,2,0",
    ]
    for i in range(2, n_rows + 1):
        lines.append(f"# row {i}: add your annotated exchange here")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f"Template created: {filepath} ({n_rows} blank rows)")


def generate_placeholder_csv(filepath, n_exchanges=60):
    """
    Generate a realistic placeholder exchange CSV for testing the
    pipeline before real annotated match data exists, consistent with
    the fighting styles modeled elsewhere in this project (CJ:
    aggressive feint-heavy blitzer; Fighter 2: patient counter-fighter).
    Deterministic, not randomly sampled, so distributional targets are
    hit exactly/closely.
    """
    rows = [
        # Feint vs Defend: 10 exchanges, F1 wins 7, avg f1_points ~2.5
        ('Feint', 'Defend', 'F1', 3, 0),
        ('Feint', 'Defend', 'F1', 2, 0),
        ('Feint', 'Defend', 'F1', 3, 0),
        ('Feint', 'Defend', 'F1', 2, 0),
        ('Feint', 'Defend', 'F1', 3, 0),
        ('Feint', 'Defend', 'F1', 3, 0),
        ('Feint', 'Defend', 'F1', 2, 0),
        ('Feint', 'Defend', 'F2', 0, 1),
        ('Feint', 'Defend', 'F2', 0, 2),
        ('Feint', 'Defend', 'None', 0, 0),

        # Attack vs Defend: 10 exchanges, F1 wins 6, avg f1_points ~1.8
        ('Attack', 'Defend', 'F1', 2, 0),
        ('Attack', 'Defend', 'F1', 1, 0),
        ('Attack', 'Defend', 'F1', 2, 0),
        ('Attack', 'Defend', 'F1', 2, 0),
        ('Attack', 'Defend', 'F1', 1, 0),
        ('Attack', 'Defend', 'F1', 3, 0),
        ('Attack', 'Defend', 'F2', 0, 1),
        ('Attack', 'Defend', 'F2', 0, 1),
        ('Attack', 'Defend', 'F2', 0, 2),
        ('Attack', 'Defend', 'Double', 1, 1),

        # Defend vs Attack: 8 exchanges, F1 wins 5 (counter), avg f1_points ~1.0
        ('Defend', 'Attack', 'F1', 1, 0),
        ('Defend', 'Attack', 'F1', 1, 0),
        ('Defend', 'Attack', 'F1', 1, 0),
        ('Defend', 'Attack', 'F1', 1, 0),
        ('Defend', 'Attack', 'F1', 1, 0),
        ('Defend', 'Attack', 'F2', 0, 2),
        ('Defend', 'Attack', 'F2', 0, 1),
        ('Defend', 'Attack', 'None', 0, 0),

        # Feint vs Attack: 7 exchanges, split 4-3 F2, avg points ~1.5 each
        ('Feint', 'Attack', 'F2', 0, 2),
        ('Feint', 'Attack', 'F2', 0, 1),
        ('Feint', 'Attack', 'F2', 0, 2),
        ('Feint', 'Attack', 'F2', 0, 1),
        ('Feint', 'Attack', 'F1', 2, 0),
        ('Feint', 'Attack', 'F1', 1, 0),
        ('Feint', 'Attack', 'F1', 2, 0),

        # Attack vs Feint: 6 exchanges, F1 wins 4, avg f1_points = 2.0
        ('Attack', 'Feint', 'F1', 2, 0),
        ('Attack', 'Feint', 'F1', 2, 0),
        ('Attack', 'Feint', 'F1', 2, 0),
        ('Attack', 'Feint', 'F1', 2, 0),
        ('Attack', 'Feint', 'F2', 0, 1),
        ('Attack', 'Feint', 'Double', 1, 1),

        # Disengage vs Attack: 6 exchanges, F2 wins 4, avg f2_points ~1.5
        ('Disengage', 'Attack', 'F2', 0, 2),
        ('Disengage', 'Attack', 'F2', 0, 1),
        ('Disengage', 'Attack', 'F2', 0, 2),
        ('Disengage', 'Attack', 'F2', 0, 1),
        ('Disengage', 'Attack', 'F1', 1, 0),
        ('Disengage', 'Attack', 'None', 0, 0),

        # Attack vs Attack: 5 exchanges, mixed Double/F1/F2
        ('Attack', 'Attack', 'Double', 1, 1),
        ('Attack', 'Attack', 'Double', 2, 2),
        ('Attack', 'Attack', 'F1', 2, 0),
        ('Attack', 'Attack', 'F1', 1, 0),
        ('Attack', 'Attack', 'F2', 0, 2),

        # Disengage vs Disengage: 4 exchanges, mostly None
        ('Disengage', 'Disengage', 'None', 0, 0),
        ('Disengage', 'Disengage', 'None', 0, 0),
        ('Disengage', 'Disengage', 'None', 0, 0),
        ('Disengage', 'Disengage', 'Double', 1, 1),

        # Remaining 4: other state pairs with realistic outcomes
        ('Feint', 'Feint', 'Double', 1, 1),
        ('Disengage', 'Defend', 'F1', 1, 0),
        ('Defend', 'Defend', 'None', 0, 0),
        ('Defend', 'Feint', 'F1', 1, 0),
    ]

    assert len(rows) == n_exchanges, (
        f"Placeholder row count {len(rows)} does not match n_exchanges={n_exchanges}")

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['f1_state', 'f2_state', 'winner', 'f1_points', 'f2_points'])
        writer.writerows(rows)

    print(f"Placeholder exchange CSV created: {filepath} ({n_exchanges} rows)")
