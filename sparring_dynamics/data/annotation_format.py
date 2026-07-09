"""
Reusable annotation format for taekwondo sparring match footage:
schema definition, blank CSV template generation, validation, loading,
and conversion to the exchange/sequence formats consumed by
estimate_payoffs_v2.py and estimate_transitions.py.
"""
import csv
import os

import pandas as pd


# ---------------------------------------------------------------------------
# SECTION 1 — Schema definition
# ---------------------------------------------------------------------------

ANNOTATION_SCHEMA = {
    'match_id': {
        'type': str,
        'required': True,
        'description': (
            'Unique identifier for the match. '
            'Format: MATCH_{date}_{location}_{f1_id}_vs_{f2_id} '
            'e.g. MATCH_20260711_Denver_CJ_vs_KIM. '
            'Use consistent IDs across all annotation files.'
        ),
        'example': 'MATCH_20260711_Denver_CJ_vs_KIM'
    },
    'round': {
        'type': int,
        'required': True,
        'valid_values': [1, 2, 3],
        'description': (
            'Round number within the match. '
            'Standard WTF point sparring: 3 rounds of 2 minutes. '
            'Golden point overtime rounds should be labeled 4, 5, etc.'
        ),
        'example': 1
    },
    'timestamp_s': {
        'type': float,
        'required': True,
        'description': (
            'Time in seconds from the start of the round '
            'when this exchange begins. '
            'Derive from video footage. '
            'Round starts at 0.0, ends at ~120.0 for standard rounds. '
            'Use 0.0 if exact timing is unavailable.'
        ),
        'example': 14.5
    },
    'exchange_id': {
        'type': int,
        'required': True,
        'description': (
            'Sequential integer identifying this exchange within '
            'the match. Starts at 1, increments by 1 for each '
            'discrete exchange annotated. Never resets between rounds.'
        ),
        'example': 7
    },
    'fighter_id': {
        'type': str,
        'required': True,
        'description': (
            'Unique identifier for Fighter 1 in this exchange. '
            'Use a short consistent code e.g. CJ, KIM, PARK. '
            'Must match the fighter_id used in match metadata.'
        ),
        'example': 'CJ'
    },
    'opponent_id': {
        'type': str,
        'required': True,
        'description': (
            'Unique identifier for Fighter 2 in this exchange. '
            'Same naming convention as fighter_id.'
        ),
        'example': 'KIM'
    },
    'fighter_state': {
        'type': str,
        'required': True,
        'valid_values': ['Attack', 'Defend', 'Disengage', 'Feint'],
        'description': (
            'Tactical state of the primary fighter during this exchange.\n'
            'Attack:    Fighter initiates a scoring technique '
            '(kick, punch) with intent to score.\n'
            'Defend:    Fighter blocks, parries, or absorbs '
            'opponent technique. Includes turtle shell, '
            'Philly shell posture.\n'
            'Disengage: Fighter creates distance, resets position, '
            'or circles without attacking or defending.\n'
            'Feint:     Fighter performs a deceptive movement '
            'intended to provoke a reaction without committing '
            'to a full scoring technique. Includes pump kicks, '
            'level switches, shoulder fakes.'
        ),
        'example': 'Feint'
    },
    'opponent_state': {
        'type': str,
        'required': True,
        'valid_values': ['Attack', 'Defend', 'Disengage', 'Feint'],
        'description': (
            'Tactical state of the opponent during this exchange. '
            'Same definitions as fighter_state applied to the opponent.'
        ),
        'example': 'Defend'
    },
    'winner': {
        'type': str,
        'required': True,
        'valid_values': ['F1', 'F2', 'Double', 'None'],
        'description': (
            'Outcome of this exchange.\n'
            'F1:     Primary fighter scored, opponent did not.\n'
            'F2:     Opponent scored, primary fighter did not.\n'
            'Double: Both fighters scored on the same exchange '
            '(simultaneous contact). Both points fields should '
            'be non-zero.\n'
            'None:   No points scored. Exchange ended with reset, '
            'out of bounds, referee stop, or no valid contact.'
        ),
        'example': 'F1'
    },
    'f1_points': {
        'type': int,
        'required': True,
        'valid_values': [0, 1, 2, 3, 4],
        'description': (
            'Points scored by Fighter 1 in this exchange.\n'
            'WT scoring rules:\n'
            '  1 point: valid punch to trunk protector\n'
            '  2 points: valid kick to trunk protector\n'
            '  3 points: valid kick to head\n'
            '  4 points: valid turning kick to head\n'
            '  0 points: no score (winner=None or winner=F2)\n'
            'Bonus points from penalties are NOT included here — '
            'log only technique points.'
        ),
        'example': 3
    },
    'f2_points': {
        'type': int,
        'required': True,
        'valid_values': [0, 1, 2, 3, 4],
        'description': (
            'Points scored by Fighter 2 in this exchange. '
            'Same point value definitions as f1_points.'
        ),
        'example': 0
    },
    'technique': {
        'type': str,
        'required': False,
        'valid_values': [
            'roundhouse', 'back_kick', 'spinning_heel',
            'axe_kick', 'cut_kick', 'push_kick', 'punch',
            'turning_roundhouse', 'hook_kick', 'other', ''
        ],
        'description': (
            'Primary scoring technique used by the winning fighter. '
            'Leave empty if winner=None or Double or technique '
            'is unclear from footage. '
            'Use the technique that scored points, not feints.'
        ),
        'example': 'roundhouse'
    },
    'body_target': {
        'type': str,
        'required': False,
        'valid_values': ['head', 'trunk', ''],
        'description': (
            'Target area of the scoring technique. '
            'head: above the collarbone. '
            'trunk: trunk protector area. '
            'Leave empty if winner=None.'
        ),
        'example': 'head'
    },
    'penalty': {
        'type': str,
        'required': False,
        'valid_values': ['gam_jeom_f1', 'gam_jeom_f2', 'none', ''],
        'description': (
            'Whether a penalty (gam-jeom) was assessed during '
            'this exchange.\n'
            'gam_jeom_f1: Fighter 1 received a penalty '
            '(awards 1 point to F2).\n'
            'gam_jeom_f2: Fighter 2 received a penalty.\n'
            'none or empty: No penalty.'
        ),
        'example': 'none'
    },
    'notes': {
        'type': str,
        'required': False,
        'description': (
            'Free-text annotation field for anything not captured '
            'by structured fields. Examples: "CJ level switched '
            'from body to head", "opponent attempted counter but '
            'missed", "referee stopped action for boundary". '
            'Keep concise.'
        ),
        'example': 'Level switch head after cut kick feint'
    }
}

# Column order for CSV output
CSV_COLUMN_ORDER = [
    'match_id', 'round', 'timestamp_s', 'exchange_id',
    'fighter_id', 'opponent_id',
    'fighter_state', 'opponent_state',
    'winner', 'f1_points', 'f2_points',
    'technique', 'body_target', 'penalty', 'notes'
]

# Required columns (all others are optional)
REQUIRED_COLUMNS = [
    col for col, spec in ANNOTATION_SCHEMA.items()
    if spec['required']
]

# Valid values for enum columns
VALID_VALUES = {
    col: spec['valid_values']
    for col, spec in ANNOTATION_SCHEMA.items()
    if 'valid_values' in spec
}


# ---------------------------------------------------------------------------
# SECTION 2 — CSV template generator
# ---------------------------------------------------------------------------

def generate_csv_template(filepath,
                           match_id='MATCH_YYYYMMDD_Location_F1_vs_F2',
                           fighter_id='CJ',
                           opponent_id='OPP',
                           n_blank_rows=50,
                           include_examples=True):
    """
    Generate a blank annotation CSV template ready for
    manual coding of real match footage.

    The template includes:
    1. A header block of comments explaining each field
    2. Example rows demonstrating realistic annotation
    3. Blank rows for actual annotation work

    Parameters:
    filepath:         where to save the template
    match_id:         pre-filled match identifier
    fighter_id:       pre-filled fighter code
    opponent_id:      pre-filled opponent code
    n_blank_rows:     number of blank rows to include
    include_examples: whether to include example rows
    """
    dirname = os.path.dirname(filepath)
    os.makedirs(dirname if dirname else '.', exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        # Comment lines are written as raw text, never through
        # csv.writer: several ("Valid values: ...") contain embedded
        # commas, and csv.writer would wrap those in double quotes,
        # turning the on-disk line into `"# Valid values: ..."` —
        # replacing the leading '#' with a '"' and silently breaking
        # every comment-skipping reader downstream.
        comment_lines = [
            '# TAEKWONDO SPARRING ANNOTATION TEMPLATE',
            '# Version: 1.0 | sparring_dynamics project',
            '# ',
            '# FIELD DEFINITIONS:',
        ]

        for col in CSV_COLUMN_ORDER:
            spec = ANNOTATION_SCHEMA[col]
            req = 'REQUIRED' if spec['required'] else 'optional'
            desc = spec['description'].split('\n')[0][:60]
            comment_lines.append(f'# {col} ({req}): {desc}')
            if 'valid_values' in spec:
                vals = ', '.join(str(v) for v in spec['valid_values'] if v != '')
                comment_lines.append(f'#   Valid values: {vals}')

        comment_lines += [
            '# ',
            '# ANNOTATION WORKFLOW:',
            '# 1. Open match video in a video player',
            '# 2. Watch exchange by exchange',
            '# 3. For each exchange, identify:',
            '#    - Which state was each fighter in? (fighter_state, opponent_state)',
            '#    - Who scored? (winner, f1_points, f2_points)',
            '#    - What technique? (technique, body_target)',
            '# 4. Fill one row per exchange',
            '# 5. Delete all comment lines before loading into pipeline',
            '# ',
            '# STATE CODING GUIDE:',
            '# Attack:    Fighter throws a committed scoring technique',
            '# Defend:    Fighter blocks or absorbs opponent technique',
            '# Disengage: Fighter resets distance without attacking/defending',
            '# Feint:     Fighter performs deceptive movement (pump kick,',
            '#            level switch, shoulder fake) without committing',
            '# ',
            '# POINT VALUES (WT rules):',
            '# 1 = valid punch to trunk',
            '# 2 = valid kick to trunk',
            '# 3 = valid kick to head',
            '# 4 = valid turning kick to head',
            '# ',
        ]

        for line in comment_lines:
            f.write(line + '\n')

        writer = csv.writer(f, lineterminator='\n')

        # Write column headers
        writer.writerow(CSV_COLUMN_ORDER)

        # Write example rows
        if include_examples:
            f.write('# --- EXAMPLE ROWS (delete before loading) ---\n')

            examples = [
                # (match_id, round, ts, exch_id, f_id, opp_id,
                #  f_state, opp_state, winner, f1_pts, f2_pts,
                #  technique, target, penalty, notes)
                [match_id, 1, 3.5, 1, fighter_id, opponent_id,
                 'Feint', 'Defend', 'F1', 3, 0,
                 'roundhouse', 'head', 'none',
                 'Cut kick feint opened guard, head kick scored'],

                [match_id, 1, 8.2, 2, fighter_id, opponent_id,
                 'Attack', 'Defend', 'F2', 0, 1,
                 'punch', 'trunk', 'none',
                 'Opponent counter-punched during blitz'],

                [match_id, 1, 12.0, 3, fighter_id, opponent_id,
                 'Disengage', 'Feint', 'None', 0, 0,
                 '', '', 'none',
                 'Pump kick probe, CJ disengaged, no contact'],

                [match_id, 1, 18.7, 4, fighter_id, opponent_id,
                 'Attack', 'Attack', 'Double', 2, 2,
                 'roundhouse', 'trunk', 'none',
                 'Simultaneous trunk kicks both fighters'],

                [match_id, 1, 25.1, 5, fighter_id, opponent_id,
                 'Feint', 'Defend', 'F1', 2, 0,
                 'roundhouse', 'trunk', 'none',
                 'Level switch body after head feint'],

                [match_id, 2, 5.3, 6, fighter_id, opponent_id,
                 'Disengage', 'Attack', 'F2', 0, 3,
                 'roundhouse', 'head', 'none',
                 'Caught during reset, opponent head kick'],

                [match_id, 2, 11.8, 7, fighter_id, opponent_id,
                 'Defend', 'Attack', 'F1', 1, 0,
                 'punch', 'trunk', 'none',
                 'Counter punch off Philly shell block'],

                [match_id, 2, 19.4, 8, fighter_id, opponent_id,
                 'Attack', 'Defend', 'None', 0, 0,
                 '', '', 'gam_jeom_f2',
                 'Opponent grabbed, penalty awarded to F1 as point'],
            ]

            for ex in examples:
                writer.writerow(ex)

            f.write('# --- YOUR ANNOTATIONS BELOW ---\n')

        # Write blank rows
        blank = [''] * len(CSV_COLUMN_ORDER)
        blank[CSV_COLUMN_ORDER.index('match_id')] = match_id
        blank[CSV_COLUMN_ORDER.index('fighter_id')] = fighter_id
        blank[CSV_COLUMN_ORDER.index('opponent_id')] = opponent_id

        for i in range(1, n_blank_rows + 1):
            row = blank.copy()
            row[CSV_COLUMN_ORDER.index('round')] = ''
            row[CSV_COLUMN_ORDER.index('exchange_id')] = i
            writer.writerow(row)

    print(f"Template created: {filepath}")
    print(f"  {n_blank_rows} blank rows pre-filled with "
          f"match_id={match_id}, "
          f"fighter={fighter_id}, opponent={opponent_id}")
    print(f"  Delete all lines starting with # before loading.")
    return filepath


# ---------------------------------------------------------------------------
# SECTION 3 — Validation
# ---------------------------------------------------------------------------

class AnnotationValidationError(Exception):
    """Raised when annotation data fails validation."""
    pass


class AnnotationValidator:
    """
    Validates a list of annotation dicts against the schema.
    Collects all errors before raising, so you see every
    problem at once rather than fixing them one at a time.
    """

    def __init__(self, strict=True):
        """
        strict=True: raise on any error
        strict=False: collect errors and return them,
                      allowing partial loading
        """
        self.strict = strict
        self.errors = []
        self.warnings = []

    def validate_row(self, row, row_num):
        """
        Validate a single annotation row dict.
        Appends to self.errors for any violation.
        Returns True if row is valid, False otherwise.
        """
        row_valid = True

        # Check required fields present and non-empty
        for col in REQUIRED_COLUMNS:
            if col not in row or str(row[col]).strip() == '':
                self.errors.append(
                    f"Row {row_num}: Missing required field '{col}'"
                )
                row_valid = False

        if not row_valid:
            return False

        # Type and value checks
        checks = [
            ('round',         int,   None),
            ('exchange_id',   int,   None),
            ('timestamp_s',   float, None),
            ('f1_points',     int,   None),
            ('f2_points',     int,   None),
        ]

        for col, expected_type, _ in checks:
            if col not in row:
                continue
            try:
                expected_type(row[col])
            except (ValueError, TypeError):
                self.errors.append(
                    f"Row {row_num}: '{col}' must be "
                    f"{expected_type.__name__}, got '{row[col]}'"
                )
                row_valid = False

        # Valid values checks
        for col, valid in VALID_VALUES.items():
            if col not in row:
                continue
            val = str(row[col]).strip()
            if val == '' and not ANNOTATION_SCHEMA[col]['required']:
                continue
            if val not in [str(v) for v in valid]:
                self.errors.append(
                    f"Row {row_num}: '{col}' value '{val}' not in "
                    f"valid values: {valid}"
                )
                row_valid = False

        # Cross-field consistency checks
        if row_valid:
            winner = str(row.get('winner', '')).strip()
            f1_pts = int(row.get('f1_points', 0))
            f2_pts = int(row.get('f2_points', 0))

            # Winner F1: f1_points > 0, f2_points == 0
            if winner == 'F1':
                if f1_pts == 0:
                    self.errors.append(
                        f"Row {row_num}: winner=F1 but f1_points=0. "
                        f"F1 must have scored at least 1 point."
                    )
                    row_valid = False
                if f2_pts != 0:
                    self.errors.append(
                        f"Row {row_num}: winner=F1 but f2_points={f2_pts}. "
                        f"Use winner=Double if both scored."
                    )
                    row_valid = False

            # Winner F2: f2_points > 0, f1_points == 0
            elif winner == 'F2':
                if f2_pts == 0:
                    self.errors.append(
                        f"Row {row_num}: winner=F2 but f2_points=0."
                    )
                    row_valid = False
                if f1_pts != 0:
                    self.errors.append(
                        f"Row {row_num}: winner=F2 but f1_points={f1_pts}. "
                        f"Use winner=Double if both scored."
                    )
                    row_valid = False

            # Double: both > 0
            elif winner == 'Double':
                if f1_pts == 0 or f2_pts == 0:
                    self.errors.append(
                        f"Row {row_num}: winner=Double but "
                        f"f1_points={f1_pts}, f2_points={f2_pts}. "
                        f"Both must be > 0 for a Double."
                    )
                    row_valid = False

            # None: both == 0
            elif winner == 'None':
                if f1_pts != 0 or f2_pts != 0:
                    self.errors.append(
                        f"Row {row_num}: winner=None but points "
                        f"f1={f1_pts} f2={f2_pts} are non-zero."
                    )
                    row_valid = False

            # Round in valid range
            try:
                rnd = int(row.get('round', 1))
                if rnd < 1:
                    self.errors.append(
                        f"Row {row_num}: round={rnd} must be >= 1."
                    )
                    row_valid = False
                if rnd > 5:
                    self.warnings.append(
                        f"Row {row_num}: round={rnd} > 3. "
                        f"OK for overtime, verify intentional."
                    )
            except (ValueError, TypeError):
                pass

            # Timestamp non-negative
            try:
                ts = float(row.get('timestamp_s', 0))
                if ts < 0:
                    self.errors.append(
                        f"Row {row_num}: timestamp_s={ts} "
                        f"must be >= 0."
                    )
                    row_valid = False
            except (ValueError, TypeError):
                pass

            # Exchange ID positive
            try:
                eid = int(row.get('exchange_id', 1))
                if eid < 1:
                    self.errors.append(
                        f"Row {row_num}: exchange_id={eid} "
                        f"must be >= 1."
                    )
                    row_valid = False
            except (ValueError, TypeError):
                pass

            # Points in valid range
            for pt_field in ('f1_points', 'f2_points'):
                try:
                    pts = int(row.get(pt_field, 0))
                    if pts not in [0, 1, 2, 3, 4]:
                        self.errors.append(
                            f"Row {row_num}: {pt_field}={pts} "
                            f"not in [0,1,2,3,4]."
                        )
                        row_valid = False
                except (ValueError, TypeError):
                    pass

        return row_valid

    def validate_sequence(self, annotations):
        """
        Validate sequence-level consistency across all rows.
        Checks that cannot be done row by row:

        1. exchange_id is strictly monotonically increasing
        2. match_id is consistent across all rows
        3. fighter_id and opponent_id are consistent
           (same two fighters throughout)
        4. No duplicate exchange_ids
        5. Timestamps are non-decreasing within each round
        """
        if not annotations:
            return

        match_ids = set(str(r.get('match_id', '')) for r in annotations)
        fighter_ids = set(str(r.get('fighter_id', '')) for r in annotations)
        opponent_ids = set(str(r.get('opponent_id', '')) for r in annotations)

        if len(match_ids) > 1:
            self.errors.append(
                f"Inconsistent match_id: {match_ids}. "
                f"One annotation file should contain one match."
            )

        if len(fighter_ids) > 1:
            self.warnings.append(
                f"Multiple fighter_ids found: {fighter_ids}. "
                f"Verify this is intentional."
            )

        if len(opponent_ids) > 1:
            self.warnings.append(
                f"Multiple opponent_ids found: {opponent_ids}. "
                f"Verify this is intentional."
            )

        # Check exchange_id monotonicity
        try:
            eids = [int(r.get('exchange_id', 0)) for r in annotations]
            for i in range(1, len(eids)):
                if eids[i] <= eids[i - 1]:
                    self.errors.append(
                        f"exchange_id not strictly increasing at "
                        f"position {i + 1}: {eids[i - 1]} → {eids[i]}"
                    )
        except (ValueError, TypeError):
            pass

        # Check no duplicate exchange_ids
        try:
            eids = [int(r.get('exchange_id', 0)) for r in annotations]
            seen = set()
            for eid in eids:
                if eid in seen:
                    self.errors.append(
                        f"Duplicate exchange_id: {eid}"
                    )
                seen.add(eid)
        except (ValueError, TypeError):
            pass

        # Check timestamps non-decreasing within rounds
        try:
            by_round = {}
            for r in annotations:
                rnd = int(r.get('round', 1))
                ts = float(r.get('timestamp_s', 0))
                if rnd not in by_round:
                    by_round[rnd] = []
                by_round[rnd].append(ts)

            for rnd, timestamps in by_round.items():
                for i in range(1, len(timestamps)):
                    if timestamps[i] < timestamps[i - 1]:
                        self.warnings.append(
                            f"Round {rnd}: timestamps not "
                            f"monotonically increasing near "
                            f"position {i + 1}: "
                            f"{timestamps[i - 1]} → {timestamps[i]}. "
                            f"Verify annotation order."
                        )
        except (ValueError, TypeError):
            pass

    def validate(self, annotations):
        """
        Validate a complete list of annotation dicts.

        Returns (valid_annotations, n_errors, n_warnings)

        If strict=True, raises AnnotationValidationError
        if any errors found.
        """
        self.errors = []
        self.warnings = []
        valid_rows = []

        for i, row in enumerate(annotations, start=2):
            row_ok = self.validate_row(row, i)
            if row_ok:
                valid_rows.append(row)

        self.validate_sequence(annotations)

        if self.warnings:
            print(f"\nValidation warnings ({len(self.warnings)}):")
            for w in self.warnings:
                print(f"  ⚠  {w}")

        if self.errors:
            print(f"\nValidation errors ({len(self.errors)}):")
            for e in self.errors:
                print(f"  ✗  {e}")

            if self.strict:
                raise AnnotationValidationError(
                    f"{len(self.errors)} validation errors found. "
                    f"Fix the annotation file and reload."
                )
            else:
                print(f"\n  Strict mode off — loading "
                      f"{len(valid_rows)}/{len(annotations)} "
                      f"valid rows.")
        else:
            print(f"\n  ✓ All {len(annotations)} rows valid.")

        return valid_rows, len(self.errors), len(self.warnings)


# ---------------------------------------------------------------------------
# SECTION 4 — Loading code
# ---------------------------------------------------------------------------

def load_annotation_csv(filepath,
                         strict=True,
                         skip_comments=True):
    """
    Load a taekwondo sparring annotation CSV file.

    Handles:
    - Comment lines starting with #
    - Optional fields missing from CSV
    - Type coercion for numeric fields
    - Validation via AnnotationValidator

    Parameters:
    filepath:      path to annotation CSV
    strict:        if True, raises on any validation error
    skip_comments: if True, skip lines starting with #

    Returns:
    annotations:   list of validated annotation dicts
    metadata:      summary dict with match info
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Annotation file not found: {filepath}\n"
            f"Generate a template with: "
            f"generate_csv_template('{filepath}')"
        )

    raw_rows = []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(
            (line for line in f
             if not (skip_comments and
                     line.strip().startswith('#'))),
        )
        for row in reader:
            # Skip blank rows (all fields empty). DictReader fills any
            # column missing from a short/malformed row with None
            # (restval defaults to None), so guard against that here
            # rather than assuming every value is a string.
            if all((v or '').strip() == '' for v in row.values()):
                continue
            raw_rows.append(dict(row))

    if not raw_rows:
        raise ValueError(
            f"No data rows found in {filepath}. "
            f"Check that comment lines are removed and "
            f"at least one data row exists."
        )

    # Type coercion
    coerce_int = ['round', 'exchange_id', 'f1_points', 'f2_points']
    coerce_float = ['timestamp_s']

    for row in raw_rows:
        for col in coerce_int:
            if col in row and row[col] is not None and row[col].strip() != '':
                try:
                    row[col] = int(float(row[col]))
                except (ValueError, TypeError):
                    pass
        for col in coerce_float:
            if col in row and row[col] is not None and row[col].strip() != '':
                try:
                    row[col] = float(row[col])
                except (ValueError, TypeError):
                    pass
        # Strip whitespace from all string fields
        for col in row:
            if isinstance(row[col], str):
                row[col] = row[col].strip()

    # Validate
    validator = AnnotationValidator(strict=strict)
    valid_rows, n_errors, n_warnings = validator.validate(raw_rows)

    # Build metadata summary
    metadata = _build_metadata_summary(valid_rows, filepath)

    return valid_rows, metadata


def _build_metadata_summary(annotations, filepath):
    """
    Build a summary dict describing the loaded annotation set.
    """
    if not annotations:
        return {}

    rounds = sorted(set(r.get('round', 1) for r in annotations))
    match_ids = set(str(r.get('match_id', '')) for r in annotations)
    fighter_ids = set(str(r.get('fighter_id', '')) for r in annotations)
    opp_ids = set(str(r.get('opponent_id', '')) for r in annotations)

    f1_wins = sum(1 for r in annotations if r.get('winner') == 'F1')
    f2_wins = sum(1 for r in annotations if r.get('winner') == 'F2')
    doubles = sum(1 for r in annotations if r.get('winner') == 'Double')
    no_score = sum(1 for r in annotations if r.get('winner') == 'None')

    total_f1_pts = sum(int(r.get('f1_points', 0)) for r in annotations)
    total_f2_pts = sum(int(r.get('f2_points', 0)) for r in annotations)

    state_combos = {}
    for r in annotations:
        key = (r.get('fighter_state', ''), r.get('opponent_state', ''))
        state_combos[key] = state_combos.get(key, 0) + 1

    most_common_combo = max(state_combos, key=state_combos.get) \
        if state_combos else ('', '')

    return {
        'filepath':             filepath,
        'match_ids':            list(match_ids),
        'fighter_ids':          list(fighter_ids),
        'opponent_ids':         list(opp_ids),
        'rounds':               rounds,
        'n_exchanges':          len(annotations),
        'n_rounds':             len(rounds),
        'f1_wins':              f1_wins,
        'f2_wins':              f2_wins,
        'doubles':              doubles,
        'no_score':             no_score,
        'total_f1_points':      total_f1_pts,
        'total_f2_points':      total_f2_pts,
        'f1_win_rate':          f1_wins / len(annotations),
        'f2_win_rate':          f2_wins / len(annotations),
        'most_common_state_combo': most_common_combo,
        'most_common_combo_count': state_combos.get(
            most_common_combo, 0
        ),
    }


def print_annotation_summary(metadata):
    """
    Print a formatted summary of loaded annotation data.
    """
    print(f"\n{'='*55}")
    print(f"ANNOTATION SUMMARY")
    print(f"{'='*55}")
    print(f"  File:          {metadata.get('filepath','')}")
    print(f"  Match ID(s):   {metadata.get('match_ids','')}")
    print(f"  Fighter:       {metadata.get('fighter_ids','')}")
    print(f"  Opponent:      {metadata.get('opponent_ids','')}")
    print(f"  Rounds:        {metadata.get('rounds','')}")
    print(f"  Exchanges:     {metadata.get('n_exchanges',0)}")
    print(f"\n  Outcomes:")
    n = metadata.get('n_exchanges', 1)
    print(f"    F1 wins:     {metadata.get('f1_wins',0):3d}  "
          f"({100*metadata.get('f1_win_rate',0):.1f}%)")
    print(f"    F2 wins:     {metadata.get('f2_wins',0):3d}  "
          f"({100*metadata.get('f2_win_rate',0):.1f}%)")
    print(f"    Doubles:     {metadata.get('doubles',0):3d}  "
          f"({100*metadata.get('doubles',0)/n:.1f}%)")
    print(f"    No score:    {metadata.get('no_score',0):3d}  "
          f"({100*metadata.get('no_score',0)/n:.1f}%)")
    print(f"\n  Points scored:")
    print(f"    Fighter 1:   {metadata.get('total_f1_points',0)}")
    print(f"    Fighter 2:   {metadata.get('total_f2_points',0)}")
    print(f"\n  Most common state combo:")
    combo = metadata.get('most_common_state_combo', ('', ''))
    count = metadata.get('most_common_combo_count', 0)
    print(f"    {combo[0]} vs {combo[1]}: {count} exchanges "
          f"({100*count/n:.1f}%)")
    print(f"{'='*55}")


# ---------------------------------------------------------------------------
# SECTION 5 — Conversion to pipeline formats
# ---------------------------------------------------------------------------

def annotations_to_exchanges(annotations):
    """
    Convert loaded annotations to the exchange format
    expected by estimate_payoffs_v2.py.

    Returns list of dicts:
    {f1_state, f2_state, winner, f1_points, f2_points}
    """
    return [
        {
            'f1_state':  r['fighter_state'],
            'f2_state':  r['opponent_state'],
            'winner':    r['winner'],
            'f1_points': int(r.get('f1_points', 0)),
            'f2_points': int(r.get('f2_points', 0)),
        }
        for r in annotations
    ]


def annotations_to_sequences(annotations):
    """
    Convert loaded annotations to state sequence format
    expected by estimate_transitions.py.

    Groups by round and extracts ordered state sequences
    for both fighters.

    Returns:
    f1_sequences: list of lists of state strings (fighter)
    f2_sequences: list of lists of state strings (opponent)
    """
    from collections import defaultdict

    by_round = defaultdict(list)
    for r in sorted(annotations,
                     key=lambda x: (
                         int(x.get('round', 1)),
                         int(x.get('exchange_id', 0))
                     )):
        by_round[int(r.get('round', 1))].append(r)

    f1_sequences = []
    f2_sequences = []

    for rnd in sorted(by_round.keys()):
        round_rows = by_round[rnd]
        f1_seq = [r['fighter_state'] for r in round_rows]
        f2_seq = [r['opponent_state'] for r in round_rows]

        if len(f1_seq) > 1:
            f1_sequences.append(f1_seq)
            f2_sequences.append(f2_seq)

    return f1_sequences, f2_sequences


def annotations_to_dataframe(annotations):
    """
    Convert annotation list to a pandas DataFrame
    for analysis, filtering, and export.
    """
    return pd.DataFrame(annotations, columns=CSV_COLUMN_ORDER)
