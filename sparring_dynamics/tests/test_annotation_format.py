"""Tests for sparring_dynamics.data.annotation_format."""
import csv
import os
import shutil
import tempfile

from sparring_dynamics.data.annotation_format import (
    ANNOTATION_SCHEMA, CSV_COLUMN_ORDER,
    generate_csv_template,
    AnnotationValidator, AnnotationValidationError,
    load_annotation_csv, _build_metadata_summary,
    annotations_to_exchanges, annotations_to_sequences,
    annotations_to_dataframe,
)


def _base_row(**overrides):
    """A minimal, fully-valid annotation row dict — override fields per test."""
    row = {
        'match_id': 'MATCH_TEST_1',
        'round': 1,
        'timestamp_s': 0.0,
        'exchange_id': 1,
        'fighter_id': 'CJ',
        'opponent_id': 'OPP',
        'fighter_state': 'Attack',
        'opponent_state': 'Defend',
        'winner': 'F1',
        'f1_points': 2,
        'f2_points': 0,
    }
    row.update(overrides)
    return row


def _write_csv(filepath, rows, header=CSV_COLUMN_ORDER, comment_lines=None):
    """Write a plain (non-template) annotation CSV for loader tests."""
    with open(filepath, 'w', newline='') as f:
        if comment_lines:
            for line in comment_lines:
                f.write(line + '\n')
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(col, '') for col in header])


def test_template_creation():
    tmpdir = tempfile.mkdtemp()
    try:
        filepath = os.path.join(tmpdir, 'template.csv')
        generate_csv_template(filepath, match_id='MATCH_X', fighter_id='CJ',
                               opponent_id='OPP', n_blank_rows=5)
        assert os.path.exists(filepath)

        with open(filepath) as f:
            content = f.read()

        header_line = ','.join(CSV_COLUMN_ORDER)
        assert header_line in content
        assert 'roundhouse' in content
        assert 'MATCH_X' in content
        print("  test_template_creation: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_valid_annotations():
    tmpdir = tempfile.mkdtemp()
    try:
        filepath = os.path.join(tmpdir, 'valid.csv')
        rows = [
            _base_row(exchange_id=1, winner='F1', f1_points=2, f2_points=0),
            _base_row(exchange_id=2, winner='F2', f1_points=0, f2_points=1,
                      fighter_state='Defend', opponent_state='Attack'),
            _base_row(exchange_id=3, winner='None', f1_points=0, f2_points=0,
                      fighter_state='Disengage', opponent_state='Feint'),
        ]
        _write_csv(filepath, rows)

        annotations, metadata = load_annotation_csv(filepath, strict=True)
        assert len(annotations) == 3
        assert metadata['n_exchanges'] == 3
        print("  test_load_valid_annotations: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_load_strips_comments():
    tmpdir = tempfile.mkdtemp()
    try:
        filepath = os.path.join(tmpdir, 'commented.csv')
        rows = [
            _base_row(exchange_id=1),
            _base_row(exchange_id=2, winner='F2', f1_points=0, f2_points=1),
        ]
        _write_csv(filepath, rows, comment_lines=[
            '# This is a comment header',
            '# Another comment line, with a comma in it',
        ])

        annotations, metadata = load_annotation_csv(filepath, strict=True)
        assert len(annotations) == 2
        for row in annotations:
            for val in row.values():
                assert not str(val).startswith('#')
        print("  test_load_strips_comments: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_validation_winner_f1_requires_f1_points():
    row = _base_row(winner='F1', f1_points=0, f2_points=0)
    validator = AnnotationValidator(strict=False)
    valid_rows, n_errors, n_warnings = validator.validate([row])
    assert n_errors > 0
    assert len(valid_rows) == 0
    print("  test_validation_winner_f1_requires_f1_points: PASS")


def test_validation_winner_none_requires_zero_points():
    row = _base_row(winner='None', f1_points=2, f2_points=0)
    validator = AnnotationValidator(strict=False)
    valid_rows, n_errors, n_warnings = validator.validate([row])
    assert n_errors > 0
    assert len(valid_rows) == 0
    print("  test_validation_winner_none_requires_zero_points: PASS")


def test_validation_double_requires_both_points():
    row = _base_row(winner='Double', f1_points=2, f2_points=0)
    validator = AnnotationValidator(strict=False)
    valid_rows, n_errors, n_warnings = validator.validate([row])
    assert n_errors > 0
    assert len(valid_rows) == 0
    print("  test_validation_double_requires_both_points: PASS")


def test_validation_invalid_state():
    row = _base_row(fighter_state='Sprint')
    validator = AnnotationValidator(strict=False)
    valid_rows, n_errors, n_warnings = validator.validate([row])
    assert n_errors > 0
    assert len(valid_rows) == 0
    print("  test_validation_invalid_state: PASS")


def test_validation_exchange_id_monotonic():
    rows = [
        _base_row(exchange_id=2, winner='F1', f1_points=2, f2_points=0),
        _base_row(exchange_id=1, winner='F2', f1_points=0, f2_points=1),
    ]
    validator = AnnotationValidator(strict=False)
    valid_rows, n_errors, n_warnings = validator.validate(rows)
    assert n_errors > 0
    assert any('exchange_id not strictly increasing' in e for e in validator.errors)
    print("  test_validation_exchange_id_monotonic: PASS")


def test_annotations_to_exchanges_format():
    annotations = [
        _base_row(exchange_id=1, winner='F1', f1_points=2, f2_points=0),
        _base_row(exchange_id=2, winner='F2', f1_points=0, f2_points=1),
    ]
    exchanges = annotations_to_exchanges(annotations)
    assert len(exchanges) == 2
    required_keys = {'f1_state', 'f2_state', 'winner', 'f1_points', 'f2_points'}
    for ex in exchanges:
        assert required_keys.issubset(set(ex.keys()))
    print("  test_annotations_to_exchanges_format: PASS")


def test_annotations_to_sequences_by_round():
    annotations = []
    eid = 1
    for rnd in (1, 2, 3):
        for _ in range(4):
            annotations.append(_base_row(round=rnd, exchange_id=eid))
            eid += 1

    f1_seqs, f2_seqs = annotations_to_sequences(annotations)
    assert len(f1_seqs) == 3
    assert len(f2_seqs) == 3
    for seq in f1_seqs:
        assert len(seq) == 4
    print("  test_annotations_to_sequences_by_round: PASS")


def test_annotations_to_dataframe_columns():
    annotations = [_base_row(exchange_id=1), _base_row(exchange_id=2)]
    df = annotations_to_dataframe(annotations)
    assert list(df.columns) == CSV_COLUMN_ORDER
    assert len(df) == 2
    print("  test_annotations_to_dataframe_columns: PASS")


def test_metadata_summary_win_rates():
    annotations = [
        _base_row(exchange_id=1, winner='F1', f1_points=2, f2_points=0),
        _base_row(exchange_id=2, winner='F1', f1_points=1, f2_points=0),
        _base_row(exchange_id=3, winner='F1', f1_points=3, f2_points=0),
        _base_row(exchange_id=4, winner='F2', f1_points=0, f2_points=1),
        _base_row(exchange_id=5, winner='F2', f1_points=0, f2_points=2),
        _base_row(exchange_id=6, winner='None', f1_points=0, f2_points=0),
    ]
    metadata = _build_metadata_summary(annotations, 'dummy.csv')
    assert metadata['n_exchanges'] == 6
    assert abs(metadata['f1_win_rate'] - 0.5) < 1e-9
    assert abs(metadata['f2_win_rate'] - (2 / 6)) < 1e-9
    assert metadata['no_score'] == 1
    print("  test_metadata_summary_win_rates: PASS")


def test_strict_vs_non_strict():
    tmpdir = tempfile.mkdtemp()
    try:
        filepath = os.path.join(tmpdir, 'one_bad_row.csv')
        rows = [
            _base_row(exchange_id=1, winner='F1', f1_points=2, f2_points=0),
            _base_row(exchange_id=2, winner='F1', f1_points=0, f2_points=0),  # invalid
            _base_row(exchange_id=3, winner='F2', f1_points=0, f2_points=1),
        ]
        _write_csv(filepath, rows)

        raised = False
        try:
            load_annotation_csv(filepath, strict=True)
        except AnnotationValidationError:
            raised = True
        assert raised, "strict=True should have raised AnnotationValidationError"

        annotations, metadata = load_annotation_csv(filepath, strict=False)
        assert len(annotations) == 2
        print("  test_strict_vs_non_strict: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def run_all():
    print("test_annotation_format.py")
    test_template_creation()
    test_load_valid_annotations()
    test_load_strips_comments()
    test_validation_winner_f1_requires_f1_points()
    test_validation_winner_none_requires_zero_points()
    test_validation_double_requires_both_points()
    test_validation_invalid_state()
    test_validation_exchange_id_monotonic()
    test_annotations_to_exchanges_format()
    test_annotations_to_sequences_by_round()
    test_annotations_to_dataframe_columns()
    test_metadata_summary_win_rates()
    test_strict_vs_non_strict()


if __name__ == "__main__":
    run_all()
    print("ALL test_annotation_format.py TESTS PASSED")
