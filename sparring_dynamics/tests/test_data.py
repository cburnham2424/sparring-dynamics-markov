"""Tests for sparring_dynamics.data (loader + validator)."""
import csv
import os
import tempfile

from sparring_dynamics.data.loader import (
    load_exchange_csv, load_sequence_csv,
    create_annotation_template, generate_placeholder_csv,
)
from sparring_dynamics.data.validator import (
    validate_stochastic_matrix, validate_payoff_matrix, validate_exchanges,
)
from sparring_dynamics.config import F1_BASE_DEFAULT, F1_PAYOFF_DEFAULT


def test_generate_and_load_placeholder_csv():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "placeholder.csv")
        generate_placeholder_csv(path, n_exchanges=60)
        assert os.path.exists(path)

        exchanges = load_exchange_csv(path)
        assert len(exchanges) == 60
        for ex in exchanges:
            assert ex['winner'] in {'F1', 'F2', 'Double', 'None'}
            assert ex['f1_points'] >= 0 and ex['f2_points'] >= 0
    print("  test_generate_and_load_placeholder_csv: PASS")


def test_load_exchange_csv_rejects_bad_state():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["f1_state", "f2_state", "winner", "f1_points", "f2_points"])
            w.writerow(["NotAState", "Defend", "F1", "2", "0"])
        try:
            load_exchange_csv(path)
            raised = False
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for invalid state"
    print("  test_load_exchange_csv_rejects_bad_state: PASS")


def test_load_exchange_csv_missing_file():
    try:
        load_exchange_csv("/nonexistent/path/does_not_exist.csv")
        raised = False
    except FileNotFoundError:
        raised = True
    assert raised
    print("  test_load_exchange_csv_missing_file: PASS")


def test_load_sequence_csv():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sequences.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fighter", "sequence"])
            w.writerow(["F1", "Attack,Feint,Attack,Disengage"])
            w.writerow(["F2", "Defend,Attack,Defend,Feint"])
        f1_seqs, f2_seqs = load_sequence_csv(path)
        assert f1_seqs == [["Attack", "Feint", "Attack", "Disengage"]]
        assert f2_seqs == [["Defend", "Attack", "Defend", "Feint"]]
    print("  test_load_sequence_csv: PASS")


def test_annotation_template_is_loadable_as_csv():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "template.csv")
        create_annotation_template(path, n_rows=10)
        assert os.path.exists(path)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                'f1_state', 'f2_state', 'winner', 'f1_points', 'f2_points'
            ]
        # The template's one worked example row should load cleanly.
        exchanges = load_exchange_csv(path)
        assert len(exchanges) == 1
    print("  test_annotation_template_is_loadable_as_csv: PASS")


def test_validate_stochastic_matrix():
    assert validate_stochastic_matrix(F1_BASE_DEFAULT, "F1_BASE_DEFAULT") is True

    bad = F1_BASE_DEFAULT.copy()
    bad[0, 0] += 1.0  # break row sum
    try:
        validate_stochastic_matrix(bad, "bad matrix")
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("  test_validate_stochastic_matrix: PASS")


def test_validate_payoff_matrix():
    assert validate_payoff_matrix(F1_PAYOFF_DEFAULT, "F1_PAYOFF_DEFAULT") is True

    bad = F1_PAYOFF_DEFAULT.copy()
    bad[0, 0] = 1.5  # out of [0,1]
    try:
        validate_payoff_matrix(bad, "bad payoff")
        raised = False
    except ValueError:
        raised = True
    assert raised
    print("  test_validate_payoff_matrix: PASS")


def test_validate_exchanges_coverage_report():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "placeholder.csv")
        generate_placeholder_csv(path, n_exchanges=60)
        exchanges = load_exchange_csv(path)
        report = validate_exchanges(exchanges)
        assert report['total'] == 60
        assert report['coverage'].shape == (4, 4)
    print("  test_validate_exchanges_coverage_report: PASS")


def run_all():
    print("test_data.py")
    test_generate_and_load_placeholder_csv()
    test_load_exchange_csv_rejects_bad_state()
    test_load_exchange_csv_missing_file()
    test_load_sequence_csv()
    test_annotation_template_is_loadable_as_csv()
    test_validate_stochastic_matrix()
    test_validate_payoff_matrix()
    test_validate_exchanges_coverage_report()


if __name__ == "__main__":
    run_all()
    print("ALL test_data.py TESTS PASSED")
