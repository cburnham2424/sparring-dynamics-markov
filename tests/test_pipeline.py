"""End-to-end tests for sparring_dynamics.pipeline."""
import importlib
import os
import tempfile
from argparse import Namespace
from contextlib import contextmanager

from sparring_dynamics import pipeline
from sparring_dynamics.data.loader import generate_placeholder_csv
from sparring_dynamics.config import OUTPUT_DIR


@contextmanager
def temp_cwd():
    """Run a block inside a fresh temp directory, restoring cwd afterward."""
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield tmp
        finally:
            os.chdir(old_cwd)


def test_pipeline_with_defaults():
    """
    Run pipeline with --no-estimate --n-sims 10 --n-steps 50.
    Confirm outputs directory contains both PNG files.
    Confirm no exceptions raised.
    """
    with temp_cwd():
        args = Namespace(csv='annotation.csv', template=False, no_estimate=True,
                          n_sims=10, n_steps=50, selection=1.0, seed=42)
        result = pipeline.run_pipeline(args)  # raises on failure

        assert os.path.exists(os.path.join(OUTPUT_DIR, 'figures', 'monte_carlo_summary.png'))
        assert os.path.exists(os.path.join(OUTPUT_DIR, 'figures', 'monte_carlo_distributions.png'))
        assert result['transition_result']['f1_estimated'] is False
        assert result['payoff_result']['estimated'] is False

    print("  test_pipeline_with_defaults: PASS")


def test_pipeline_with_template():
    """
    Run --template flag.
    Confirm annotation.csv template was created.
    Confirm it has correct headers.
    """
    with temp_cwd():
        pipeline.main(['--template'])
        assert os.path.exists('annotation.csv')

        import csv
        with open('annotation.csv', newline='') as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
                'f1_state', 'f2_state', 'winner', 'f1_points', 'f2_points'
            ]

    print("  test_pipeline_with_template: PASS")


def test_pipeline_with_placeholder_csv():
    """
    Generate placeholder_exchanges.csv using generate_placeholder_csv().
    Run pipeline with that CSV.
    Confirm payoff matrices are marked as estimated (not default).
    Confirm output PNGs saved.
    """
    with temp_cwd():
        generate_placeholder_csv('placeholder_exchanges.csv', n_exchanges=60)

        args = Namespace(csv='placeholder_exchanges.csv', template=False, no_estimate=False,
                          n_sims=10, n_steps=50, selection=1.0, seed=42)
        result = pipeline.run_pipeline(args)

        assert result['payoff_result']['estimated'] is True
        assert os.path.exists(os.path.join(OUTPUT_DIR, 'figures', 'monte_carlo_summary.png'))
        assert os.path.exists(os.path.join(OUTPUT_DIR, 'figures', 'monte_carlo_distributions.png'))

    print("  test_pipeline_with_placeholder_csv: PASS")


def test_module_imports():
    """Import every module in sparring_dynamics package. Confirm no ImportError."""
    modules = [
        'sparring_dynamics',
        'sparring_dynamics.config',
        'sparring_dynamics.data',
        'sparring_dynamics.data.loader',
        'sparring_dynamics.data.validator',
        'sparring_dynamics.estimation',
        'sparring_dynamics.estimation.transitions',
        'sparring_dynamics.estimation.payoffs',
        'sparring_dynamics.simulation',
        'sparring_dynamics.simulation.fighter',
        'sparring_dynamics.simulation.match',
        'sparring_dynamics.analysis',
        'sparring_dynamics.analysis.statistics',
        'sparring_dynamics.analysis.monte_carlo',
        'sparring_dynamics.analysis.monte_carlo_legacy',
        'sparring_dynamics.visualization',
        'sparring_dynamics.visualization.plots',
        'sparring_dynamics.models.markov_two_agent',
        'sparring_dynamics.pipeline',
    ]
    for name in modules:
        importlib.import_module(name)
    print(f"  test_module_imports: PASS ({len(modules)} modules)")


def run_all():
    print("test_pipeline.py")
    test_pipeline_with_defaults()
    test_pipeline_with_template()
    test_pipeline_with_placeholder_csv()
    test_module_imports()


if __name__ == "__main__":
    run_all()
    print("ALL test_pipeline.py TESTS PASSED")
