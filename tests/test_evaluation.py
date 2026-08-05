from pathlib import Path

from egg_companion.evaluation import evaluate_trace, load_trace
from egg_companion.evaluation.event_boundaries import boundary_metrics
from egg_companion.evaluation.identity_metrics import identity_metrics


def test_baseline_trace_passes_every_gate() -> None:
    trace = load_trace(Path(__file__).parent / "fixtures" / "traces" / "baseline.json")

    passed, report = evaluate_trace(trace)

    assert passed
    assert all(report["gates"].values())
    assert report["retrieval"]["provenance_coverage"] == 1.0


def test_metrics_report_false_boundaries_and_identity_merges() -> None:
    boundaries = boundary_metrics([10], [10, 30])
    identities = identity_metrics(
        [
            {"truth_id": "a", "predicted_id": "same"},
            {"truth_id": "b", "predicted_id": "same"},
        ]
    )

    assert boundaries["false_positives"] == 1
    assert identities["false_merges"] == 1
