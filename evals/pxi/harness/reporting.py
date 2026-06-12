"""Reporting helpers for the PXI eval experiment runner.

Pure functions over :class:`RanExperiment` and :class:`EvalDataset` -- no
Phoenix network calls -- so the console summary and failure reports stay
unit-testable without a live server.
"""

from __future__ import annotations

from typing import Any, Sequence
from urllib.parse import urljoin

from evals.pxi.harness.datasets import EvalDataset
from phoenix.client.resources.experiments.types import ExperimentEvaluationRun, RanExperiment

PASSING_SCORE = 1.0
MAX_TABLE_CELL_WIDTH = 80


def _format_table(headers: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> str:
    widths = [
        max(len(header), *(len(row[index]) for row in rows)) for index, header in enumerate(headers)
    ]
    rule = "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def format_row(row: tuple[str, ...]) -> str:
        cells = [value.ljust(widths[index]) for index, value in enumerate(row)]
        return "| " + " | ".join(cells) + " |"

    lines = [rule, format_row(headers), rule]
    lines.extend(format_row(row) for row in rows)
    lines.append(rule)
    return "\n".join(lines)


def _truncate_cell(value: Any) -> str:
    text = str(value)
    if len(text) <= MAX_TABLE_CELL_WIDTH:
        return text
    return text[: MAX_TABLE_CELL_WIDTH - 3] + "..."


def _result_field(evaluation_run: ExperimentEvaluationRun, key: str) -> str:
    result = evaluation_run.result
    if not isinstance(result, dict):
        return ""
    value = result.get(key)
    return "" if value is None else _truncate_cell(value)


def _task_error_rows(experiment: RanExperiment) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for task_run in experiment.get("task_runs", []):
        output = task_run.get("output")
        output_error = output.get("error") if isinstance(output, dict) else None
        error = task_run.get("error") or output_error
        if not error:
            continue
        stable_example_id = output.get("stable_example_id") if isinstance(output, dict) else None
        example_id = stable_example_id or task_run["dataset_example_id"]
        rows.append((_truncate_cell(example_id), _truncate_cell(error)))
    return rows


def _failed_evaluation_rows(
    experiment: RanExperiment,
    evaluation_runs: Sequence[ExperimentEvaluationRun],
) -> list[tuple[str, str, str, str, str]]:
    task_runs_by_id = {
        str(task_run["id"]): task_run
        for task_run in experiment.get("task_runs", [])
        if "id" in task_run
    }
    rows: list[tuple[str, str, str, str, str]] = []
    for evaluation_run in evaluation_runs:
        score = _score(evaluation_run)
        if evaluation_run.error is None and score is not None and score >= PASSING_SCORE:
            continue
        task_run = task_runs_by_id.get(str(evaluation_run.experiment_run_id))
        example_id = (
            task_run["dataset_example_id"] if task_run else str(evaluation_run.experiment_run_id)
        )
        rows.append(
            (
                _truncate_cell(example_id),
                _truncate_cell(evaluation_run.name or "unknown"),
                (
                    "error"
                    if evaluation_run.error is not None
                    else "missing"
                    if score is None
                    else f"{score:g}"
                ),
                _result_field(evaluation_run, "label"),
                _truncate_cell(
                    evaluation_run.error or _result_field(evaluation_run, "explanation")
                ),
            )
        )
    return rows


def _score(evaluation_run: ExperimentEvaluationRun) -> float | None:
    result = evaluation_run.result
    if not isinstance(result, dict):
        return None
    value = result.get("score")
    return float(value) if isinstance(value, (int, float)) else None


def _example_splits_by_id(dataset: EvalDataset) -> dict[str, set[str]]:
    return {
        str(example["id"]): {str(split) for split in example["splits"]}
        for example in dataset.examples
    }


def _is_failed_evaluation(evaluation_run: ExperimentEvaluationRun) -> bool:
    score = _score(evaluation_run)
    return evaluation_run.error is not None or score is None or score < PASSING_SCORE


def _evaluator_summary_rows(
    evaluation_runs: Sequence[ExperimentEvaluationRun],
) -> list[tuple[str, dict[str, int]]]:
    by_evaluator: dict[str, dict[str, int]] = {}
    for evaluation_run in evaluation_runs:
        name = str(evaluation_run.name or "unknown")
        summary = by_evaluator.setdefault(
            name,
            {"total": 0, "passing": 0, "failing": 0, "missing_score": 0, "errors": 0},
        )
        summary["total"] += 1
        score = _score(evaluation_run)
        if evaluation_run.error is not None:
            summary["errors"] += 1
            summary["failing"] += 1
        elif score is None:
            summary["missing_score"] += 1
            summary["failing"] += 1
        elif score >= PASSING_SCORE:
            summary["passing"] += 1
        else:
            summary["failing"] += 1
    return sorted(by_evaluator.items())


def _has_regression_evaluator_failure(
    dataset: EvalDataset,
    experiment: RanExperiment,
    evaluation_runs: Sequence[ExperimentEvaluationRun],
) -> bool:
    splits_by_id = _example_splits_by_id(dataset)
    task_runs_by_id = {
        str(task_run["id"]): task_run
        for task_run in experiment.get("task_runs", [])
        if "id" in task_run
    }
    for evaluation_run in evaluation_runs:
        if not _is_failed_evaluation(evaluation_run):
            continue
        task_run = task_runs_by_id.get(str(evaluation_run.experiment_run_id))
        if task_run is None:
            continue
        example_id = str(task_run["dataset_example_id"])
        if "regression" in splits_by_id.get(example_id, set()):
            return True
    return False


def _print_score_summary(dataset: EvalDataset, experiment: RanExperiment, *, base_url: str) -> bool:
    if experiment["experiment_id"]:
        experiment_url = urljoin(
            base_url.rstrip("/") + "/",
            f"datasets/{experiment['dataset_id']}/compare?experimentId={experiment['experiment_id']}",
        )
        print(f"Experiment: {experiment_url}")
    evaluation_runs = list(experiment.get("evaluation_runs") or [])
    evaluator_summaries = _evaluator_summary_rows(evaluation_runs)

    print(f"Dataset: {dataset.dataset_name} ({len(experiment.get('task_runs', []))} examples run)")
    task_error_rows = _task_error_rows(experiment)
    if task_error_rows:
        print(f"Task errors: {len(task_error_rows)}/{len(experiment.get('task_runs', []))}")
        print(_format_table(("Example", "Error"), task_error_rows))
    if evaluator_summaries:
        rows = []
        for name, summary in evaluator_summaries:
            total = int(summary["total"])
            passing = int(summary["passing"])
            pass_rate = f"{passing / total:.0%}" if total else "n/a"
            rows.append(
                (
                    name,
                    str(total),
                    str(passing),
                    str(summary["failing"]),
                    str(summary["errors"]),
                    str(summary["missing_score"]),
                    pass_rate,
                )
            )
        print(f"Evaluator results (passing score >= {PASSING_SCORE:g}):")
        print(
            _format_table(
                ("Evaluator", "Total", "Passed", "Failed", "Errors", "Missing", "Pass Rate"),
                rows,
            )
        )
        failed_rows = _failed_evaluation_rows(experiment, evaluation_runs)
        if failed_rows:
            print("Failed evaluations:")
            print(_format_table(("Example", "Evaluator", "Score", "Label", "Details"), failed_rows))
    return _has_regression_evaluator_failure(dataset, experiment, evaluation_runs)
