# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Outcome classification and multi-run reliability for simulation results.

A simulation run ends in one of three outcomes:

* ``PASS`` - the agent met every goal and expectation.
* ``FAIL`` - the agent missed a goal or expectation.
* ``HARNESS_ERROR`` - the simulated user could not produce its next turn
  (for example its model call failed after all retries), so the
  conversation stopped early through no fault of the agent. These runs
  are inconclusive: they are reported separately and excluded from the
  pass rate and from ``k`` in the reliability metrics.

With ``runs > 1`` every case is also summarised across its scored runs:

* ``pass^k`` - the case passed in *all* k scored runs (consistency).
* ``pass@k`` - the case passed in *at least one* of k scored runs.
* ``label`` - ``stable_pass``, ``stable_fail``, ``flaky`` (mixed), or
  ``inconclusive`` when every run was a harness error.
"""

import enum
from collections.abc import Iterable
from typing import Any


class SimOutcome(str, enum.Enum):
    """Final outcome of a single simulation run."""

    PASS = "PASS"
    FAIL = "FAIL"
    HARNESS_ERROR = "HARNESS_ERROR"


class ReliabilityLabel(str, enum.Enum):
    """Stability of a case across repeated runs."""

    STABLE_PASS = "stable_pass"
    STABLE_FAIL = "stable_fail"
    FLAKY = "flaky"
    INCONCLUSIVE = "inconclusive"


_VALID_OUTCOMES = {o.value for o in SimOutcome}


def result_outcome(row: dict[str, Any]) -> str:
    """Returns the outcome of a result row.

    Rows written before outcomes existed only carry ``passed``; they are
    mapped to ``PASS``/``FAIL`` so older ``sim_results.json`` files keep
    loading.
    """
    outcome = row.get("outcome")
    if isinstance(outcome, SimOutcome):
        return outcome.value
    if isinstance(outcome, str) and outcome in _VALID_OUTCOMES:
        return outcome
    if isinstance(row.get("harness_error"), dict):
        return SimOutcome.HARNESS_ERROR.value
    return SimOutcome.PASS.value if row.get("passed") else SimOutcome.FAIL.value


def is_harness_error(row: dict[str, Any]) -> bool:
    """True when the run is inconclusive because the harness failed."""
    return result_outcome(row) == SimOutcome.HARNESS_ERROR.value


def summarize_outcomes(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Counts outcomes and computes a pass rate that ignores harness errors.

    Returns:
        A dict with ``total``, ``passed``, ``failed``, ``harness_errors``,
        ``scored`` (passed + failed) and ``pass_rate_pct`` (over ``scored``).
    """
    passed = failed = harness = 0
    for row in rows:
        outcome = result_outcome(row)
        if outcome == SimOutcome.PASS.value:
            passed += 1
        elif outcome == SimOutcome.HARNESS_ERROR.value:
            harness += 1
        else:
            failed += 1
    scored = passed + failed
    return {
        "total": scored + harness,
        "passed": passed,
        "failed": failed,
        "harness_errors": harness,
        "scored": scored,
        "pass_rate_pct": round(100 * passed / scored, 2) if scored else 0.0,
    }


def case_reliability(
    rows: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregates repeated runs of each case into reliability metrics.

    Harness-error runs are counted in ``harness_errors`` but excluded from
    ``k``, so a simulator outage cannot turn a stable case into a flaky one.

    Returns:
        One dict per case (in first-seen order) with ``name``, ``runs``,
        ``k``, ``passes``, ``harness_errors``, ``pass_rate``, ``pass_hat_k``,
        ``pass_at_k`` and ``label``.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_name.setdefault(str(row.get("name", "")), []).append(row)

    out = []
    for name, case_rows in by_name.items():
        outcomes = [result_outcome(r) for r in case_rows]
        harness = outcomes.count(SimOutcome.HARNESS_ERROR.value)
        passes = outcomes.count(SimOutcome.PASS.value)
        k = len(outcomes) - harness
        if k == 0:
            label = ReliabilityLabel.INCONCLUSIVE
        elif passes == k:
            label = ReliabilityLabel.STABLE_PASS
        elif passes == 0:
            label = ReliabilityLabel.STABLE_FAIL
        else:
            label = ReliabilityLabel.FLAKY
        out.append(
            {
                "name": name,
                "runs": len(outcomes),
                "k": k,
                "passes": passes,
                "harness_errors": harness,
                "pass_rate": round(passes / k, 4) if k else None,
                "pass_hat_k": (passes == k) if k else None,
                "pass_at_k": (passes > 0) if k else None,
                "label": label.value,
            }
        )
    return out


def reliability_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Suite-level reliability: per-case metrics plus their aggregate.

    ``pass_hat_k`` / ``pass_at_k`` are the fraction of conclusive cases
    (at least one scored run) that pass in all / any of their runs.
    """
    rows = list(rows)
    cases = case_reliability(rows)
    conclusive = [c for c in cases if c["k"] > 0]
    n = len(conclusive)
    labels = {label.value: 0 for label in ReliabilityLabel}
    for c in cases:
        labels[c["label"]] += 1
    return {
        "cases": len(cases),
        "conclusive_cases": n,
        "max_runs": max((c["runs"] for c in cases), default=0),
        "pass_hat_k": round(sum(c["pass_hat_k"] for c in conclusive) / n, 4)
        if n
        else None,
        "pass_at_k": round(sum(c["pass_at_k"] for c in conclusive) / n, 4)
        if n
        else None,
        **labels,
        "per_case": cases,
    }


def harness_error_brief(row: dict[str, Any], width: int = 90) -> str:
    """One-line description of a run's harness error."""
    err = row.get("harness_error")
    if not isinstance(err, dict):
        return ""
    text = (
        f"{err.get('error_type', 'Error')} at turn {err.get('turn', '?')}"
        f": {err.get('message', '')}"
    )
    if err.get("elapsed_s") is not None:
        text += f" ({err['elapsed_s']}s)"
    return text if len(text) <= width else text[: width - 3] + "..."


def format_simulation_summary(rows: Iterable[dict[str, Any]]) -> str:
    """Plain-text summary printed after a simulation run.

    Always includes the outcome counts; adds the list of harness errors
    when there are any, and a reliability table when a case ran more
    than once.
    """
    rows = list(rows)
    if not rows:
        return ""
    s = summarize_outcomes(rows)
    rate = (
        f"pass rate {s['pass_rate_pct']:.1f}% "
        f"({s['passed']}/{s['scored']} scored runs)"
        if s["scored"]
        else "pass rate n/a (no scored runs)"
    )
    lines = [
        "--- Simulation Summary ---",
        f"{s['passed']} passed, {s['failed']} failed, "
        f"{s['harness_errors']} harness errors (inconclusive) | {rate}",
    ]
    harness_rows = [r for r in rows if is_harness_error(r)]
    if harness_rows:
        lines.append(
            "Harness errors (simulated user stopped; excluded from pass rate):"
        )
        for r in harness_rows:
            lines.append(
                f"  - {r.get('name')} run {r.get('run', '?')}: "
                f"{harness_error_brief(r)}"
            )

    rel = reliability_summary(rows)
    if rel["max_runs"] > 1:
        width = max(len("case"), *(len(c["name"]) for c in rel["per_case"]))
        width = min(width, 48)
        lines.append("")
        lines.append(
            "Reliability across runs (harness errors excluded from k):"
        )
        lines.append(
            f"  {'case':<{width}}  {'pass/k':>6}  {'harness':>7}  "
            f"{'pass^k':>6}  {'pass@k':>6}  label"
        )
        for c in rel["per_case"]:
            hat = _yes_no(c["pass_hat_k"])
            at = _yes_no(c["pass_at_k"])
            lines.append(
                f"  {c['name'][:width]:<{width}}  "
                f"{str(c['passes']) + '/' + str(c['k']):>6}  "
                f"{c['harness_errors']:>7}  {hat:>6}  {at:>6}  {c['label']}"
            )
        n = rel["conclusive_cases"]
        if n:
            lines.append(
                f"  pass^k {_pct(rel['pass_hat_k'])} | "
                f"pass@k {_pct(rel['pass_at_k'])} over {n} cases | "
                f"stable_pass {rel['stable_pass']}, flaky {rel['flaky']}, "
                f"stable_fail {rel['stable_fail']}, "
                f"inconclusive {rel['inconclusive']}"
            )
    return "\n".join(lines)


def _yes_no(value: bool | None) -> str:
    if value is None:
        return "-"
    return "yes" if value else "no"


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"
