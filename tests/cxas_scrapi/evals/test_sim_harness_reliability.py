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

"""Tests for simulator harness errors and multi-run reliability."""

import json
import os
import typing
from unittest.mock import MagicMock, patch

import pytest

from cxas_scrapi.evals.sim_reliability import (
    SimOutcome,
    case_reliability,
    format_simulation_summary,
    reliability_summary,
    result_outcome,
    summarize_outcomes,
)
from cxas_scrapi.evals.simulation_evals import (
    LLMUserConversation,
    SimulationEvals,
    Step,
    StepProgress,
    StepStatus,
)
from cxas_scrapi.utils.reporting import (
    generate_combined_html_report,
    generate_combined_json_report,
    generate_html_report,
)

_STEP = Step(goal="Book a flight", success_criteria="Flight booked")


def _conv(client: MagicMock) -> LLMUserConversation:
    return LLMUserConversation(
        genai_client=client,
        genai_model="model",
        test_case={"name": "tc", "steps": [_STEP.model_dump()]},
    )


def _row(
    name: str, run: int, outcome: str, **extra: typing.Any
) -> dict[str, typing.Any]:
    row = {
        "name": name,
        "run": run,
        "passed": outcome == "PASS",
        "outcome": outcome,
        "goals": "1/1" if outcome == "PASS" else "0/1",
        "expectations": "0/0",
        "turns": 3,
        "session_id": f"{name}-{run}",
        "step_details": [
            {
                "goal": "g",
                "success_criteria": "c",
                "status": "Completed" if outcome == "PASS" else "Not Started",
                "justification": "",
            }
        ],
        "expectation_details": [],
    }
    if outcome == "HARNESS_ERROR":
        row["harness_error"] = {
            "turn": 4,
            "error_type": "EmptyGeneration",
            "message": "simulated-user model returned no output",
            "elapsed_s": 189.0,
        }
    row.update(extra)
    return row


# --- Harness error detection -------------------------------------------------


def test_generation_returning_none_records_harness_error() -> None:
    client = MagicMock()
    client.generate.return_value = None
    conv = _conv(client)
    conv.next_user_utterance()  # turn 0 is the static welcome event

    utterance, variables = conv.next_user_utterance("Hi, how can I help?")

    assert utterance == ""
    assert variables == {}
    assert conv.harness_error is not None
    assert conv.harness_error["turn"] == 1
    assert conv.harness_error["error_type"] == "EmptyGeneration"
    assert conv.harness_error["elapsed_s"] >= 0


def test_harness_error_reports_underlying_generator_error() -> None:
    client = MagicMock()
    client.generate.return_value = None
    client.last_error = RuntimeError("404 NOT_FOUND. Publisher model missing")
    conv = _conv(client)
    conv.next_user_utterance()

    conv.next_user_utterance("Hello")

    assert conv.harness_error["error_type"] == "RuntimeError"
    assert "404 NOT_FOUND" in conv.harness_error["message"]


def test_generation_raising_records_harness_error() -> None:
    client = MagicMock()
    client.generate.side_effect = RuntimeError("429 RESOURCE_EXHAUSTED")
    conv = _conv(client)
    conv.next_user_utterance()

    utterance, _ = conv.next_user_utterance("Hello")

    assert utterance == ""
    assert conv.harness_error["error_type"] == "RuntimeError"
    assert "RESOURCE_EXHAUSTED" in conv.harness_error["message"]


def test_first_harness_error_is_kept() -> None:
    client = MagicMock()
    client.generate.side_effect = [RuntimeError("first"), None]
    conv = _conv(client)
    conv.next_user_utterance()
    conv.next_user_utterance("a")
    conv.next_user_utterance("b")
    assert conv.harness_error["message"] == "first"


def test_natural_conversation_end_is_not_a_harness_error() -> None:
    client = MagicMock()
    client.generate.return_value = LLMUserConversation.Output(
        next_user_utterance="",
        step_progresses=[StepProgress(step=_STEP, status=StepStatus.COMPLETED)],
    )
    conv = _conv(client)
    conv.next_user_utterance()
    conv.next_user_utterance("Your flight is booked.")
    # All goals are complete now, so the next call ends without the LLM.
    utterance, _ = conv.next_user_utterance("Anything else?")

    assert utterance == ""
    assert conv.harness_error is None


def test_max_turns_end_is_not_a_harness_error() -> None:
    client = MagicMock()
    conv = LLMUserConversation(
        genai_client=client,
        genai_model="model",
        test_case={"name": "tc", "steps": [_STEP.model_dump()]},
        max_turns=1,
    )
    conv.next_user_utterance()
    utterance, _ = conv.next_user_utterance("hi")
    assert utterance == ""
    assert conv.harness_error is None
    client.generate.assert_not_called()


# --- Outcome classification in the job runner --------------------------------


def _evals() -> SimulationEvals:
    with patch("cxas_scrapi.evals.simulation_evals.GeminiGenerate"):  # noqa: SIM117
        with patch("cxas_scrapi.core.apps.AgentServiceClient"):
            return SimulationEvals(app_name="projects/p/locations/l/apps/a")


def _fake_conv(harness_error: dict[str, typing.Any] | None) -> MagicMock:
    conv = MagicMock()
    conv.steps_progress = [
        StepProgress(step=_STEP, status=StepStatus.NOT_STARTED)
    ]
    conv.expectation_results = []
    conv.expectations = []
    conv.audio_expectations = []
    conv.naturalness_result = None
    conv.current_turn = 2
    conv.get_transcript.return_value = "User: event: welcome"
    conv.detailed_trace = []
    conv._detailed_trace = []
    conv.turn_traces = []
    conv._turn_traces = []
    conv.harness_error = harness_error
    return conv


def _run_job(evals: SimulationEvals) -> dict[str, typing.Any]:
    return evals._run_single_simulation_job(
        {"name": "tc", "steps": [_STEP.model_dump()]},
        0,
        1,
        "m",
        "m",
        "text",
        False,
        1,
    )


def test_job_with_harness_error_is_harness_error_outcome(
    capsys: pytest.CaptureFixture[str],
) -> None:
    evals = _evals()
    err = {
        "turn": 1,
        "error_type": "EmptyGeneration",
        "message": "x",
        "elapsed_s": 3.0,
    }
    evals.simulate_conversation = MagicMock(return_value=_fake_conv(err))

    row = _run_job(evals)

    assert row["outcome"] == "HARNESS_ERROR"
    assert row["passed"] is False
    assert row["harness_error"] == err
    assert "HARNESS" in capsys.readouterr().out


def test_job_without_harness_error_keeps_fail_outcome() -> None:
    evals = _evals()
    evals.simulate_conversation = MagicMock(return_value=_fake_conv(None))

    row = _run_job(evals)

    assert row["outcome"] == "FAIL"
    assert row["harness_error"] is None
    assert row["goals"] == "0/1"


def test_harness_error_is_retried_as_inconclusive() -> None:
    reason = SimulationEvals._inconclusive_reason(
        _row("a", 1, "HARNESS_ERROR"), {}
    )
    assert reason == "simulated user failed: EmptyGeneration"
    assert SimulationEvals._inconclusive_reason(_row("a", 1, "FAIL"), {}) is (
        None
    )


def test_run_simulations_prints_summary(
    capsys: pytest.CaptureFixture[str],
) -> None:
    evals = _evals()
    rows = iter(
        [
            _row("a", 1, "PASS"),
            _row("a", 2, "HARNESS_ERROR"),
            _row("b", 1, "FAIL"),
            _row("b", 2, "PASS"),
        ]
    )
    evals._run_single_simulation_job = MagicMock(
        side_effect=lambda *a, **k: next(rows)
    )

    evals.run_simulations([{"name": "a"}, {"name": "b"}], runs=2)

    out = capsys.readouterr().out
    assert "2 passed, 1 failed, 1 harness errors (inconclusive)" in out
    assert "pass rate 66.7%" in out
    assert "flaky" in out


# --- Pass-rate math and reliability ------------------------------------------


def test_result_outcome_backward_compatible() -> None:
    assert result_outcome({"passed": True}) == "PASS"
    assert result_outcome({"passed": False}) == "FAIL"
    assert result_outcome({"passed": False, "error": "boom"}) == "FAIL"
    assert result_outcome({"harness_error": {"turn": 1}}) == "HARNESS_ERROR"
    assert result_outcome({"outcome": SimOutcome.PASS}) == "PASS"


def test_pass_rate_excludes_harness_errors() -> None:
    rows = (
        [_row("a", i, "PASS") for i in range(12)]
        + [_row("b", i, "FAIL") for i in range(3)]
        + [_row("c", i, "HARNESS_ERROR") for i in range(2)]
    )
    s = summarize_outcomes(rows)
    assert s == {
        "total": 17,
        "passed": 12,
        "failed": 3,
        "harness_errors": 2,
        "scored": 15,
        "pass_rate_pct": 80.0,
    }


def test_pass_rate_all_harness_errors_is_zero_not_crash() -> None:
    s = summarize_outcomes([_row("a", 1, "HARNESS_ERROR")])
    assert s["scored"] == 0
    assert s["pass_rate_pct"] == 0.0
    text = format_simulation_summary([_row("a", 1, "HARNESS_ERROR")])
    assert "pass rate n/a (no scored runs)" in text


def test_case_reliability_labels_and_pass_k() -> None:
    rows = [
        _row("stable", 1, "PASS"),
        _row("stable", 2, "PASS"),
        _row("stable", 3, "HARNESS_ERROR"),
        _row("flaky", 1, "PASS"),
        _row("flaky", 2, "FAIL"),
        _row("flaky", 3, "PASS"),
        _row("broken", 1, "FAIL"),
        _row("broken", 2, "FAIL"),
        _row("broken", 3, "FAIL"),
        _row("unknown", 1, "HARNESS_ERROR"),
    ]
    cases = {c["name"]: c for c in case_reliability(rows)}

    assert cases["stable"]["k"] == 2
    assert cases["stable"]["harness_errors"] == 1
    assert cases["stable"]["pass_hat_k"] is True
    assert cases["stable"]["label"] == "stable_pass"

    assert cases["flaky"]["pass_hat_k"] is False
    assert cases["flaky"]["pass_at_k"] is True
    assert cases["flaky"]["pass_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert cases["flaky"]["label"] == "flaky"

    assert cases["broken"]["pass_at_k"] is False
    assert cases["broken"]["label"] == "stable_fail"

    assert cases["unknown"]["k"] == 0
    assert cases["unknown"]["pass_hat_k"] is None
    assert cases["unknown"]["label"] == "inconclusive"

    rel = reliability_summary(rows)
    assert rel["conclusive_cases"] == 3
    assert rel["pass_hat_k"] == pytest.approx(1 / 3, abs=1e-4)
    assert rel["pass_at_k"] == pytest.approx(2 / 3, abs=1e-4)
    assert (rel["stable_pass"], rel["flaky"], rel["stable_fail"]) == (1, 1, 1)
    assert rel["inconclusive"] == 1


def test_format_summary_single_run_has_no_reliability_table() -> None:
    text = format_simulation_summary([_row("a", 1, "PASS")])
    assert "1 passed, 0 failed, 0 harness errors" in text
    assert "Reliability" not in text


def test_format_summary_lists_harness_errors_and_table() -> None:
    rows = [
        _row("a", 1, "PASS"),
        _row("a", 2, "HARNESS_ERROR"),
        _row("b", 1, "PASS"),
        _row("b", 2, "FAIL"),
    ]
    text = format_simulation_summary(rows)
    assert "a run 2: EmptyGeneration at turn 4" in text
    assert "Reliability across runs" in text
    assert "pass^k 50% | pass@k 100% over 2 cases" in text


# --- Report rendering ----------------------------------------------------------


def _report_rows() -> list[dict[str, typing.Any]]:
    return [
        _row("sim_ok", 1, "PASS"),
        _row("sim_ok", 2, "PASS"),
        _row("sim_stall", 1, "PASS"),
        _row("sim_stall", 2, "HARNESS_ERROR"),
    ]


def test_combined_html_report_shows_harness_badge(tmp_path: typing.Any) -> None:
    out = os.path.join(tmp_path, "r.html")
    generate_combined_html_report(sim_results=_report_rows(), output_path=out)
    with open(out) as f:
        html = f.read()

    assert "Harness error" in html
    assert "badge harness" in html
    assert "run-dot h" in html
    assert "EmptyGeneration" in html
    # Pass rate ignores the harness error: 3 of 3 scored runs passed.
    assert "3/3" in html
    assert "1 harness error excluded" in html
    assert "Reliability across runs" in html
    # Harness errors are not agent failures.
    assert "Goal not completed" not in html


def test_combined_json_report_excludes_harness_from_rate(
    tmp_path: typing.Any,
) -> None:
    out = os.path.join(tmp_path, "r.json")
    rows = [*_report_rows(), _row("sim_bad", 1, "FAIL")]
    generate_combined_json_report(sim_results=rows, output_path=out)
    with open(out) as f:
        report = json.load(f)

    sim = report["summary"]["simulation"]
    assert sim["harness_errors"] == 1
    assert sim["failed"] == 1
    assert sim["pass_rate_pct"] == 75.0
    assert report["summary"]["pass_rate_pct"] == 75.0
    rel = report["simulation_reliability"]
    assert {c["name"]: c["label"] for c in rel["per_case"]} == {
        "sim_ok": "stable_pass",
        "sim_stall": "stable_pass",
        "sim_bad": "stable_fail",
    }
    assert report["results"]["simulation"][3]["outcome"] == "HARNESS_ERROR"


def test_legacy_html_report_shows_harness_error(tmp_path: typing.Any) -> None:
    out = os.path.join(tmp_path, "legacy.html")
    generate_html_report(_report_rows(), out, "text", "model")
    with open(out) as f:
        html = f.read()
    assert "100.0%" in html
    assert "1 harness errors (inconclusive)" in html
    assert "HARNESS ERROR" in html
    assert "run-dot h" in html


def test_reports_show_na_when_every_run_is_a_harness_error(
    tmp_path: typing.Any,
) -> None:
    rows = [_row("a", 1, "HARNESS_ERROR"), _row("b", 1, "HARNESS_ERROR")]
    combined = os.path.join(tmp_path, "c.html")
    generate_combined_html_report(sim_results=rows, output_path=combined)
    with open(combined) as f:
        html = f.read()
    assert '<div class="big harness">n/a</div>' in html
    assert '<div class="value harness">0/0</div>' in html
    assert "(0/0 — n/a; 2 harness errors excluded)" in html
    assert "INCONCLUSIVE" in html

    legacy = os.path.join(tmp_path, "l.html")
    generate_html_report(rows, legacy, "text", "model")
    with open(legacy) as f:
        assert '<div class="big harness">n/a</div>' in f.read()
