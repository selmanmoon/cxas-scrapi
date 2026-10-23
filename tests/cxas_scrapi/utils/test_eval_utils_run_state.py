"""Unit regression tests for evaluation run state polling in EvalUtils."""

import enum
import logging
from typing import Any
from unittest.mock import MagicMock, patch

from cxas_scrapi.utils.eval_utils import EvalUtils

APP_NAME = "projects/mock-project/locations/mock-location/apps/mock-app"
RUN_NAME = f"{APP_NAME}/evaluationRuns/run-123"


class _MockEvaluationRunState(enum.IntEnum):
    UNSPECIFIED = 0
    RUNNING = 1
    COMPLETED = 2
    ERROR = 3


def _make_eval_utils() -> EvalUtils:
    """Create an EvalUtils instance with mocked service clients."""
    with (
        patch("cxas_scrapi.core.evaluations.EvaluationServiceClient"),
        patch("cxas_scrapi.core.tools.Tools.get_tools_map", return_value={}),
        patch("cxas_scrapi.core.agents.Agents.get_agents_map", return_value={}),
    ):
        return EvalUtils(app_name=APP_NAME, creds=MagicMock())


def test_wait_for_run_handles_raw_int_unknown_state_5() -> None:
    """Test 1: Raw integer state=5 (unknown enum value) finishes cleanly."""
    utils = _make_eval_utils()
    expected_results = [{"name": f"{APP_NAME}/evaluations/e1/results/r1"}]

    run_status = MagicMock()
    run_status.state = 5
    utils.eval_client.get_evaluation_run = MagicMock(return_value=run_status)
    utils.eval_client.list_evaluation_results_by_run = MagicMock(
        return_value=expected_results
    )

    with patch("cxas_scrapi.utils.eval_utils.time.sleep") as mock_sleep:
        results = utils.wait_for_run_and_get_results(RUN_NAME)

    assert results == expected_results
    utils.eval_client.get_evaluation_run.assert_called_once_with(RUN_NAME)
    utils.eval_client.list_evaluation_results_by_run.assert_called_once_with(
        RUN_NAME
    )
    mock_sleep.assert_not_called()


def test_wait_for_run_handles_raw_int_completed_state_2() -> None:
    """Test 2a: Raw integer state=2 (COMPLETED) terminates polling cleanly."""
    utils = _make_eval_utils()
    expected_results = [{"name": f"{APP_NAME}/evaluations/e1/results/r2"}]

    run_status = MagicMock()
    run_status.state = 2
    utils.eval_client.get_evaluation_run = MagicMock(return_value=run_status)
    utils.eval_client.list_evaluation_results_by_run = MagicMock(
        return_value=expected_results
    )

    with patch("cxas_scrapi.utils.eval_utils.time.sleep") as mock_sleep:
        results = utils.wait_for_run_and_get_results(RUN_NAME)

    assert results == expected_results
    utils.eval_client.get_evaluation_run.assert_called_once_with(RUN_NAME)
    utils.eval_client.list_evaluation_results_by_run.assert_called_once_with(
        RUN_NAME
    )
    mock_sleep.assert_not_called()


def test_wait_for_run_handles_raw_int_error_state_3(
    caplog: Any,
) -> None:
    """Test 2b: Raw integer state=3 (ERROR) logs warning and terminates."""
    utils = _make_eval_utils()
    expected_results = [{"name": f"{APP_NAME}/evaluations/e1/results/r3"}]

    run_status = MagicMock()
    run_status.state = 3
    utils.eval_client.get_evaluation_run = MagicMock(return_value=run_status)
    utils.eval_client.list_evaluation_results_by_run = MagicMock(
        return_value=expected_results
    )

    with (
        patch("cxas_scrapi.utils.eval_utils.time.sleep") as mock_sleep,
        caplog.at_level(logging.WARNING),
    ):
        results = utils.wait_for_run_and_get_results(RUN_NAME)

    assert results == expected_results
    assert "finished with error state" in caplog.text
    mock_sleep.assert_not_called()


def test_wait_for_run_handles_standard_enum_completed() -> None:
    """Test 3a: Standard Enum with .name == 'COMPLETED' succeeds."""
    utils = _make_eval_utils()
    expected_results = [{"name": f"{APP_NAME}/evaluations/e1/results/r4"}]

    run_status = MagicMock()
    run_status.state = _MockEvaluationRunState.COMPLETED
    utils.eval_client.get_evaluation_run = MagicMock(return_value=run_status)
    utils.eval_client.list_evaluation_results_by_run = MagicMock(
        return_value=expected_results
    )

    with patch("cxas_scrapi.utils.eval_utils.time.sleep") as mock_sleep:
        results = utils.wait_for_run_and_get_results(RUN_NAME)

    assert results == expected_results
    utils.eval_client.get_evaluation_run.assert_called_once_with(RUN_NAME)
    utils.eval_client.list_evaluation_results_by_run.assert_called_once_with(
        RUN_NAME
    )
    mock_sleep.assert_not_called()


def test_wait_for_run_handles_standard_enum_error() -> None:
    """Test 3b: Standard Enum with .name == 'ERROR' succeeds."""
    utils = _make_eval_utils()
    expected_results = [{"name": f"{APP_NAME}/evaluations/e1/results/r4b"}]

    run_status = MagicMock()
    run_status.state = _MockEvaluationRunState.ERROR
    utils.eval_client.get_evaluation_run = MagicMock(return_value=run_status)
    utils.eval_client.list_evaluation_results_by_run = MagicMock(
        return_value=expected_results
    )

    with patch("cxas_scrapi.utils.eval_utils.time.sleep") as mock_sleep:
        results = utils.wait_for_run_and_get_results(RUN_NAME)

    assert results == expected_results
    utils.eval_client.get_evaluation_run.assert_called_once_with(RUN_NAME)
    utils.eval_client.list_evaluation_results_by_run.assert_called_once_with(
        RUN_NAME
    )
    mock_sleep.assert_not_called()


def test_wait_for_run_polls_through_pending_int_states_to_terminal() -> None:
    """Test 4a: Pending raw int states (0=UNSPECIFIED, 1=RUNNING) keep polling."""
    utils = _make_eval_utils()
    expected_results = [{"name": f"{APP_NAME}/evaluations/e1/results/r5"}]

    s0 = MagicMock()
    s0.state = 0
    s1 = MagicMock()
    s1.state = 1
    s5 = MagicMock()
    s5.state = 5

    utils.eval_client.get_evaluation_run = MagicMock(side_effect=[s0, s1, s5])
    utils.eval_client.list_evaluation_results_by_run = MagicMock(
        return_value=expected_results
    )

    with patch("cxas_scrapi.utils.eval_utils.time.sleep") as mock_sleep:
        results = utils.wait_for_run_and_get_results(
            RUN_NAME, timeout_seconds=300
        )

    assert results == expected_results
    assert utils.eval_client.get_evaluation_run.call_count == 3
    assert mock_sleep.call_count == 2


def test_wait_for_run_pending_int_state_times_out() -> None:
    """Test 4b: Pending raw int state (1=RUNNING) raises TimeoutError."""
    utils = _make_eval_utils()

    s1 = MagicMock()
    s1.state = 1
    utils.eval_client.get_evaluation_run = MagicMock(return_value=s1)

    timed_out = False
    with (
        patch("cxas_scrapi.utils.eval_utils.time.sleep"),
        patch(
            "cxas_scrapi.utils.eval_utils.time.time",
            side_effect=[0.0, 15.0],
        ),
    ):
        try:
            utils.wait_for_run_and_get_results(RUN_NAME, timeout_seconds=10)
        except TimeoutError as exc:
            timed_out = "timed out" in str(exc)

    assert timed_out
