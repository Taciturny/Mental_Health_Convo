import sys
import uuid
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from monitoring.app import MentalHealthMonitoringApp

# Adjust the path to ensure the monitoring folder is found
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


@pytest.fixture
def mock_search_engine():
    return Mock()


@pytest.fixture
def mock_database():
    return Mock()


@pytest.fixture
def app(mock_search_engine, mock_database):
    with patch("monitoring.app.SearchEngine", return_value=mock_search_engine), patch(
        "monitoring.app.Database", return_value=mock_database
    ):
        return MentalHealthMonitoringApp()


class MockSessionState(dict):
    def __getattr__(self, name):
        if name not in self:
            self[name] = None
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


@pytest.fixture
def mock_streamlit():
    mock_state = MockSessionState()
    mock_state.user_id = "test_user_id"
    mock_state.conversation_id = None
    mock_state.last_response = None
    mock_state.show_feedback = False

    with patch("streamlit.text_input", return_value="How to manage stress?"), patch(
        "streamlit.selectbox", side_effect=["RAG", "Hybrid", "Very Helpful"]
    ), patch("streamlit.button", return_value=True), patch(
        "streamlit.write"
    ) as mock_write, patch(
        "streamlit.success"
    ) as mock_success, patch(
        "streamlit.error"
    ) as mock_error, patch(
        "streamlit.session_state", mock_state
    ):
        yield mock_state, mock_write, mock_success, mock_error


def test_post_process_response(app):
    # Test various scenarios for post-processing
    assert (
        app.post_process_response("Response: This is a test.", "Query")
        == "This is a test."
    )
    assert (
        app.post_process_response("A: lowercase start", "Query") == "Lowercase start."
    )
    assert app.post_process_response("Answer: No period", "Query") == "No period."
    assert (
        app.post_process_response("AI: Multiple sentences. Second one.", "Query")
        == "Multiple sentences. Second one."
    )


def test_multiple_conversations_and_metrics(app, mock_database):
    # Mock the store_conversation method
    mock_database.store_conversation = Mock()

    # Simulate multiple conversations
    for i in range(5):
        app.db.store_conversation(
            str(uuid.uuid4()),
            f"Query {i}",
            f"Response {i}",
            "Hybrid" if i % 2 == 0 else "Dense (Vector)",
            "RAG" if i % 2 == 0 else "GPT-2",
            0.8 + (i * 0.02),
            0.5 + (i * 0.1),
        )

    # Mock the get_fresh_metrics method to return a predefined dictionary
    app.get_fresh_metrics = Mock(
        return_value={
            "total_conversations": 5,
            "avg_response_time": 0.7,
            "search_type_stats": {"Hybrid": 3, "Dense (Vector)": 2},
            "model_performance": {"RAG": 0.85, "GPT-2": 0.82},
        }
    )

    # Check metrics
    metrics = app.get_fresh_metrics()
    assert metrics["total_conversations"] == 5
    assert pytest.approx(metrics["avg_response_time"], 0.1) == 0.7


if __name__ == "__main__":
    pytest.main()
