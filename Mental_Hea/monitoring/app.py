import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.core.search_engine import SearchEngine

from .database_monitor import Database

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠", layout="wide")


class MentalHealthMonitoringApp:
    def __init__(self):
        self.search_engine = SearchEngine(collection_name="mental_health_qa")
        self.db = Database()
        self.initialize_session_state()

    def initialize_session_state(self):
        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = None

    def run(self):
        st.title("Mental Health Monitoring Dashboard")

        col1, col2 = st.columns([3, 1])

        with col1:
            self.main_interface()

        with col2:
            self.display_metrics()
            self.display_metrics_visualization()

    def main_interface(self):
        st.header("Query Interface")

        query = st.text_input("Enter your query:")
        model_type = st.selectbox(
            "Select model type:", ["RAG", "GPT-2", "DialoGPT", "DISTILGPT2"]
        )

        if model_type == "RAG":
            search_type = "Hybrid"
            st.info("RAG automatically uses Hybrid search.")
        else:
            search_type = st.selectbox(
                "Select search type:",
                ["Hybrid", "Dense (Vector)", "Late (Keyword)"],
            )

        if st.button("Generate Response"):
            if query:
                start_time = time.time()
                response = self.generate_response(query, search_type, model_type)
                end_time = time.time()
                response_time = end_time - start_time

                st.session_state.last_response = response["response"]
                st.session_state.conversation_id = response["conversation_id"]
                st.session_state.show_feedback = True

                self.store_conversation(
                    query, response, search_type, model_type, response_time
                )

        if "last_response" in st.session_state:
            st.subheader("Generated Response:")
            st.write(st.session_state.last_response)

        if "show_feedback" in st.session_state and st.session_state.show_feedback:
            self.collect_feedback()

    def generate_response(
        self, query: str, search_type: str, model_type: str
    ) -> Dict[str, Any]:
        search_type_mapping = {
            "Hybrid": "hybrid",
            "Dense (Vector)": "dense",
            "Late (Keyword)": "late",
        }
        search_type_for_engine = search_type_mapping.get(search_type, "hybrid")

        model_type_mapping = {
            "RAG": "ensemble",
            "GPT-2": "gpt2",
            "DialoGPT": "dialogpt",
            "DISTILGPT2": "distilgpt2",
        }
        model_type_for_engine = model_type_mapping.get(model_type, "ensemble")

        result = self.search_engine.rag(
            query, search_type_for_engine, model_type_for_engine
        )
        # Extract only the response part
        response = self.post_process_response(result["response"], query)

        return {
            "response": response,
            "confidence_score": result["confidence_score"],
            "conversation_id": str(uuid.uuid4()),
        }

    def post_process_response(self, response: str, query: str) -> str:
        # Remove the query and any common prefixes from the response
        response = response.replace(query, "").strip()
        prefixes_to_remove = ["Response:", "A:", "Answer:", "AI:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix) :].strip()

        # Ensure the response starts with a capital letter and ends with a period
        if response:
            response = response[0].upper() + response[1:]
            if not response.endswith("."):
                response += "."

        return response

    def store_conversation(
        self,
        query: str,
        response: Dict[str, Any],
        search_type: str,
        model_type: str,
        response_time: float,
    ):
        conversation_id = self.db.store_conversation(
            st.session_state.user_id,
            query,
            response["response"],
            search_type,
            model_type,
            response["confidence_score"],
            response_time,
        )
        st.session_state.conversation_id = conversation_id

    def collect_feedback(self):
        st.subheader("Provide Feedback")
        feedback = st.selectbox(
            "How helpful was this response?",
            [
                "Very Helpful",
                "Somewhat Helpful",
                "Neutral",
                "Somewhat Unhelpful",
                "Very Unhelpful",
            ],
        )
        if st.button("Submit Feedback"):
            if st.session_state.conversation_id:
                if (
                    not hasattr(st.session_state, "feedback_submitted")
                    or not st.session_state.feedback_submitted
                ):
                    try:
                        self.db.store_feedback(
                            st.session_state.conversation_id, feedback
                        )
                        self.display_feedback_message(feedback)
                        st.session_state.feedback_submitted = True
                    except ValueError as e:
                        st.error(
                            f"An error occurred while submitting feedback: {str(e)}"
                        )
                    except Exception as e:
                        st.error(
                            f"An error occurred while submitting feedback: {str(e)}"
                        )
                else:
                    st.warning(
                        "Feedback has already been submitted for this conversation."
                    )
            else:
                st.error("Unable to submit feedback. No conversation ID found.")

    def display_feedback_message(self, feedback: str):
        if feedback in ["Very Helpful", "Somewhat Helpful"]:
            st.success(
                "Thank you for your positive feedback! We're glad the response was helpful."
            )
        elif feedback == "Neutral":
            st.info(
                "Thank you for your feedback. We'll continue to work on improving our responses."
            )
        else:
            st.warning(
                "We apologize that the response wasn't helpful. We'll use your feedback to improve our system."
            )

    @st.cache_data(ttl=1)  # Cache for 1 second
    def get_fresh_metrics(_self):
        return {
            "total_conversations": _self.db.get_conversation_stats()[
                "total_conversations"
            ],
            "avg_response_time": _self.db.get_average_response_time(),
            "engagement_rate": _self.db.get_user_engagement_rate(),
            "error_rate": _self.db.get_error_rate(),
            "model_performance": _self.db.get_model_performance_stats(),
            "search_type_stats": _self.db.get_search_type_stats(),
            "daily_conversations": _self.db.get_daily_conversation_count(last_n_days=5),
            "feedback_distribution": _self.db.get_feedback_distribution(),
        }

    def display_metrics(self):
        st.sidebar.header("Performance Metrics")

        metrics = self.get_fresh_metrics()

        st.sidebar.metric("Total Conversations", metrics["total_conversations"])
        if metrics["avg_response_time"]:
            st.sidebar.metric(
                "Avg Response Time", f"{metrics['avg_response_time']:.2f}s"
            )
        if metrics["engagement_rate"]:
            st.sidebar.metric(
                "User Engagement Rate", f"{metrics['engagement_rate']:.2f}%"
            )
        if metrics["error_rate"]:
            st.sidebar.metric("Error Rate", f"{metrics['error_rate']:.2f}%")

        # Model Performance
        st.sidebar.subheader("Model Performance")
        for model in metrics["model_performance"]:
            with st.sidebar.expander(f"{model['model_type']} Stats"):
                st.write(f"Positive Feedback Rate: {model['positive_feedback_rate']}%")
                st.write(f"Avg Response Length: {model['avg_response_length']}")
                st.write(f"Usage Count: {model['usage_count']}")
                st.write(f"Avg Confidence Score: {model['avg_confidence_score']}")

        # Search Type Distribution
        if metrics["search_type_stats"]:
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=list(metrics["search_type_stats"].keys()),
                        values=list(metrics["search_type_stats"].values()),
                    )
                ]
            )
            fig.update_layout(title_text="Search Type Distribution")
            st.sidebar.plotly_chart(fig, use_container_width=True)

    def display_metrics_visualization(self):
        st.sidebar.subheader("Metrics Visualization")

        metrics = self.get_fresh_metrics()

        visualization_options = [
            "Search Type Distribution",
            "Daily Conversation Trend",
            "Feedback Distribution",
            "Model Performance Comparison",
        ]

        selected_visualization = st.sidebar.radio(
            "Select Metric to Visualize:", visualization_options
        )

        if selected_visualization == "Search Type Distribution":
            self.plot_search_type_distribution(metrics["search_type_stats"])
        elif selected_visualization == "Daily Conversation Trend":
            self.plot_daily_conversation_trend(metrics["daily_conversations"])
        elif selected_visualization == "Feedback Distribution":
            self.plot_feedback_distribution(metrics["feedback_distribution"])
        elif selected_visualization == "Model Performance Comparison":
            self.plot_model_performance_comparison(metrics["model_performance"])

    def plot_search_type_distribution(self, search_type_stats):
        fig = go.Figure(
            data=[
                go.Pie(
                    labels=list(search_type_stats.keys()),
                    values=list(search_type_stats.values()),
                )
            ]
        )
        fig.update_layout(title_text="Search Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

    def plot_daily_conversation_trend(self, daily_conversations):
        df = pd.DataFrame(daily_conversations, columns=["date", "count"])
        fig = px.line(df, x="date", y="count", title="Daily Conversation Trend")
        st.plotly_chart(fig, use_container_width=True)

    def plot_feedback_distribution(self, feedback_distribution):
        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(feedback_distribution.keys()),
                    y=list(feedback_distribution.values()),
                )
            ]
        )
        fig.update_layout(
            title_text="Feedback Distribution",
            xaxis_title="Feedback",
            yaxis_title="Count",
        )
        st.plotly_chart(fig, use_container_width=True)

    def plot_model_performance_comparison(self, model_performance):
        models = [model["model_type"] for model in model_performance]
        positive_feedback_rates = [
            model["positive_feedback_rate"] for model in model_performance
        ]
        avg_confidence_scores = [
            model["avg_confidence_score"] for model in model_performance
        ]

        fig = go.Figure(
            data=[
                go.Bar(
                    name="Positive Feedback Rate",
                    x=models,
                    y=positive_feedback_rates,
                ),
                go.Bar(
                    name="Avg Confidence Score",
                    x=models,
                    y=avg_confidence_scores,
                ),
            ]
        )
        fig.update_layout(title_text="Model Performance Comparison", barmode="group")
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    app = MentalHealthMonitoringApp()
    app.run()
