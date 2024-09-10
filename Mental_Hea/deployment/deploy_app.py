import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from cohere_model import CohereModel
from database import SQLiteDatabase
from dotenv import load_dotenv
from search_engine import SearchEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

QDRANT_URL = "https://e932e81a-113e-440f-96c0-c17b530bfe79.europe-west3-0.gcp.cloud.qdrant.io:6333/dashboard"
COLLECTION_NAME_CLOUD = "mental_health_collection"

# Load environment variables
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")

st.set_page_config(
    page_title="Mental Health Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)


class MentalHealthChatbot:
    def __init__(self):
        self.search_engine = SearchEngine(COLLECTION_NAME_CLOUD)
        self.llm_model = CohereModel(COHERE_API_KEY)
        self.database = SQLiteDatabase()
        self.database.connect()

    def run(self):
        try:
            st.sidebar.title("Navigation")
            page = st.sidebar.radio("Go to", ["Chat", "Metrics"])

            if page == "Chat":
                self.run_chat()
            elif page == "Metrics":
                self.display_metrics_page()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in run method: {str(e)}", exc_info=True)

    def run_chat(self):
        try:
            st.title("Mental Health Chatbot")
            logger.info("Starting run method")

            self.initialize_session_state()
            self.display_chat_history()
            self.handle_user_input()
            self.show_sidebar()

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Error in run method: {str(e)}", exc_info=True)

    def initialize_session_state(self):
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello there! How can I help you today?",
                }
            ]
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())

    def get_db_connection(self):
        return SQLiteDatabase()

    def display_chat_history(self):
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    def handle_user_input(self):
        user_input = st.chat_input(
            "Enter your message:", key="user_chat_input"
        )
        if user_input:
            preprocessed_input = self.preprocess_input(user_input)
            simple_response, end_conversation = self.handle_simple_inputs(
                preprocessed_input
            )

            self.add_message_to_chat("user", user_input)

            if simple_response:
                self.add_message_to_chat("assistant", simple_response)
                if end_conversation:
                    st.stop()
            else:
                (
                    response,
                    response_time,
                    prompt_tokens,
                    response_tokens,
                    completion_tokens,
                    relevance,
                ) = self.generate_response(user_input)
                self.add_message_to_chat("assistant", response)
                self.update_conversation(
                    user_input,
                    response,
                    response_time,
                    prompt_tokens,
                    response_tokens,
                    completion_tokens,
                    relevance,
                )

    def add_message_to_chat(self, role: str, content: str):
        st.session_state.messages.append({"role": role, "content": content})
        with st.chat_message(role):
            st.write(content)

    def generate_response(
        self, user_input: str
    ) -> Tuple[str, float, int, int, int, str]:
        try:
            start_time = time.time()

            search_results = self.search_engine.search_dense(user_input)
            context = ""
            max_score = 0

            if (
                search_results
                and hasattr(search_results, "points")
                and search_results.points
            ):
                for point in search_results.points:
                    payload = point.payload
                    context += f"{payload.get('text', '')}\n"
                    max_score = max(max_score, point.score)
                    logger.info(
                        f"Answer: {payload.get('answer', 'N/A')}, Score: {point.score}"
                    )

            if max_score < 0.3:
                response = (
                    "I'm not confident I have accurate information to answer your question. "
                    "Could you please rephrase your question or ask about a different mental health topic? "
                    "I'm here to provide reliable information and support related to mental health."
                )
                relevance = "PARTLY_RELEVANT"
            else:
                prompt = (
                    f"Context: {context}\n\nUser: {user_input}\nAssistant:"
                )
                response = self.llm_model.generate_response(prompt)
                relevance = "RELEVANT"

            end_time = time.time()
            response_time = end_time - start_time

            prompt_tokens = len(prompt.split()) if "prompt" in locals() else 0
            response_tokens = len(response.split())
            completion_tokens = prompt_tokens + response_tokens

            return (
                response,
                response_time,
                prompt_tokens,
                response_tokens,
                completion_tokens,
                relevance,
            )
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return (
                "I'm sorry, I encountered an error while processing your request. Could you please try again?",
                0.0,
                0,
                0,
                0,
                "NON_RELEVANT",
            )

    def preprocess_input(self, user_input: str) -> str:
        return user_input.lower().strip()

    def handle_simple_inputs(
        self, preprocessed_input: str
    ) -> Tuple[str, bool]:
        simple_responses = {
            "thank you": (
                "You're welcome! I'm glad I could help. Is there anything else you'd like to discuss?",
                True,
            ),
            "thanks": (
                "You're welcome! Is there anything else on your mind?",
                True,
            ),
            "bye": (
                "Take care! Remember, it's okay to reach out whenever you need support.",
                True,
            ),
            "hello": ("Hello! How can I assist you today?", False),
            "hi": ("Hi there! What would you like to talk about?", False),
        }
        return simple_responses.get(preprocessed_input, (None, False))

    def update_conversation(
        self,
        user_input,
        response,
        response_time,
        prompt_tokens,
        response_tokens,
        completion_tokens,
        relevance,
    ):
        try:
            with self.get_db_connection() as db:
                if st.session_state.conversation_id:
                    db.store_conversation(
                        user_id=st.session_state.conversation_id,
                        user_input=user_input,
                        response=response,
                        response_time=response_time,
                        search_method="dense",
                        model_used="cohere",
                    )
                    db.store_conversation_metrics(
                        st.session_state.conversation_id,
                        prompt_tokens,
                        response_tokens,
                        completion_tokens,
                        relevance,
                    )
            self.display_feedback_buttons(st.session_state.conversation_id)
        except Exception as e:
            logger.error(f"Error updating conversation: {str(e)}")
            st.error(
                "An error occurred while saving the conversation. Your chat may not be persisted."
            )

    def display_feedback_buttons(self, conversation_id: str):
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👍 Helpful", key=f"helpful_{conversation_id}"):
                self.submit_feedback(conversation_id, "Helpful")
        with col2:
            if st.button(
                "👎 Not Helpful", key=f"not_helpful_{conversation_id}"
            ):
                self.submit_feedback(conversation_id, "Not Helpful")
        with col3:
            if st.button(
                "🤔 Needs Improvement",
                key=f"needs_improvement_{conversation_id}",
            ):
                self.submit_feedback(conversation_id, "Needs Improvement")

    def submit_feedback(self, conversation_id: str, feedback_type: str):
        try:
            with self.get_db_connection() as db:
                db.store_feedback(conversation_id, feedback_type)
                # Update the relevance based on feedback
                relevance = self.get_relevance_from_feedback(feedback_type)
                db.update_conversation_relevance(conversation_id, relevance)
            st.success("Thank you for your feedback!")
            logger.info(
                f"Feedback submitted: {feedback_type} for conversation ID: {conversation_id}"
            )
            # Force a rerun of the app to update the metrics
            time.sleep(
                0.5
            )  # Add a small delay to ensure the database update is complete
            st.experimental_rerun()
        except Exception as e:
            logger.error(f"Error submitting feedback: {str(e)}")
            st.error(
                "An error occurred while submitting your feedback. Please try again."
            )

    def get_relevance_from_feedback(self, feedback_type: str) -> str:
        if feedback_type == "Helpful":
            return "RELEVANT"
        elif feedback_type == "Needs Improvement":
            return "PARTLY_RELEVANT"
        elif feedback_type == "Not Helpful":
            return "NON_RELEVANT"
        else:
            return "UNKNOWN"

    def display_metrics_page(self):
        st.title("Chatbot Metrics Dashboard")

        try:
            # Fetch data
            total_conversations = self.database.get_total_conversations()
            avg_response_time = self.database.get_average_response_time()
            feedback_stats = self.database.get_feedback_stats()
            popular_methods = self.database.get_popular_search_methods(limit=3)
            model_stats = self.database.get_model_usage_stats()
            avg_tokens = self.database.get_average_tokens()
            relevance_stats = self.database.get_relevance_stats()

            # Display summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Conversations", total_conversations)
            with col2:
                st.metric("Avg Response Time", f"{avg_response_time:.2f} s")
            with col3:
                st.metric("Total Feedback", sum(feedback_stats.values()))

            # Create separate visualizations

            # Feedback Distribution
            feedback_df = pd.DataFrame(
                list(feedback_stats.items()), columns=["Feedback", "Count"]
            )
            fig_feedback = px.pie(
                feedback_df,
                values="Count",
                names="Feedback",
                title="Feedback Distribution",
            )
            st.plotly_chart(fig_feedback)

            # Search Methods
            methods_df = pd.DataFrame(
                popular_methods, columns=["Method", "Count"]
            )
            fig_methods = px.bar(
                methods_df,
                x="Method",
                y="Count",
                title="Popular Search Methods",
            )
            st.plotly_chart(fig_methods)

            # Model Usage
            model_df = pd.DataFrame(
                list(model_stats.items()), columns=["Model", "Count"]
            )
            fig_model = px.bar(
                model_df, x="Model", y="Count", title="Model Usage"
            )
            st.plotly_chart(fig_model)

            # Response Relevance
            relevance_df = pd.DataFrame(
                list(relevance_stats.items()), columns=["Relevance", "Count"]
            )
            fig_relevance = px.pie(
                relevance_df,
                values="Count",
                names="Relevance",
                title="Response Relevance",
            )
            st.plotly_chart(fig_relevance)

            # Token Usage
            token_df = pd.DataFrame(
                {
                    "Type": ["Prompt", "Response", "Completion"],
                    "Tokens": avg_tokens,
                }
            )
            fig_tokens = px.bar(
                token_df, x="Type", y="Tokens", title="Average Token Usage"
            )
            st.plotly_chart(fig_tokens)

            # Additional textual summaries
            st.subheader("Detailed Metrics")
            st.write(
                f"- Average response time: {avg_response_time:.2f} seconds"
            )
            st.write("- Feedback received:")
            for feedback_type, count in feedback_stats.items():
                st.write(f"  • {feedback_type}: {count}")
            st.write("- Top search methods:")
            for method, count in popular_methods:
                st.write(f"  • {method}: {count} times")
            st.write("- Model usage:")
            for model, count in model_stats.items():
                st.write(f"  • {model}: {count} times")
            st.write("- Average token usage:")
            st.write(f"  • Prompt tokens: {avg_tokens[0]:.2f}")
            st.write(f"  • Response tokens: {avg_tokens[1]:.2f}")
            st.write(f"  • Completion tokens: {avg_tokens[2]:.2f}")
            st.write("- Response relevance:")
            for relevance, count in relevance_stats.items():
                st.write(f"  • {relevance}: {count}")

        except Exception as e:
            st.error(f"An error occurred while fetching metrics: {str(e)}")
            logger.error(
                f"Error in display_metrics_page: {str(e)}", exc_info=True
            )

    def show_sidebar(self):
        st.sidebar.title("💡 Get a Mental Health Tip")
        tip = self.get_mental_health_tip()
        st.sidebar.write(tip)

        st.sidebar.title("📊 Quick Mood Check")
        mood = st.sidebar.slider("How are you feeling today?", 1, 5, 3)
        mood_labels = {
            1: "Very Low",
            2: "Low",
            3: "Neutral",
            4: "Good",
            5: "Excellent",
        }
        st.sidebar.write(f"You're feeling: {mood_labels[mood]}")

        st.sidebar.title("Need immediate help?")
        st.sidebar.write(
            "If you're in crisis, please call your local emergency number or a mental health helpline."
        )

        st.sidebar.title("🚀 Helpful Resources")
        st.sidebar.markdown(
            """
        - [Meditation App](https://www.headspace.com/)
        - [Crisis Helpline](https://www.crisistextline.org/)
        - [Self-Care Ideas](https://www.verywellmind.com/self-care-strategies-overall-stress-reduction-3144729)
        """
        )

        st.sidebar.title("About")
        st.sidebar.info(
            "This chat application uses advanced AI to provide mental health support. While it can offer helpful advice, it's not a substitute for professional medical help."
        )

    def get_mental_health_tip(self) -> str:
        tips = [
            "Practice mindfulness for 5 minutes each day to reduce stress and anxiety.",
            "Reach out to a friend or family member today - social connections are crucial for mental health.",
            "Try a new hobby or activity to stimulate your mind and boost your mood.",
            "Get at least 7-8 hours of sleep tonight for better emotional regulation tomorrow.",
            "Take a short walk outside to boost your mood and get some vitamin D.",
        ]
        return random.choice(tips)


def main():
    chatbot = MentalHealthChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()

    # def update_conversation(self, user_input, response, response_time, prompt_tokens, response_tokens, completion_tokens, relevance):
    #     try:
    #         with SQLiteDatabase() as db:
    #             if st.session_state.conversation_id:
    #                 conversation_id = db.store_conversation(
    #                     user_id=st.session_state.conversation_id,
    #                     user_input=user_input,
    #                     response=response,
    #                     response_time=response_time
    #                 )
    #                 if conversation_id is None:
    #                     raise Exception("Failed to store conversation")

    #                 db.store_conversation_metrics(
    #                     conversation_id,
    #                     prompt_tokens,
    #                     response_tokens,
    #                     completion_tokens,
    #                     relevance
    #                 )
    #             self.display_feedback_buttons(st.session_state.conversation_id)
    #     except Exception as e:
    #         logger.error(f"Error updating conversation: {str(e)}", exc_info=True)
    #         st.error(f"An error occurred while saving the conversation: {str(e)}")
