import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Dict

import plotly.graph_objects as go
import streamlit as st
from cohere_model import CohereModel
from database import SQLiteDB
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


class MentalHealthApp:
    def __init__(self):
        self.llm_model = CohereModel(COHERE_API_KEY)
        self.search_engine = SearchEngine(COLLECTION_NAME_CLOUD)
        self.db = SQLiteDB("mental_health_app.db")
        self.greetings = [
            "hello",
            "hi",
            "hey",
            "greetings",
            "good morning",
            "good afternoon",
            "good evening",
            "howdy",
            "what's up",
            "yo",
            "hiya",
        ]
        self.greeting_responses = [
            "Hello! How can I assist you with your mental health today?",
            "Hi there! Is there anything specific you'd like to talk about regarding your mental well-being?",
            "Hey! I'm here to support you. What's on your mind?",
            "Greetings! How are you feeling today?",
            "Good to see you! What would you like to discuss about mental health?",
            "Welcome! I'm here to listen and help. What brings you here today?",
            "Hello! How can I support your mental health journey today?",
            "Hi! I'm here to chat about anything related to mental well-being. What's on your mind?",
        ]

        if "user_id" not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        if "feedback" not in st.session_state:
            st.session_state.feedback = None
        if "show_feedback" not in st.session_state:
            st.session_state.show_feedback = False
        if "current_response" not in st.session_state:
            st.session_state.current_response = ""
        if "conversation_history" not in st.session_state:
            st.session_state.conversation_history = []

    def get_greeting_response(self):
        return random.choice(self.greeting_responses)

    def calculate_metrics(
        self,
        prompt: str,
        response: str,
        start_time: float,
        end_time: float,
        relevance: float,
    ) -> Dict:
        prompt_tokens = len(prompt.split())
        response_tokens = len(response.split())
        return {
            "response": response,
            "response_time": end_time - start_time,
            "prompt_tokens": prompt_tokens,
            "response_tokens": response_tokens,
            "completion_tokens": prompt_tokens + response_tokens,
            "relevance": relevance,
        }

    def check_relevance(self, user_input: str, response: str) -> float:
        prompt = f"""
        As an expert in mental health conversations, evaluate the relevance of the AI's response to the user's input.
        Consider the following criteria:
        1. Does the response directly address the user's question or concern?
        2. Is the information provided accurate and helpful in a mental health context?
        3. Does the response show empathy and understanding of the user's situation?
        4. Is the language used appropriate for a mental health support conversation?

        User input: {user_input}
        AI response: {response}

        On a scale of 0 to 1, how relevant is the AI's response to the user's input, considering the above criteria?
        0 means completely irrelevant, and 1 means perfectly relevant.
        Provide only a number as the answer, rounded to two decimal places.
        """
        relevance_check = self.llm_model.generate_response(
            prompt=prompt,
            max_tokens=1,
            temperature=0.0,
            k=0,
            stop_sequences=[],
            return_likelihoods="NONE",
        )
        try:
            relevance_score = float(relevance_check.strip())
            return round(
                max(0, min(1, relevance_score)), 2
            )  # Ensure the score is between 0 and 1, rounded to 2 decimal places
        except ValueError:
            return 0.50  # Default to neutral if parsing fails

    def visualize_metrics(self):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT response_time, prompt_tokens, response_tokens, completion_tokens, relevance
            FROM conversations WHERE user_id = ?
        """,
            (st.session_state.user_id,),
        )
        results = cursor.fetchall()

        if not results:
            st.write("No conversation data available yet.")
            return

        # Unpack the results
        (
            response_times,
            prompt_tokens,
            response_tokens,
            completion_tokens,
            relevance_scores,
        ) = zip(*results)

        # Textual summary
        st.subheader("Conversation Summary")
        total_conversations = len(results)
        avg_response_time = (
            sum(response_times) / total_conversations if total_conversations > 0 else 0
        )
        avg_relevance = (
            sum(relevance_scores) / total_conversations
            if total_conversations > 0
            else 0
        )

        st.write(f"Total conversations: {total_conversations}")
        st.write(f"Average response time: {avg_response_time:.2f} seconds")
        st.write(f"Average relevance score: {avg_relevance:.2f}")

        # Line chart for response times
        fig_response_time = go.Figure()
        fig_response_time.add_trace(
            go.Scatter(y=response_times, mode="lines+markers", name="Response Time")
        )
        fig_response_time.update_layout(
            title="Response Time Trend",
            xaxis_title="Conversation Number",
            yaxis_title="Time (s)",
        )
        st.plotly_chart(fig_response_time)

        # Bar chart for token counts
        fig_tokens = go.Figure()
        fig_tokens.add_trace(
            go.Bar(
                x=["Prompt", "Response", "Completion"],
                y=[sum(prompt_tokens), sum(response_tokens), sum(completion_tokens)],
                name="Token Counts",
            )
        )
        fig_tokens.update_layout(
            title="Total Token Usage", xaxis_title="Token Type", yaxis_title="Count"
        )
        st.plotly_chart(fig_tokens)

        # Line chart for relevance scores
        fig_relevance = go.Figure()
        fig_relevance.add_trace(
            go.Scatter(y=relevance_scores, mode="lines+markers", name="Relevance Score")
        )
        fig_relevance.update_layout(
            title="Relevance Score Trend",
            xaxis_title="Conversation Number",
            yaxis_title="Relevance (0-1)",
        )
        st.plotly_chart(fig_relevance)

        # Pie chart for feedback distribution
        cursor.execute(
            """
            SELECT feedback, COUNT(*) as count FROM feedback WHERE user_id = ? GROUP BY feedback
        """,
            (st.session_state.user_id,),
        )
        feedback_data = cursor.fetchall()

        if feedback_data:
            fig_feedback = go.Figure(
                data=[
                    go.Pie(
                        labels=[row[0] for row in feedback_data],
                        values=[row[1] for row in feedback_data],
                    )
                ]
            )
            fig_feedback.update_layout(title="Feedback Distribution")
            st.plotly_chart(fig_feedback)
        else:
            st.write("No feedback data available yet.")

    def run(self):
        st.title("Mental Health Support Chat")

        self.show_sidebar()

        # Radio button for page selection
        page = st.radio("Navigation", ("Chat", "Metrics"))

        if page == "Chat":
            # Display conversation history
            for message in st.session_state.conversation_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Input box at the bottom
            user_input = st.chat_input("Type your message here...")

            if user_input:
                # Add user message to chat history
                st.session_state.conversation_history.append(
                    {"role": "user", "content": user_input}
                )
                with st.chat_message("user"):
                    st.markdown(user_input)

                start_time = time.time()

                # Check for greetings
                if any(greeting in user_input.lower() for greeting in self.greetings):
                    response = self.get_greeting_response()
                    is_greeting = True
                else:
                    is_greeting = False
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

                    if max_score < 0.3:
                        response = (
                            "I'm not confident I have accurate information to answer your question. "
                            "Could you please rephrase your question or ask about a different mental health topic? "
                            "I'm here to provide reliable information and support related to mental health."
                        )
                    else:
                        prompt = f"Context: {context}\n\nUser: {user_input}\nAssistant:"
                        response = self.llm_model.generate_response(prompt=prompt)

                end_time = time.time()

                # Add AI response to chat history
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": response}
                )
                with st.chat_message("assistant"):
                    st.markdown(response)

                if not is_greeting:
                    relevance = self.check_relevance(user_input, response)
                    metrics = self.calculate_metrics(
                        user_input, response, start_time, end_time, relevance
                    )

                    self.db.save_conversation(
                        st.session_state.user_id,
                        st.session_state.conversation_id,
                        user_input,
                        response,
                        metrics,
                    )

                st.session_state.current_response = response
                st.session_state.show_feedback = not is_greeting

            # Feedback section
            if st.session_state.show_feedback:
                feedback = st.radio(
                    "Was this response helpful?",
                    ("Yes", "No", "Not selected"),
                    index=2,
                    key="feedback_radio",
                )

                if feedback != "Not selected":
                    st.session_state.feedback = feedback

                if st.button("Submit Feedback"):
                    if st.session_state.feedback:
                        self.db.save_feedback(
                            st.session_state.user_id,
                            st.session_state.conversation_id,
                            st.session_state.feedback,
                        )

                        st.success("Thank you for your feedback!")
                        st.session_state.show_feedback = False
                        st.session_state.feedback = None
                    else:
                        st.warning("Please select a feedback option before submitting.")

        elif page == "Metrics":
            self.visualize_metrics()

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


if __name__ == "__main__":
    app = MentalHealthApp()
    app.run()
