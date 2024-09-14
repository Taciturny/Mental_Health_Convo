import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Dict

import streamlit as st
from admin_dashboard import admin_dashboard
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
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")


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

    def initialize_session_state(self):
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
            st.session_state.conversation_history = self.load_conversation_history()
        if "current_view" not in st.session_state:
            st.session_state.current_view = "chat"

    def load_conversation_history(self):
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            SELECT user_input, ai_response FROM conversations
            WHERE user_id = ? ORDER BY timestamp ASC
            """,
            (st.session_state.user_id,),
        )
        history = cursor.fetchall()
        return [
            (
                {"role": "user", "content": user_input}
                if i % 2 == 0
                else {"role": "assistant", "content": ai_response}
            )
            for i, (user_input, ai_response) in enumerate(history)
        ]

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
            return round(max(0, min(1, relevance_score)), 2)
        except ValueError:
            return 0.50

    def run(self):
        self.initialize_session_state()

        # Set the page config (this doesn't change the visible title)
        st.set_page_config(
            page_title="Mental Health App",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Move the admin login to the sidebar
        self.show_sidebar()

        # Navigation based on current view
        if st.session_state.current_view == "admin":
            st.title("Enhanced Admin Dashboard")
            admin_dashboard()
        else:
            st.title("Mental Health Support Chat")
            self.show_chat_interface()

    def show_chat_interface(self):
        # Chat interface
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Type your message here...")

        if user_input:
            st.session_state.conversation_history.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            start_time = time.time()

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

    def show_sidebar(self):

        with st.sidebar:
            st.title("Navigation")
            if st.session_state.current_view == "admin":
                if st.button("Back to Chat"):
                    st.session_state.current_view = "chat"
                    st.rerun()
            else:
                # Admin login option
                st.title("Admin Login")
                password = st.text_input("Enter admin password", type="password")
                if st.button("Login as Admin"):
                    if (
                        password == ADMIN_PASSWORD
                    ):  # Replace with your actual admin password
                        st.session_state.current_view = "admin"
                        st.rerun()
                    else:
                        st.error("Incorrect password")
        st.sidebar.title("💡 Get a Mental Health Tip")
        tip = self.get_mental_health_tip()
        st.sidebar.write(tip)

        st.sidebar.title("📊 Quick Mood Check")
        mood = st.sidebar.slider("How are you feeling today?", 1, 5, 3)
        mood_labels = {1: "Very Low", 2: "Low", 3: "Neutral", 4: "Good", 5: "Excellent"}
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
