import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import streamlit as st
import pandas as pd
import uuid
import time
from typing import Dict, Any
from database_monitor import Database
from src.core.llm_model import EnsembleModel
from src.core.utils import initialize_components, index_data
# from src.core.search_engine import SearchEngine
import logging


st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠", layout="wide")
class MentalHealthMonitoringApp:
    def __init__(self):
        # Configuration
        self.COLLECTION_NAME = "mental_health_qa"
        self.VECTOR_SIZE = 384
        self.HOST = "localhost"
        self.PORT = 6333
        self.DATA_PATH = "./data/preprocessed_data.parquet"

        self.search_engine, self.data_loader, self.embeddings_model = initialize_components(
            self.COLLECTION_NAME, self.VECTOR_SIZE, self.HOST, self.PORT, self.DATA_PATH
        )
        self.ensemble_model = EnsembleModel.get_instance()
        self.db = Database()
        self._ensure_data_indexed()
        self.logger = logging.getLogger(__name__)

    def _ensure_data_indexed(self):
        if self.search_engine.qdrant_manager.collection_is_empty(self.search_engine.collection_name):
            with st.spinner("Initializing data index. This may take a few moments..."):
                index_data(self.search_engine, self.data_loader, self.embeddings_model)

    def run(self):
        st.title("Mental Health Chatbot Monitoring Dashboard")
        self.display_sidebar()
        self.handle_user_query()
        self.display_metrics()

    def display_sidebar(self):
        st.sidebar.header("Configuration")
        self.model_type = st.sidebar.selectbox("Select Model", ["RAG", "gpt2", "dialogpt", "distilgpt2", "None"], key="model_type")
        self.search_type = st.sidebar.selectbox("Select Search Type", ["Hybrid", "Vector", "Keyword", "None"], key="search_type")

    def handle_user_query(self):
        query = st.text_input("Enter your query:", key="user_query")
        if st.button("Submit Query"):
            if query:
                response_data = self.process_query(query)
                self.display_response(response_data)
                conversation_id = self.store_conversation(query, response_data)
                self.handle_feedback(conversation_id, response_data)

    def process_query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        if self.model_type == "RAG" or self.search_type != "None":
            response_data = self.search_engine.rag(query)
        elif self.model_type in ["GPT2", "DialoGPT", "DistilGPT2"]:
            response_data = self.generate_response_with_ensemble_model(query)
        else:
            response_data = self.search_engine.fallback_to_hybrid(query)

        response_data["response"] = self.search_engine.post_process_response(
            response=response_data.get("response", ""),
            query=query
        )
        self.last_query = query
        response_data["response"] = self.remove_prompt_and_question(response_data["response"])
        response_data["response_time"] = time.time() - start_time

        if not response_data.get("response"):
            self.logger.error(f"Empty response generated for query: {query}")
            response_data["response"] = "I apologize, but I couldn't generate a proper response. Could you please rephrase your question?"

        return response_data

    def generate_response_with_ensemble_model(self, query: str) -> Dict[str, Any]:
        # Generate response using the selected model from the ensemble
        response = self.ensemble_model.generate_text(query, model_name=self.model_type)
        return {
            "response": response[0] if response else "Unable to generate a response.",
            "confidence_score": 0.5  # You may want to implement a method to calculate this
        }

    def remove_prompt_and_question(self, response: str) -> str:
        # List of common artifacts to remove
        artifacts = [
            "Response:", "Q:", "q:", "User Query:", "A:", "a:", ":",
            "Context:", "Question:", "Answer:", "Human:", "Assistant:"
        ]
        
        # Remove artifacts
        for artifact in artifacts:
            response = response.replace(artifact, "")
        
        # Split the response into lines
        lines = response.split('\n')
        
        # Remove lines that contain the original query or end with a question mark
        cleaned_lines = [line for line in lines if not (line.strip().endswith('?') or 'query:' in line.lower())]
        
        # Join the remaining lines
        cleaned_response = ' '.join(cleaned_lines).strip()
        
        # If the response still starts with the query, remove it
        if cleaned_response.lower().startswith(self.last_query.lower()):
            cleaned_response = cleaned_response[len(self.last_query):].strip()
        
        return cleaned_response
    
    def store_conversation(self, query: str, response_data: Dict[str, Any]) -> str:
        try:
            conversation_id = self.db.store_conversation(
                user_id=str(uuid.uuid4()),
                query=query,
                response=response_data["response"],
                search_type=self.search_type,
                model_type=self.model_type,
                confidence_score=response_data.get("confidence_score", 0.0),
                response_time=response_data.get("response_time", 0.0)
            )
            self.logger.info(f"Conversation stored with ID: {conversation_id}")
            return conversation_id
        except Exception as e:
            self.logger.error(f"Error storing conversation: {str(e)}")
            return None
        
    def display_response(self, response_data: Dict[str, Any]):
        if self.model_type == "None":
            st.subheader("Top 5 Search Results")
            for idx, result in enumerate(response_data.get("search_results", [])[:5], 1):
                st.write(f"{idx}. {result['question']}")
                st.write(f"   Answer: {result['answer']}")
                st.write(f"   Score: {result['score']:.2f}")
                st.write("---")
        else:
            st.subheader("Model Response")
            st.write(response_data["response"])

    def handle_feedback(self, conversation_id: str, response_data: Dict[str, Any]):
        st.write("---")
        st.subheader("Feedback")
        feedback = st.radio(
            "How helpful was this response?",
            ["Very Helpful", "Somewhat Helpful", "Neutral", "Somewhat Unhelpful", "Very Unhelpful"]
        )
        feedback_text = st.text_area("Additional comments (optional):")
        if st.button("Submit Feedback"):
            try:
                self.db.store_feedback(conversation_id, feedback, feedback_text)
                st.success("Thank you for your feedback!")
                self.logger.info(f"Feedback stored for conversation ID: {conversation_id}")
            except Exception as e:
                self.logger.error(f"Error storing feedback: {str(e)}")
                st.error("An error occurred while submitting your feedback. Please try again.")
            st.experimental_rerun()

    def display_metrics(self):
        try:
            st.write("---")
            st.header("Performance Metrics")
            st.write("---")

            col1, col2 = st.columns(2)
            with col1:
                self.display_conversation_stats()
                self.display_model_performance()
                self.display_query_complexity()

            with col2:
                self.display_user_engagement()
                self.display_response_time()
                self.display_error_rate()

            st.write("---")
            st.subheader("Additional Metrics")
            col3, col4 = st.columns(2)
            with col3:
                self.display_top_queries()
                self.display_model_confidence()

            with col4:
                self.display_active_users()
                self.display_avg_conversation_length()

            engagement_rate = self.db.get_user_engagement_rate()
            st.metric("User Engagement Rate", f"{engagement_rate:.2f}%" if engagement_rate is not None else "N/A")

            avg_response_time = self.db.get_average_response_time()
            st.metric("Average Response Time", f"{avg_response_time:.2f} seconds" if avg_response_time is not None else "N/A")

            error_rate = self.db.get_error_rate()
            st.metric("Error Rate", f"{error_rate:.2f}%" if error_rate is not None else "N/A")
            if not self.db.has_sufficient_data():
                st.warning("Not enough data to calculate accurate metrics. Please generate more conversations.")
        except Exception as e:
            self.logger.error(f"Error displaying metrics: {str(e)}")
            st.error("An error occurred while fetching metrics. Please try refreshing the page.")


    def display_conversation_stats(self):
        st.subheader("Conversation Statistics")
        stats = self.db.get_conversation_stats()
        st.metric("Total Conversations", stats["total_conversations"])

    def display_model_performance(self):
        st.subheader("Model Performance")
        performance_stats = self.db.get_model_performance_stats()
        if performance_stats:
            df = pd.DataFrame(performance_stats)
            st.dataframe(df)
        else:
            st.write("No performance data available yet.")

    def display_query_complexity(self):
        st.subheader("Query Complexity")
        avg_length, min_length, max_length = self.db.get_query_complexity_stats()
        
        st.metric("Average Query Length", self.safe_format(avg_length))
        st.metric("Min Query Length", self.safe_format(min_length))
        st.metric("Max Query Length", self.safe_format(max_length))


    def display_user_engagement(self):
        st.subheader("User Engagement")
        engagement_rate = self.db.get_user_engagement_rate()
        st.metric("User Engagement Rate", self.safe_format(engagement_rate, "{:.2f}%", "N/A"))

    def display_response_time(self):
        st.subheader("Response Time")
        avg_response_time = self.db.get_average_response_time()
        st.metric("Average Response Time", f"{avg_response_time:.2f} seconds" if avg_response_time is not None else "N/A")

    def safe_format(self, value, format_str="{:.2f}", na_text="N/A"):
        """Safely format the value or return a default text if None."""
        return format_str.format(value) if value is not None else na_text

    def display_error_rate(self):
        st.subheader("Error Rate")
        error_rate = self.db.get_error_rate()
        st.metric("Error Rate", self.safe_format(error_rate, "{:.2f}%"))

    def display_top_queries(self):
        st.subheader("Top Queries")
        top_queries = self.db.get_top_queries(n=5)
        df = pd.DataFrame(top_queries, columns=["Query", "Frequency"])
        st.dataframe(df)

    def display_model_confidence(self):
        st.subheader("Model Confidence")
        avg_confidence, min_confidence, max_confidence = self.db.get_model_confidence_stats()
        st.metric("Average Confidence", self.safe_format(avg_confidence, "{:.2f}"))
        st.metric("Min Confidence", self.safe_format(min_confidence, "{:.2f}"))
        st.metric("Max Confidence", self.safe_format(max_confidence, "{:.2f}"))

    def display_active_users(self):
        st.subheader("Active Users")
        active_users = self.db.get_active_users(days=7)
        st.metric("Active Users (Last 7 Days)", active_users)

    def display_avg_conversation_length(self):
        st.subheader("Average Conversation Length")
        avg_length = self.db.get_avg_conversation_length()
        st.metric("Average Exchanges per Conversation", self.safe_format(avg_length, "{:.2f}"))

    def has_sufficient_data(self):
        result = self.execute_query("SELECT COUNT(*) FROM conversations")
        conversation_count = result[0][0] if result else 0
        return conversation_count >= 10  # Adjust this threshold as needed


if __name__ == "__main__":
    app = MentalHealthMonitoringApp()
    app.run()
