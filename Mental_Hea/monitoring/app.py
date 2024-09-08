import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import time
import streamlit as st
import uuid
import plotly.graph_objects as go
from typing import Dict, Any
from .database_monitor import Database
from src.core.utils import is_relevant_query
from src.core.search_engine import SearchEngine



st.set_page_config(page_title="Mental Health Chatbot", page_icon="🧠", layout="wide")

class MentalHealthMonitoringApp:
    def __init__(self):
        self.search_engine = SearchEngine(collection_name="mental_health_qa")
        self.db = Database()
        self.initialize_session_state()

    def initialize_session_state(self):
        if 'user_id' not in st.session_state:
            st.session_state.user_id = str(uuid.uuid4())
        if 'conversation_id' not in st.session_state:
            st.session_state.conversation_id = None

    def run(self):
        st.title("Mental Health Monitoring Dashboard")

        col1, col2 = st.columns([3, 1])

        with col1:
            self.main_interface()

        with col2:
            self.display_metrics()

    def main_interface(self):
        st.header("Query Interface")

        query = st.text_input("Enter your query:")
        model_type = st.selectbox("Select model type:", ["RAG", "GPT-2", "DialoGPT", "DISTILGPT2"])

        if model_type == "RAG":
            search_type = "Hybrid"
            st.info("RAG automatically uses Hybrid search.")
        else:
            search_type = st.selectbox("Select search type:", ["Hybrid", "Dense (Vector)", "Late (Keyword)"])

        if st.button("Generate Response"):
            if query:
                start_time = time.time()
                response = self.generate_response(query, search_type, model_type)
                end_time = time.time()
                response_time = end_time - start_time

                st.session_state.last_response = response['response']
                st.session_state.conversation_id = response['conversation_id']
                st.session_state.show_feedback = True

                self.store_conversation(query, response, search_type, model_type, response_time)

        if 'last_response' in st.session_state:
            st.subheader("Generated Response:")
            st.write(st.session_state.last_response)

        if 'show_feedback' in st.session_state and st.session_state.show_feedback:
            self.collect_feedback()


    def generate_response(self, query: str, search_type: str, model_type: str) -> Dict[str, Any]:
        search_type_mapping = {
            "Hybrid": "hybrid",
            "Dense (Vector)": "dense",
            "Late (Keyword)": "late"
        }
        search_type_for_engine = search_type_mapping.get(search_type, "hybrid")

        model_type_mapping = {
            "RAG": "ensemble",
            "GPT-2": "gpt2",
            "DialoGPT": "dialogpt",
            "DISTILGPT2": "distilgpt2"
        }
        model_type_for_engine = model_type_mapping.get(model_type, "ensemble")

        result = self.search_engine.rag(query, search_type_for_engine, model_type_for_engine)      
        # Extract only the response part
        response = self.post_process_response(result['response'], query)
        
        return {
            'response': response,
            'confidence_score': result['confidence_score'],
            'conversation_id': str(uuid.uuid4())
        }

    def post_process_response(self, response: str, query: str) -> str:
        # Remove the query and any common prefixes from the response
        response = response.replace(query, "").strip()
        prefixes_to_remove = ["Response:", "A:", "Answer:", "AI:"]
        for prefix in prefixes_to_remove:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()
        
        # Ensure the response starts with a capital letter and ends with a period
        if response:
            response = response[0].upper() + response[1:]
            if not response.endswith('.'):
                response += '.'
        
        return response


    def store_conversation(self, query: str, response: Dict[str, Any], search_type: str, model_type: str, response_time: float):
        conversation_id = self.db.store_conversation(
            st.session_state.user_id,
            query,
            response['response'],
            search_type,
            model_type,
            response['confidence_score'],
            response_time
        )
        st.session_state.conversation_id = conversation_id


    def collect_feedback(self):
        st.subheader("Provide Feedback")
        feedback = st.selectbox(
            "How helpful was this response?",
            ["Very Helpful", "Somewhat Helpful", "Neutral", "Somewhat Unhelpful", "Very Unhelpful"]
        )
        if st.button("Submit Feedback"):
            if st.session_state.conversation_id:
                if not hasattr(st.session_state, 'feedback_submitted') or not st.session_state.feedback_submitted:
                    try:
                        self.db.store_feedback(st.session_state.conversation_id, feedback)
                        self.display_feedback_message(feedback)
                        st.session_state.feedback_submitted = True
                    except ValueError as e:
                        st.error(f"An error occurred while submitting feedback: {str(e)}")
                    except Exception as e:
                        st.error(f"An error occurred while submitting feedback: {str(e)}")
                else:
                    st.warning("Feedback has already been submitted for this conversation.")
            else:
                st.error("Unable to submit feedback. No conversation ID found.")


    def display_feedback_message(self, feedback: str):
        if feedback in ["Very Helpful", "Somewhat Helpful"]:
            st.success("Thank you for your positive feedback! We're glad the response was helpful.")
        elif feedback == "Neutral":
            st.info("Thank you for your feedback. We'll continue to work on improving our responses.")
        else:
            st.warning("We apologize that the response wasn't helpful. We'll use your feedback to improve our system.")

    @st.cache_data(ttl=60)  # Cache for 60 seconds
    def get_fresh_metrics(_self):
        return {
            'total_conversations': _self.db.get_conversation_stats()['total_conversations'],
            'avg_response_time': _self.db.get_average_response_time(),
            'engagement_rate': _self.db.get_user_engagement_rate(),
            'error_rate': _self.db.get_error_rate(),
            'model_performance': _self.db.get_model_performance_stats(),
            'search_type_stats': _self.db.get_search_type_stats()
        }

    def display_metrics(self):
        st.sidebar.header("Performance Metrics")

        metrics = self.get_fresh_metrics()

        st.sidebar.metric("Total Conversations", metrics['total_conversations'])
        if metrics['avg_response_time']:
            st.sidebar.metric("Avg Response Time", f"{metrics['avg_response_time']:.2f}s")
        if metrics['engagement_rate']:
            st.sidebar.metric("User Engagement Rate", f"{metrics['engagement_rate']:.2f}%")
        if metrics['error_rate']:
            st.sidebar.metric("Error Rate", f"{metrics['error_rate']:.2f}%")

        # Model Performance
        st.sidebar.subheader("Model Performance")
        for model in metrics['model_performance']:
            with st.sidebar.expander(f"{model['model_type']} Stats"):
                st.write(f"Positive Feedback Rate: {model['positive_feedback_rate']}%")
                st.write(f"Avg Response Length: {model['avg_response_length']}")
                st.write(f"Usage Count: {model['usage_count']}")
                st.write(f"Avg Confidence Score: {model['avg_confidence_score']}")

        # Search Type Distribution
        if metrics['search_type_stats']:
            fig = go.Figure(data=[go.Pie(labels=list(metrics['search_type_stats'].keys()), values=list(metrics['search_type_stats'].values()))])
            fig.update_layout(title_text="Search Type Distribution")
            st.sidebar.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    app = MentalHealthMonitoringApp()
    app.run()
