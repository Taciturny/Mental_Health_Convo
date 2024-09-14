import logging
import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud


def admin_dashboard():
    try:
        # Connect to the database
        conn = sqlite3.connect("mental_health_app.db")

        # Metrics
        # Date range selector
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date")
        with col2:
            end_date = st.date_input("End Date")

        # Metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            total_conversations = pd.read_sql_query(
                f"SELECT COUNT(*) as count FROM conversations WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'",
                conn,
            ).iloc[0]["count"]
            st.metric("Total Conversations", total_conversations)
        with col2:
            total_users = pd.read_sql_query(
                f"SELECT COUNT(DISTINCT user_id) as count FROM conversations WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'",
                conn,
            ).iloc[0]["count"]
            st.metric("Total Users", total_users)
        with col3:
            avg_response_time = pd.read_sql_query(
                f"SELECT AVG(response_time) as avg_time FROM conversations WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}'",
                conn,
            ).iloc[0]["avg_time"]
            if avg_response_time is not None:
                st.metric("Average Response Time", f"{avg_response_time:.2f} seconds")
            else:
                st.metric("Average Response Time", "No data")

        # Conversation history
        st.subheader("Recent Conversations")
        conversations = pd.read_sql_query(
            "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT 10", conn
        )
        st.dataframe(conversations)

        # Feedback distribution (Pie Chart)
        st.subheader("Feedback Distribution")
        feedback = pd.read_sql_query(
            "SELECT feedback, COUNT(*) as count FROM feedback GROUP BY feedback", conn
        )
        fig_feedback = px.pie(
            feedback, values="count", names="feedback", title="Feedback Distribution"
        )
        st.plotly_chart(fig_feedback)

        # Daily conversation count (Line Chart with 7-day moving average)
        st.subheader("Daily Conversation Count")
        daily_count = pd.read_sql_query(
            f"SELECT DATE(timestamp) as date, COUNT(*) as count FROM conversations WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}' GROUP BY DATE(timestamp)",
            conn,
        )
        daily_count["date"] = pd.to_datetime(daily_count["date"])
        daily_count = daily_count.set_index("date")
        daily_count["7_day_ma"] = daily_count["count"].rolling(window=7).mean()

        fig_daily = go.Figure()
        fig_daily.add_trace(
            go.Scatter(
                x=daily_count.index,
                y=daily_count["count"],
                mode="lines",
                name="Daily Count",
            )
        )
        fig_daily.add_trace(
            go.Scatter(
                x=daily_count.index,
                y=daily_count["7_day_ma"],
                mode="lines",
                name="7-day Moving Average",
            )
        )
        fig_daily.update_layout(
            title="Daily Conversation Count", xaxis_title="Date", yaxis_title="Count"
        )
        st.plotly_chart(fig_daily)

        # User Growth (Cumulative Line Chart)
        st.subheader("User Growth Over Time")
        user_growth = pd.read_sql_query(
            f"SELECT DATE(timestamp) as date, COUNT(DISTINCT user_id) as new_users FROM conversations WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}' GROUP BY DATE(timestamp)",
            conn,
        )
        user_growth["date"] = pd.to_datetime(user_growth["date"])
        user_growth = user_growth.set_index("date")
        user_growth["cumulative_users"] = user_growth["new_users"].cumsum()

        fig_growth = px.line(
            user_growth,
            x=user_growth.index,
            y="cumulative_users",
            title="Cumulative User Growth",
        )
        fig_growth.update_layout(xaxis_title="Date", yaxis_title="Total Users")
        st.plotly_chart(fig_growth)

        # User Engagement Distribution (Histogram)
        st.subheader("User Engagement Distribution")
        user_engagement = pd.read_sql_query(
            f"SELECT user_id, COUNT(*) as conversation_count FROM conversations WHERE DATE(timestamp) BETWEEN '{start_date}' AND '{end_date}' GROUP BY user_id",
            conn,
        )
        fig_engagement_dist = px.histogram(
            user_engagement,
            x="conversation_count",
            nbins=20,
            title="Distribution of User Engagement",
        )
        fig_engagement_dist.update_layout(
            xaxis_title="Number of Conversations", yaxis_title="Number of Users"
        )
        st.plotly_chart(fig_engagement_dist)

        # User Engagement (Bar Chart)
        st.subheader("User Engagement")
        user_engagement = pd.read_sql_query(
            "SELECT user_id, COUNT(*) as conversation_count FROM conversations GROUP BY user_id ORDER BY conversation_count DESC LIMIT 10",
            conn,
        )
        fig_engagement = px.bar(
            user_engagement,
            x="user_id",
            y="conversation_count",
            title="Top 10 Users by Conversation Count",
        )
        fig_engagement.update_layout(
            xaxis_title="User ID", yaxis_title="Number of Conversations"
        )
        st.plotly_chart(fig_engagement)

        # Relevance Score Over Time (Scatter Plot)
        st.subheader("Relevance Score Over Time")
        relevance_data = pd.read_sql_query(
            "SELECT timestamp, relevance FROM conversations ORDER BY timestamp", conn
        )
        fig_relevance = px.scatter(
            relevance_data,
            x="timestamp",
            y="relevance",
            title="Relevance Score Over Time",
        )
        fig_relevance.update_layout(
            xaxis_title="Timestamp", yaxis_title="Relevance Score"
        )
        st.plotly_chart(fig_relevance)

        # Token Usage (Stacked Area Chart)
        st.subheader("Token Usage Over Time")
        token_usage = pd.read_sql_query(
            "SELECT timestamp, prompt_tokens, response_tokens FROM conversations ORDER BY timestamp",
            conn,
        )
        token_usage["timestamp"] = pd.to_datetime(token_usage["timestamp"])
        token_usage = (
            token_usage.set_index("timestamp").resample("D").sum().reset_index()
        )
        fig_tokens = go.Figure()
        fig_tokens.add_trace(
            go.Scatter(
                x=token_usage["timestamp"],
                y=token_usage["prompt_tokens"],
                name="Prompt Tokens",
                mode="none",
                fill="tonexty",
            )
        )
        fig_tokens.add_trace(
            go.Scatter(
                x=token_usage["timestamp"],
                y=token_usage["response_tokens"],
                name="Response Tokens",
                mode="none",
                fill="tozeroy",
            )
        )
        fig_tokens.update_layout(
            title="Daily Token Usage",
            xaxis_title="Date",
            yaxis_title="Number of Tokens",
        )
        st.plotly_chart(fig_tokens)

        # Word Cloud of User Inputs
        st.subheader("Word Cloud of User Inputs")
        user_inputs = pd.read_sql_query("SELECT user_input FROM conversations", conn)
        text = " ".join(user_inputs["user_input"])
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate(
            text
        )
        fig_wordcloud, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig_wordcloud)

        # Heatmap of Conversation Activity
        st.subheader("Heatmap of Conversation Activity")
        activity_data = pd.read_sql_query(
            "SELECT strftime('%w', timestamp) as day_of_week, strftime('%H', timestamp) as hour, COUNT(*) as count FROM conversations GROUP BY day_of_week, hour",
            conn,
        )
        activity_data["day_of_week"] = pd.to_numeric(activity_data["day_of_week"])
        activity_data["hour"] = pd.to_numeric(activity_data["hour"])

        # Create a complete dataframe with all days and hours
        all_days = pd.DataFrame({"day_of_week": range(7)})
        all_hours = pd.DataFrame({"hour": range(24)})
        complete_data = all_days.merge(all_hours, how="cross")

        # Merge with actual data and fill missing values with 0
        activity_pivot = complete_data.merge(
            activity_data, on=["day_of_week", "hour"], how="left"
        ).fillna(0)
        activity_pivot = activity_pivot.pivot(
            index="day_of_week", columns="hour", values="count"
        )

        # Create heatmap
        fig_heatmap = px.imshow(
            activity_pivot,
            labels=dict(x="Hour of Day", y="Day of Week", color="Conversation Count"),
            x=[str(i).zfill(2) for i in range(24)],
            y=["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"],
            title="Heatmap of Conversation Activity",
        )
        fig_heatmap.update_layout(height=500)  # Adjust height for better visibility
        st.plotly_chart(fig_heatmap)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logging.error(f"Admin dashboard error: {str(e)}")

    # Add a button to return to the chat interface
    if st.button("Back to Chat", key="admin_back_to_chat"):
        st.session_state.current_view = "chat"
        st.rerun()
