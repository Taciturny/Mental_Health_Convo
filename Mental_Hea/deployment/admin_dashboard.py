import sqlite3

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from wordcloud import WordCloud


def admin_dashboard():
    st.title("Enhanced Admin Dashboard")

    # Connect to the database
    conn = sqlite3.connect("mental_health_app.db")

    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        total_conversations = pd.read_sql_query(
            "SELECT COUNT(*) as count FROM conversations", conn
        ).iloc[0]["count"]
        st.metric("Total Conversations", total_conversations)
    with col2:
        total_users = pd.read_sql_query(
            "SELECT COUNT(DISTINCT user_id) as count FROM conversations", conn
        ).iloc[0]["count"]
        st.metric("Total Users", total_users)
    with col3:
        avg_response_time = pd.read_sql_query(
            "SELECT AVG(response_time) as avg_time FROM conversations", conn
        ).iloc[0]["avg_time"]
        st.metric("Average Response Time", f"{avg_response_time:.2f} seconds")

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

    # Daily conversation count (Line Chart)
    st.subheader("Daily Conversation Count")
    daily_count = pd.read_sql_query(
        "SELECT DATE(timestamp) as date, COUNT(*) as count FROM conversations GROUP BY DATE(timestamp)",
        conn,
    )
    fig_daily = px.line(
        daily_count, x="date", y="count", title="Daily Conversation Count"
    )
    st.plotly_chart(fig_daily)

    # Response Time Distribution (Histogram)
    st.subheader("Response Time Distribution")
    response_times = pd.read_sql_query("SELECT response_time FROM conversations", conn)
    fig_response_time = px.histogram(
        response_times,
        x="response_time",
        nbins=20,
        title="Distribution of Response Times",
    )
    fig_response_time.update_layout(
        xaxis_title="Response Time (seconds)", yaxis_title="Frequency"
    )
    st.plotly_chart(fig_response_time)

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
        relevance_data, x="timestamp", y="relevance", title="Relevance Score Over Time"
    )
    fig_relevance.update_layout(xaxis_title="Timestamp", yaxis_title="Relevance Score")
    st.plotly_chart(fig_relevance)

    # Token Usage (Stacked Area Chart)
    st.subheader("Token Usage Over Time")
    token_usage = pd.read_sql_query(
        "SELECT timestamp, prompt_tokens, response_tokens FROM conversations ORDER BY timestamp",
        conn,
    )
    token_usage["timestamp"] = pd.to_datetime(token_usage["timestamp"])
    token_usage = token_usage.set_index("timestamp").resample("D").sum().reset_index()
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
        title="Daily Token Usage", xaxis_title="Date", yaxis_title="Number of Tokens"
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

    # Add a button to return to the chat interface
    if st.button("Back to Chat", key="admin_back_to_chat"):
        st.session_state.current_view = "chat"
        st.rerun()
