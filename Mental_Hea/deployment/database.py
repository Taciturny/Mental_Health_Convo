import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


class SQLiteDB:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                conversation_id TEXT,
                user_input TEXT,
                ai_response TEXT,
                response_time FLOAT,
                prompt_tokens INTEGER,
                response_tokens INTEGER,
                completion_tokens INTEGER,
                relevance FLOAT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                conversation_id TEXT,
                feedback TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.conn.commit()

    def save_conversation(
        self, user_id, conversation_id, user_input, ai_response, metrics
    ):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO conversations (
                user_id, conversation_id, user_input, ai_response,
                response_time, prompt_tokens, response_tokens,
                completion_tokens, relevance
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                user_id,
                conversation_id,
                user_input,
                ai_response,
                metrics["response_time"],
                metrics["prompt_tokens"],
                metrics["response_tokens"],
                metrics["completion_tokens"],
                metrics["relevance"],
            ),
        )
        self.conn.commit()

    def save_feedback(self, user_id, conversation_id, feedback):
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO feedback (user_id, conversation_id, feedback)
            VALUES (?, ?, ?)
        """,
            (user_id, conversation_id, feedback),
        )
        self.conn.commit()
