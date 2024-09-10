import logging
import sqlite3
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


class SQLiteDatabase:
    def __init__(self, db_path="chatbot.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            self.create_tables()
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")
            raise

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None

    def create_tables(self):
        try:
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    user_input TEXT,
                    response TEXT,
                    response_time FLOAT,
                    search_method TEXT,
                    model_used TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    feedback_type TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """
            )
            self.cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id INTEGER,
                    prompt_tokens INTEGER,
                    response_tokens INTEGER,
                    completion_tokens INTEGER,
                    relevance TEXT,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (id)
                )
            """
            )
            self.conn.commit()
        except sqlite3.Error as e:
            logging.error(f"Error creating tables: {e}")
            raise

    def execute_query(self, query, params=None):
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            self.conn.commit()
            return self.cursor.lastrowid
        except sqlite3.Error as e:
            logging.error(f"Error executing query: {e}")
            self.conn.rollback()
            raise

    def fetch_one(self, query, params=None):
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchone()
        except sqlite3.Error as e:
            logging.error(f"Error fetching one row: {e}")
            raise

    def fetch_all(self, query, params=None):
        try:
            if params:
                self.cursor.execute(query, params)
            else:
                self.cursor.execute(query)
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error fetching all rows: {e}")
            raise

    def create_user(self, username, password_hash):
        query = "INSERT INTO users (username, password_hash) VALUES (?, ?)"
        return self.execute_query(query, (username, password_hash))

    def get_user(self, username):
        query = "SELECT * FROM users WHERE username = ?"
        return self.fetch_one(query, (username,))

    def store_conversation(
        self,
        user_id,
        user_input,
        response,
        response_time,
        search_method,
        model_used,
    ):
        query = """
            INSERT INTO conversations
            (user_id, user_input, response, response_time, search_method, model_used)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        return self.execute_query(
            query,
            (
                user_id,
                user_input,
                response,
                response_time,
                search_method,
                model_used,
            ),
        )

    def update_conversation(
        self, conversation_id, user_input, response, search_method, model_used
    ):
        query = """
            UPDATE conversations
            SET user_input = ?, response = ?, search_method = ?, model_used = ?
            WHERE id = ?
        """
        self.execute_query(
            query,
            (user_input, response, search_method, model_used, conversation_id),
        )

    def store_feedback(self, conversation_id, feedback_type):
        query = "INSERT INTO feedback (conversation_id, feedback_type) VALUES (?, ?)"
        return self.execute_query(query, (conversation_id, feedback_type))

    def get_conversation_history(self, user_id, limit=10):
        query = """
            SELECT * FROM conversations
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """
        return self.fetch_all(query, (user_id, limit))

    def get_feedback_stats(self):
        query = """
            SELECT feedback_type, COUNT(*) as count
            FROM feedback
            GROUP BY feedback_type
        """
        results = self.fetch_all(query)
        feedback_dict = dict(results)

        # Ensure all feedback types are represented, even if count is 0
        all_feedback_types = ["Helpful", "Not Helpful", "Needs Improvement"]
        for feedback_type in all_feedback_types:
            if feedback_type not in feedback_dict:
                feedback_dict[feedback_type] = 0

        return feedback_dict

    def get_user_stats(self, user_id):
        query = """
            SELECT
                COUNT(*) as total_conversations,
                AVG(response_time) as avg_response_time,
                MAX(timestamp) as last_conversation
            FROM conversations
            WHERE user_id = ?
        """
        return self.fetch_one(query, (user_id,))

    def delete_user(self, user_id):
        query = "DELETE FROM users WHERE id = ?"
        self.execute_query(query, (user_id,))

    def delete_conversation(self, conversation_id):
        query = "DELETE FROM conversations WHERE id = ?"
        self.execute_query(query, (conversation_id,))

    def get_popular_search_methods(self, limit=5):
        query = """
            SELECT search_method, COUNT(*) as count
            FROM conversations
            GROUP BY search_method
            ORDER BY count DESC
            LIMIT ?
        """
        return self.fetch_all(query, (limit,))

    def get_model_usage_stats(self):
        query = """
            SELECT model_used, COUNT(*) as count
            FROM conversations
            GROUP BY model_used
        """
        results = self.fetch_all(query)
        return dict(results)

    def get_total_conversations(self):
        query = "SELECT COUNT(*) FROM conversations"
        result = self.fetch_one(query)
        return result[0] if result else 0

    def get_average_response_time(self):
        query = "SELECT AVG(response_time) FROM conversations"
        result = self.fetch_one(query)
        return result[0] if result else 0

    def store_conversation_metrics(
        self,
        conversation_id,
        prompt_tokens,
        response_tokens,
        completion_tokens,
        relevance,
    ):
        query = """
            INSERT INTO conversation_metrics
            (conversation_id, prompt_tokens, response_tokens, completion_tokens, relevance)
            VALUES (?, ?, ?, ?, ?)
        """
        self.execute_query(
            query,
            (
                conversation_id,
                prompt_tokens,
                response_tokens,
                completion_tokens,
                relevance,
            ),
        )

    def get_average_tokens(self):
        query = """
            SELECT
                AVG(CASE WHEN prompt_tokens IS NOT NULL THEN prompt_tokens ELSE 0 END) as avg_prompt_tokens,
                AVG(CASE WHEN response_tokens IS NOT NULL THEN response_tokens ELSE 0 END) as avg_response_tokens,
                AVG(CASE WHEN completion_tokens IS NOT NULL THEN completion_tokens ELSE 0 END) as avg_completion_tokens
            FROM conversation_metrics
        """
        try:
            result = self.fetch_one(query)
            return result if result else (0, 0, 0)
        except sqlite3.OperationalError as e:
            logging.error(f"Error getting average tokens: {e}")
            return (0, 0, 0)  # Return default values if the query fails

    def get_relevance_stats(self):
        query = """
            SELECT relevance, COUNT(*) as count
            FROM conversation_metrics
            GROUP BY relevance
        """
        results = self.fetch_all(query)
        return dict(results)

    def update_conversation_relevance(
        self, conversation_id: str, relevance: str
    ):
        query = """
            UPDATE conversation_metrics
            SET relevance = ?
            WHERE conversation_id = ?
        """
        self.execute_query(query, (relevance, conversation_id))
