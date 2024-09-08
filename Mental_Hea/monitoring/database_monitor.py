
import os
import logging
from typing import List, Tuple, Dict, Optional, Any
from psycopg2.pool import SimpleConnectionPool
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.pool = SimpleConnectionPool(
            1, 20,
            host=os.getenv("POSTGRES_HOST", "localhost"),
            database=os.getenv("POSTGRES_DB", "chatbot_monitoring"),
            user=os.getenv("POSTGRES_USER", "tacy"),
            password=os.getenv("POSTGRES_PASSWORD", "1234"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        self.create_tables()

    def get_conn(self):
        return self.pool.getconn()

    def put_conn(self, conn):
        self.pool.putconn(conn)

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Tuple]:
        conn = self.get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()
                if cur.description:
                    return cur.fetchall()
                return []
        except Exception as e:
            logger.error(f"Error executing query: {query}")
            logger.error(f"Error details: {str(e)}")
            conn.rollback()
            raise
        finally:
            self.put_conn(conn)

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        self.execute_query("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";")

        queries = [
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                user_id UUID,
                query TEXT NOT NULL,
                response TEXT NOT NULL,
                search_type TEXT NOT NULL,
                model_type TEXT NOT NULL,
                confidence_score FLOAT,
                response_time FLOAT, 
                query_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                response_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
                feedback TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS search_types (
                type TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS model_types (
                type TEXT PRIMARY KEY,
                count INTEGER DEFAULT 0
            )
            """
        ]

        for query in queries:
            self.execute_query(query)

        logger.info("Database tables created or verified")

    def store_conversation(self, user_id: str, query: str, response: str, search_type: str, model_type: str, confidence_score: float, response_time: float) -> str:
        """Store a conversation and update related statistics."""
        insert_query = """
        INSERT INTO conversations (user_id, query, response, search_type, model_type, confidence_score, response_time) 
        VALUES (%s::uuid, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        result = self.execute_query(insert_query, (user_id, query, response, search_type, model_type, confidence_score, response_time))
        conversation_id = result[0][0]

        self.execute_query(
            "INSERT INTO search_types (type, count) VALUES (%s, 1) ON CONFLICT (type) DO UPDATE SET count = search_types.count + 1",
            (search_type,)
        )
        self.execute_query(
            "INSERT INTO model_types (type, count) VALUES (%s, 1) ON CONFLICT (type) DO UPDATE SET count = model_types.count + 1",
            (model_type,)
        )
        logger.info(f"Stored conversation with ID: {conversation_id}")
        return str(conversation_id)
    
    def store_feedback(self, conversation_id: str, feedback: str):
        """Store feedback for a conversation."""
        try:
            self.execute_query(
                "INSERT INTO feedback (conversation_id, feedback) VALUES (%s::uuid, %s)",
                (conversation_id, feedback)
            )
            logger.info(f"Stored feedback for conversation ID: {conversation_id}")
        except Exception as e:
            logger.error(f"Error storing feedback: {str(e)}")
            # Check if the conversation exists
            conversation_exists = self.execute_query(
                "SELECT EXISTS(SELECT 1 FROM conversations WHERE id = %s::uuid)",
                (conversation_id,)
            )[0][0]
            if not conversation_exists:
                logger.error(f"Conversation with ID {conversation_id} does not exist")
                raise ValueError(f"Conversation with ID {conversation_id} does not exist")
            else:
                raise  # Re-raise the original exception if the conversation exists


    def get_conversation_stats(self) -> Dict[str, int]:
        """Get conversation statistics."""
        result = self.execute_query("SELECT COUNT(*) FROM conversations")
        total_conversations = result[0][0]
        logger.info(f"Retrieved conversation stats: {total_conversations} total conversations")
        return {"total_conversations": total_conversations}

    def get_feedback_stats(self) -> Dict[str, int]:
        """Get feedback statistics."""
        result = self.execute_query("""
            SELECT feedback, COUNT(*) 
            FROM feedback 
            GROUP BY feedback
        """)
        feedback_stats = dict(result)
        logger.info(f"Retrieved feedback stats: {feedback_stats}")
        return feedback_stats

    def get_search_type_stats(self) -> Dict[str, int]:
        """Get search type statistics."""
        result = self.execute_query("SELECT type, count FROM search_types ORDER BY count DESC")
        search_type_stats = dict(result)
        logger.info(f"Retrieved search type stats: {search_type_stats}")
        return search_type_stats

    def get_model_type_stats(self) -> Dict[str, int]:
        """Get model type statistics."""
        result = self.execute_query("SELECT type, count FROM model_types ORDER BY count DESC")
        model_type_stats = dict(result)
        logger.info(f"Retrieved model type stats: {model_type_stats}")
        return model_type_stats
    
    def get_user_engagement_stats(self, days: int = 7) -> Dict[str, int]:
        """Get user engagement statistics for the last n days."""
        result = self.execute_query(f"""
            SELECT DATE(query_timestamp) as date, COUNT(*) 
            FROM conversations 
            WHERE query_timestamp > NOW() - INTERVAL '{days} days'
            GROUP BY DATE(query_timestamp) 
            ORDER BY date DESC 
        """)
        engagement_stats = dict(result)
        logger.info(f"Retrieved user engagement stats for the last {days} days")
        return engagement_stats
    

    def get_model_performance_stats(self) -> List[Dict[str, Any]]:
        """Get model performance statistics."""
        query = """
        SELECT 
            c.model_type,
            AVG(CASE WHEN f.feedback IN ('Somewhat Helpful', 'Very Helpful') THEN 1 ELSE 0 END) as positive_feedback_rate,
            AVG(LENGTH(c.response)) as avg_response_length,
            COUNT(*) as usage_count,
            AVG(c.confidence_score) as avg_confidence_score
        FROM conversations c
        LEFT JOIN feedback f ON c.id = f.conversation_id
        GROUP BY c.model_type
        """
        result = self.execute_query(query)
        performance_stats = [
            {
                "model_type": stat[0],
                "positive_feedback_rate": round(stat[1] * 100, 2) if stat[1] is not None else None,
                "avg_response_length": round(stat[2], 2) if stat[2] is not None else None,
                "usage_count": stat[3],
                "avg_confidence_score": round(stat[4], 2) if stat[4] is not None else None
            }
            for stat in result
        ]
        logger.info(f"Retrieved model performance stats: {performance_stats}")
        return performance_stats

    def get_average_response_time(self) -> Optional[float]:
        query = """
        SELECT AVG(response_time) as avg_response_time
        FROM conversations
        WHERE response_time IS NOT NULL AND response_time > 0
        """
        result = self.execute_query(query)
        avg_response_time = result[0][0] if result and result[0][0] is not None else None
        logger.info(f"Average response time: {avg_response_time}")
        return avg_response_time
    
    
    def get_user_engagement_rate(self) -> Optional[float]:
        query = """
        SELECT 
            COUNT(DISTINCT CASE WHEN conversation_count > 1 THEN user_id END) * 100.0 / NULLIF(COUNT(DISTINCT user_id), 0) as engagement_rate
        FROM (
            SELECT user_id, COUNT(*) as conversation_count
            FROM conversations
            GROUP BY user_id
        ) user_counts
        """
        result = self.execute_query(query)
        engagement_rate = result[0][0] if result and result[0][0] is not None else None
        logger.info(f"User engagement rate: {engagement_rate}")
        return engagement_rate

    def get_query_complexity_stats(self) -> Tuple[float, int, int]:
        """Get query complexity statistics."""
        query = """
        SELECT 
            AVG(LENGTH(query)) as avg_query_length,
            MIN(LENGTH(query)) as min_query_length,
            MAX(LENGTH(query)) as max_query_length
        FROM conversations
        """
        return self.execute_query(query)[0]
    
    def get_error_rate(self) -> Optional[float]:
        query = """
        SELECT 
            COUNT(CASE WHEN feedback IN ('Very Unhelpful', 'Somewhat Unhelpful') THEN 1 END) * 100.0 / NULLIF(COUNT(*), 0) as error_rate
        FROM conversations
        LEFT JOIN feedback ON conversations.id = feedback.conversation_id
        WHERE feedback IS NOT NULL
        """
        result = self.execute_query(query)
        error_rate = round(result[0][0], 2) if result and result[0][0] is not None else None
        logger.info(f"Error rate: {error_rate}")
        return error_rate

    def get_top_queries(self, n: int = 5) -> List[Tuple[str, int]]:
        """Get top n queries by frequency."""
        query = """
        SELECT query, COUNT(*) as frequency
        FROM conversations
        GROUP BY query
        ORDER BY frequency DESC
        LIMIT %s
        """
        return self.execute_query(query, (n,))
    
    def get_model_confidence_stats(self) -> Tuple[float, float, float]:
        query = """
        SELECT 
            AVG(confidence_score) as avg_confidence,
            MIN(NULLIF(confidence_score, 0)) as min_confidence,
            MAX(confidence_score) as max_confidence
        FROM conversations
        WHERE confidence_score IS NOT NULL
        """
        result = self.execute_query(query)
        return tuple(round(val, 2) if val is not None else None for val in result[0])


    def get_active_users(self, days: int = 7) -> int:
        """Get number of active users in the last n days."""
        query = """
        SELECT COUNT(DISTINCT user_id) as active_users
        FROM conversations
        WHERE query_timestamp > NOW() - INTERVAL %s
        """
        result = self.execute_query(query, (f"{days} days",))
        return result[0][0] if result else 0

    
    def get_avg_conversation_length(self) -> Optional[float]:
        query = """
        SELECT AVG(exchange_count) as avg_conversation_length
        FROM (
            SELECT user_id, COUNT(*) as exchange_count
            FROM conversations
            GROUP BY user_id
        ) conversation_lengths
        """
        result = self.execute_query(query)
        return round(result[0][0], 2) if result and result[0][0] is not None else None
   

    def get_daily_conversation_count(self, last_n_days: int = 30) -> List[Tuple[datetime.date, int]]:
        """Get the daily conversation count for the last n days."""
        query = """
        SELECT DATE(query_timestamp) as date, COUNT(*) as count
        FROM conversations
        WHERE query_timestamp > CURRENT_DATE - INTERVAL '%s days'
        GROUP BY DATE(query_timestamp)
        ORDER BY date ASC
        """
        result = self.execute_query(query, (last_n_days,))
        logger.info(f"Retrieved daily conversation count for the last {last_n_days} days")
        return result

    def get_feedback_distribution(self) -> Dict[str, int]:
        """Get the distribution of feedback ratings."""
        query = """
        SELECT feedback, COUNT(*) as count
        FROM feedback
        GROUP BY feedback
        ORDER BY count DESC
        """
        result = self.execute_query(query)
        feedback_distribution = dict(result)
        logger.info(f"Retrieved feedback distribution: {feedback_distribution}")
        return feedback_distribution
