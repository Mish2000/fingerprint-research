import psycopg
from pgvector.psycopg import register_vector
import logging

DB_CONN_STR = "dbname=biometric_db user=admin password=biometric_secret host=127.0.0.1 port=5432"


class BiometricDBService:
    def __init__(self):
        self.conn_str = DB_CONN_STR
        self._init_db()

    def _get_connection(self):
        conn = psycopg.connect(self.conn_str, autocommit=True)
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        register_vector(conn)
        return conn

    def _init_db(self):
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # יצירת טבלה אחת המכילה את שני הוקטורים בגדלים המדויקים
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS users (
                            user_id SERIAL PRIMARY KEY,
                            username VARCHAR(100) UNIQUE NOT NULL,
                            face_vector vector(512),
                            fingerprint_vector vector(512),
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        );
                    """)

                    # יצירת אינדקסים לחיפוש מהיר לכל אחד מהוקטורים
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_face 
                        ON users USING hnsw (face_vector vector_cosine_ops);
                    """)
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_finger 
                        ON users USING hnsw (fingerprint_vector vector_cosine_ops);
                    """)
        except Exception as e:
            logging.error(f"DB Init error: {e}")
            raise

    def enroll_user(self, username: str, face_vec: list = None, finger_vec: list = None) -> int:
        # כאן אנחנו משתמשים בטרנזקציה: או שהכל נשמר יחד, או שכלום לא נשמר. אין יותר נתונים חלקיים.
        with psycopg.connect(self.conn_str) as conn:
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO users (username, face_vector, fingerprint_vector) 
                    VALUES (%s, %s, %s) RETURNING user_id;
                """, (username, face_vec, finger_vec))
                user_id = cur.fetchone()[0]
                conn.commit()
                return user_id

    def authenticate_fusion(self, face_probe: list, finger_probe: list, face_w: float, finger_w: float):
        query = """
            SELECT user_id, username,
                   ( %s * (1 - (face_vector <=> %s::vector)) + 
                     %s * (1 - (fingerprint_vector <=> %s::vector)) ) AS fused_score
            FROM users
            ORDER BY fused_score DESC
            LIMIT 1;
        """
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (face_w, face_probe, finger_w, finger_probe))
                result = cur.fetchone()
                if result:
                    return {"user_id": result[0], "username": result[1], "score": result[2]}
                return None