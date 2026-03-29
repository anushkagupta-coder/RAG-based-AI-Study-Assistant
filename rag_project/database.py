import sqlite3

def create_connection():
    conn = sqlite3.connect("study_app.db", check_same_thread=False)
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()


def insert_chat(question, answer):
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO chat_history (question, answer)
    VALUES (?, ?)
    """, (question, answer))

    conn.commit()
    conn.close()


def get_chat_history():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM chat_history ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()
    return data


def clear_history():
    conn = create_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM chat_history")

    conn.commit()
    conn.close()