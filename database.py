import sqlite3
import os

DB_PATH = "uploads.db"

def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def save_result(filename, predicted_label, confidence):
    """Save a prediction result to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO uploads (filename, predicted_label, confidence)
        VALUES (?, ?, ?)
    """, (filename, predicted_label, confidence))
    conn.commit()
    conn.close()

def get_all_results():
    """Fetch all past upload results from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, filename, predicted_label, confidence, timestamp FROM uploads ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
