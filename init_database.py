import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from dotenv import load_dotenv
import os

# --- Load env ---
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")

# 1️⃣ Connect to system DB
conn = psycopg2.connect(
    dbname="postgres", user=DB_USER, password=DB_PASSWORD, host=DB_HOST
)
conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
cursor = conn.cursor()

# 2️⃣ Create DB if missing
cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}';")
exists = cursor.fetchone()
if not exists:
    cursor.execute(f"CREATE DATABASE {DB_NAME};")
    print(f"✅ Created database {DB_NAME}")
else:
    print(f"✅ Database {DB_NAME} already exists.")

cursor.close()
conn.close()

# 3️⃣ Connect to your DB
conn = psycopg2.connect(
    dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST
)
cursor = conn.cursor()

# 4️⃣ Create tables if missing
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
"""
)

cursor.execute(
    """
CREATE TABLE IF NOT EXISTS chat_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    title TEXT,
    messages JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
"""
)

conn.commit()
cursor.close()
conn.close()

print("✅ Tables ready.")
