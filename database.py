"""
SQLite database module for Health Analysis application.
Replaces JSON file storage with SQLite for users and contacts.
"""

import sqlite3
import os
import json
from contextlib import contextmanager

# Database path
DB_PATH = os.path.join("data", "app.db")

# JSON paths for migration
USERS_JSON_PATH = os.path.join("data", "users.json")
CONTACTS_JSON_PATH = os.path.join("data", "contacts.json")


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    """Initialize the database tables and migrate existing JSON data."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
            )
        """)
        
        # Create contacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                age TEXT NOT NULL,
                gender TEXT NOT NULL,
                phone TEXT NOT NULL,
                problem TEXT NOT NULL,
                details TEXT,
                status TEXT NOT NULL DEFAULT 'new'
            )
        """)
        
        conn.commit()
        
        # Migrate existing JSON data if tables are empty
        _migrate_json_data(conn)


def _migrate_json_data(conn):
    """Migrate existing JSON data to SQLite (one-time operation)."""
    cursor = conn.cursor()
    
    # Check if users table is empty
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0 and os.path.exists(USERS_JSON_PATH):
        try:
            with open(USERS_JSON_PATH, "r") as f:
                data = json.load(f)
                users = data.get("users", [])
                for user in users:
                    cursor.execute(
                        "INSERT OR IGNORE INTO users (email, password, role) VALUES (?, ?, ?)",
                        (user.get("email"), user.get("password"), user.get("role", "user"))
                    )
            print(f"Migrated {len(users)} users from JSON to SQLite")
        except Exception as e:
            print(f"Error migrating users: {e}")
    
    # Check if contacts table is empty
    cursor.execute("SELECT COUNT(*) FROM contacts")
    if cursor.fetchone()[0] == 0 and os.path.exists(CONTACTS_JSON_PATH):
        try:
            with open(CONTACTS_JSON_PATH, "r") as f:
                data = json.load(f)
                submissions = data.get("submissions", [])
                for contact in submissions:
                    cursor.execute(
                        """INSERT INTO contacts (name, age, gender, phone, problem, details, status) 
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            contact.get("name"),
                            contact.get("age"),
                            contact.get("gender"),
                            contact.get("phone"),
                            contact.get("problem"),
                            contact.get("details", ""),
                            contact.get("status", "new")
                        )
                    )
            print(f"Migrated {len(submissions)} contacts from JSON to SQLite")
        except Exception as e:
            print(f"Error migrating contacts: {e}")
    
    conn.commit()


# ============ User Functions ============

def get_all_users():
    """Get all users from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email, password, role FROM users")
        return [dict(row) for row in cursor.fetchall()]


def get_user_by_email(email):
    """Get a user by email address."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT email, password, role FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        return dict(row) if row else None


def add_user(email, password, role="user"):
    """Add a new user to the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (email, password, role) VALUES (?, ?, ?)",
                (email, password, role)
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Email already exists


def user_exists(email):
    """Check if a user with the given email exists."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM users WHERE email = ?", (email,))
        return cursor.fetchone() is not None


# ============ Contact Functions ============

def get_all_contacts():
    """Get all contact submissions from the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, name, age, gender, phone, problem, details, status FROM contacts")
        return [dict(row) for row in cursor.fetchall()]


def get_active_contacts():
    """Get contact submissions that are not resolved."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, name, age, gender, phone, problem, details, status FROM contacts WHERE status != 'resolved'"
        )
        return [dict(row) for row in cursor.fetchall()]


def add_contact(name, age, gender, phone, problem, details=""):
    """Add a new contact submission."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO contacts (name, age, gender, phone, problem, details, status)
               VALUES (?, ?, ?, ?, ?, ?, 'new')""",
            (name, age, gender, phone, problem, details)
        )
        conn.commit()
        return cursor.lastrowid


def update_contact_status(contact_id, status):
    """Update the status of a contact submission."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE contacts SET status = ? WHERE id = ?",
            (status, contact_id)
        )
        conn.commit()
        return cursor.rowcount > 0
