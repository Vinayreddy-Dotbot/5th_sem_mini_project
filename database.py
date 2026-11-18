import sqlite3
from pathlib import Path

USER_COLUMNS = ['user_id', 'name', 'phone', 'monthly_income', 'savings_goal', 'password_hash']


def connect(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def create_tables(db_path):
    with connect(db_path) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            phone TEXT NOT NULL UNIQUE,
            monthly_income REAL NOT NULL DEFAULT 0,
            savings_goal REAL NOT NULL DEFAULT 0,
            password_hash TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS expenses (
            expense_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            expense_date TEXT NOT NULL,
            category TEXT NOT NULL,
            subcategory TEXT,
            amount REAL NOT NULL CHECK(amount > 0),
            note TEXT,
            FOREIGN KEY(user_id) REFERENCES users(user_id))''')
        conn.execute('''CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            predicted_amount REAL NOT NULL,
            prediction_date TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(user_id))''')
        # Older databases were created before login existed: add the column if missing.
        columns = [row[1] for row in conn.execute('PRAGMA table_info(users)')]
        if 'password_hash' not in columns:
            conn.execute('ALTER TABLE users ADD COLUMN password_hash TEXT')


# ---------- users ----------

def _get_user(db_path, where, value):
    with connect(db_path) as conn:
        row = conn.execute(f'SELECT {", ".join(USER_COLUMNS)} FROM users WHERE {where}=?',
                           (value,)).fetchone()
    return dict(zip(USER_COLUMNS, row)) if row else None


def get_user(db_path, user_id):
    return _get_user(db_path, 'user_id', user_id)


def get_user_by_phone(db_path, phone):
    return _get_user(db_path, 'phone', phone)


def create_user(db_path, name, phone, password_hash, income, goal):
    """Create an account. An old profile (same phone, no password yet) is claimed instead."""
    with connect(db_path) as conn:
        conn.execute('''INSERT INTO users(name, phone, monthly_income, savings_goal, password_hash)
            VALUES (?, ?, ?, ?, ?) ON CONFLICT(phone) DO UPDATE SET
            name=excluded.name, monthly_income=excluded.monthly_income,
            savings_goal=excluded.savings_goal, password_hash=excluded.password_hash
            WHERE users.password_hash IS NULL''',
            (name, phone, income, goal, password_hash))
    return get_user_by_phone(db_path, phone)


def update_profile(db_path, user_id, name, income, goal):
    with connect(db_path) as conn:
        conn.execute('UPDATE users SET name=?, monthly_income=?, savings_goal=? WHERE user_id=?',
                     (name, income, goal, user_id))


def update_password(db_path, user_id, password_hash):
    with connect(db_path) as conn:
        conn.execute('UPDATE users SET password_hash=? WHERE user_id=?', (password_hash, user_id))


# ---------- expenses ----------

def add_expense(db_path, user_id, expense_date, category, subcategory, amount, note):
    with connect(db_path) as conn:
        conn.execute('''INSERT INTO expenses
            (user_id, expense_date, category, subcategory, amount, note)
            VALUES (?, ?, ?, ?, ?, ?)''',
            (user_id, expense_date, category, subcategory, amount, note))


def get_expenses(db_path, user_id, start_date=None, end_date=None):
    query = '''SELECT expense_id, expense_date, category, subcategory, amount, note
               FROM expenses WHERE user_id=?'''
    params = [user_id]
    if start_date:
        query += ' AND expense_date >= ?'
        params.append(start_date)
    if end_date:
        query += ' AND expense_date <= ?'
        params.append(end_date)
    query += ' ORDER BY expense_date DESC, expense_id DESC'
    with connect(db_path) as conn:
        return conn.execute(query, params).fetchall()


def delete_expense(db_path, user_id, expense_id):
    with connect(db_path) as conn:
        cur = conn.execute('DELETE FROM expenses WHERE expense_id=? AND user_id=?',
                           (expense_id, user_id))
        return cur.rowcount > 0


# ---------- predictions ----------

def add_prediction(db_path, user_id, amount):
    with connect(db_path) as conn:
        conn.execute('INSERT INTO predictions(user_id, predicted_amount) VALUES (?, ?)',
                     (user_id, amount))


def get_predictions(db_path, user_id):
    with connect(db_path) as conn:
        return conn.execute('''SELECT prediction_id, predicted_amount, prediction_date
            FROM predictions WHERE user_id=? ORDER BY prediction_id DESC''',
            (user_id,)).fetchall()
