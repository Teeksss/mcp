import sqlite3

class DatabaseConnection:
    def __init__(self, db_url):
        self.db_url = db_url
        self.conn = None

    def connect(self):
        if self.conn is None:
            self.conn = sqlite3.connect(self.db_url.replace("sqlite:///", ""))
        return self.conn

    def close(self):
        if self.conn:
            self.conn.close()