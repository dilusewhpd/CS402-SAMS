"""SQLite persistence for attendance records."""

import os
import sqlite3
from models import AttendanceRecord


class AttendanceDatabase:
    def __init__(self, db_path: str):
        self.db_path = db_path
        folder = os.path.dirname(db_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        self._create_table()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_table(self):
        conn = self._connect()
        conn.execute('''CREATE TABLE IF NOT EXISTS attendance (
            student_id TEXT,
            name TEXT,
            date TEXT,
            status TEXT
        )''')
        conn.commit()
        conn.close()

    def save_records(self, records: list[AttendanceRecord]):
        """Replaces any existing records for the dates being saved (so
        re-running the same sheet doesn't create duplicates), then inserts
        the new set."""
        conn = self._connect()
        cursor = conn.cursor()

        dates_in_batch = {r.date for r in records}
        for date in dates_in_batch:
            cursor.execute("DELETE FROM attendance WHERE date=?", (date,))

        for r in records:
            cursor.execute(
                "INSERT INTO attendance VALUES (?, ?, ?, ?)",
                (r.student_id, r.name, r.date, r.status)
            )
            print(f"Saved: {r.name} - {r.status} - {r.date}")

        conn.commit()
        conn.close()
        print("Attendance records saved to database successfully!")

    def get_records_for_student(self, student_id: str) -> list[tuple]:
        """Returns (date, status, name) tuples for a student, or [] if none exist."""
        conn = self._connect()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "SELECT date, status, name FROM attendance WHERE student_id=?",
                (student_id,)
            )
            rows = cursor.fetchall()
        except sqlite3.OperationalError:
            rows = []
        conn.close()
        return rows