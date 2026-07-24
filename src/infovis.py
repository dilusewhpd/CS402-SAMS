"""
Attendance visualization for a single student.

Usage:
    python infovis.py <student_id>
"""

import sys
from datetime import datetime
import matplotlib.pyplot as plt

import config
from database import AttendanceDatabase


class AttendanceVisualizer:
    def _init_(self, database: AttendanceDatabase):
        self.database = database

    def show(self, student_id: str):
        rows = self.database.get_records_for_student(student_id)
        if not rows:
            print(f"No records found for student ID: {student_id}")
            return

        student_name = rows[0][2]

        rows_sorted = sorted(rows, key=lambda r: datetime.strptime(r[0], "%d-%m-%Y"))
        dates = [r[0] for r in rows_sorted]
        statuses = [1 if r[1] == "Present" else 0 for r in rows_sorted]
        colors = ["green" if s else "red" for s in statuses]