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
    def __init__(self, database: AttendanceDatabase):
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

        present_count = sum(statuses)
        total = len(statuses)
        pct = present_count / total * 100

        plt.figure(figsize=(10, 6))
        plt.bar(dates, statuses, color=colors)
        plt.title(f"Attendance History - {student_name} ({student_id})\n"
                  f"Present {present_count}/{total} ({pct:.0f}%)")
        plt.xlabel("Date")
        plt.ylabel("Status")
        plt.xticks(rotation=45)
        plt.yticks([0, 1], ["Absent", "Present"])
        plt.tight_layout()
        plt.savefig("attendance_chart.png")
        plt.show()

        print(f"Present: {present_count}/{total} ({pct:.0f}%)")
        print("Chart saved as attendance_chart.png")


def main():
    if len(sys.argv) != 2:
        print("Usage: python infovis.py <student_id>")
        print("Example: python infovis.py 10000409")
        sys.exit(1)

    student_id = sys.argv[1]
    database = AttendanceDatabase(config.DB_PATH)
    AttendanceVisualizer(database).show(student_id)


if __name__ == "__main__":
    main()