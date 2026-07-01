import sqlite3
import matplotlib.pyplot as plt
import sys

def show_attendance(student_id):
    # Connect to database
    conn = sqlite3.connect("db/attendance.db")
    cursor = conn.cursor()

    # Get attendance records for the student
    cursor.execute(
        "SELECT date, status FROM attendance WHERE student_id=?",
        (student_id,)
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        print(f"No records found for student ID: {student_id}")
        return

    # Prepare data for chart
    dates = [r[0] for r in rows]
    statuses = [1 if r[1] == "Present" else 0 for r in rows]
    colors = ["green" if s == 1 else "red" for s in statuses]

    # Plot bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(dates, statuses, color=colors)
    plt.title(f"Attendance History - {student_id}")
    plt.xlabel("Date")
    plt.ylabel("Present (1) / Absent (0)")
    plt.xticks(rotation=45)
    plt.yticks([0, 1], ["Absent", "Present"])
    plt.tight_layout()
    plt.savefig("attendance_chart.png")
    plt.show()
    print("Chart saved as attendance_chart.png")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python infovis.py <student_id>")
        print("Example: python infovis.py 10000409")
        sys.exit(1)

    student_id = sys.argv[1]
    show_attendance(student_id)