import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from models import Student, AttendanceRecord

def test_student_model():
    student = Student(no="1", student_id="10000409", name="John Doe")
    assert student.no == "1"
    assert student.student_id == "10000409"
    assert student.name == "John Doe"

def test_attendance_record_model():
    record = AttendanceRecord(student_id="10000409", name="John Doe", date="10-07-2019", status="Present")
    assert record.student_id == "10000409"
    assert record.name == "John Doe"
    assert record.date == "10-07-2019"
    assert record.status == "Present"
