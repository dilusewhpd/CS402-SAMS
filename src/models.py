from dataclasses import dataclass


@dataclass
class Student:
    no: str
    student_id: str
    name: str


@dataclass
class AttendanceRecord:
    student_id: str
    name: str
    date: str
    status: str  # "Present" or "Absent"