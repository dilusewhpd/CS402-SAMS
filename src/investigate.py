"""Investigate a student's signature consistency across signing sheets."""

import glob
import os
import re
import sys
from typing import List, Tuple

import config
from database import AttendanceDatabase
from image_processing import ImageProcessor
from student_repository import StudentRepository

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class SignatureInvestigator:
    def __init__(self, repository: StudentRepository, database: AttendanceDatabase):
        self.repository = repository
        self.database = database
        self.image_proc = ImageProcessor(show_steps=False)

    def investigate(self, student_id: str) -> None:
        try:
            records = self.database.get_records_for_student(student_id)
            present_dates = [r[0] for r in records if r[1] == "Present"]

            if not present_dates:
                print(f"No present attendance records found for student {student_id}.")
                return

            print(f"Found {len(present_dates)} 'Present' records.")
            students = self.repository.load_students()
            student = next((s for s in students if s.student_id == student_id), None)
            
            if student is None:
                print("Student not found in XML.")
                return
                
            # TODO: Extract signatures from images
        except Exception as exc:
            print(f"Investigation failed: {exc}")

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python investigate.py <student_id>")
        sys.exit(1)
    investigator = SignatureInvestigator(StudentRepository(config.XML_PATH), AttendanceDatabase(config.DB_PATH))
    investigator.investigate(sys.argv[1])

if __name__ == "__main__":
    main()