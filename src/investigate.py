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
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

class SignatureInvestigator:
    def __init__(self, repository: StudentRepository, database: AttendanceDatabase):
        self.repository = repository
        self.database = database
        self.image_proc = ImageProcessor(show_steps=False)

    def investigate(self, student_id: str) -> None:
        print(f"Starting investigation for student: {student_id}")
        # TODO: Implement database fetch and image extraction

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python investigate.py <student_id>")
        sys.exit(1)
    
    investigator = SignatureInvestigator(StudentRepository(config.XML_PATH), AttendanceDatabase(config.DB_PATH))
    investigator.investigate(sys.argv[1])

if __name__ == "__main__":
    main()