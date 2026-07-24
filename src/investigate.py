"""Investigate a student's signature consistency across signing sheets."""

import glob
import os
import re
import sys
from typing import List, Tuple
import cv2
import numpy as np
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
            if not present_dates: return

            students = self.repository.load_students()
            row_index = next((i for i, s in enumerate(students) if s.student_id == student_id), -1)
            if row_index == -1: return

            samples: List[Tuple[str, str, np.ndarray]] = []
            image_files = glob.glob(os.path.join(config.IMAGE_DIR, "*.jpeg")) + glob.glob(os.path.join(config.IMAGE_DIR, "*.png"))

            for image_path in image_files:
                image_name = os.path.basename(image_path)
                match = re.search(r"(\d{2}\.\d{2}\.\d{4})", image_name)
                image_date = match.group(1).replace(".", "-") if match else config.IMAGE_DATES.get(image_name)

                if image_date in present_dates:
                    binary = self.image_proc.process(image_path)
                    coords = config.CELL_COORDINATES.get(image_name)
                    if not coords or row_index >= len(coords): continue

                    y1, y2, x1, x2 = coords[row_index]
                    crop = binary[y1:y2, x1:x2]
                    
                    # Placeholder for preprocessing
                    samples.append((image_date, image_name, crop))
            
            print(f"Extracted {len(samples)} signature crops.")
        except Exception as exc:
            print(f"Investigation failed: {exc}")

def main() -> None:
    if len(sys.argv) != 2: sys.exit(1)
    SignatureInvestigator(StudentRepository(config.XML_PATH), AttendanceDatabase(config.DB_PATH)).investigate(sys.argv[1])

if __name__ == "__main__":
    main()