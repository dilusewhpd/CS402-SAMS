"""
Main entry point for the Student Attendance Management System (SAMS).

Usage:
    python sams.py <image_path> <xml_path>
"""

import sys
import os
import re
import cv2
import pytesseract

import config
from image_processor import ImageProcessor
from signature_detector import SignatureDetector
from student_repository import StudentRepository
from database import AttendanceDatabase
from models import AttendanceRecord

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


class AttendanceSystem:
    """Coordinates the whole pipeline: image -> detection -> database."""

    def __init__(self, xml_path: str):
        self.image_processor = ImageProcessor(show_steps=True)
        self.signature_detector = SignatureDetector(
            config.CELL_COORDINATES,
            config.INK_THRESHOLD_PERCENT,
            config.BLOB_AREA_THRESHOLD,
        )
        self.student_repository = StudentRepository(xml_path)
        self.database = AttendanceDatabase(config.DB_PATH)

    def process_sheet(self, image_path: str):
        image_name = os.path.basename(image_path)
        date = self._resolve_date(image_name)

        print(f"Processing image: {image_path}")
        print(f"Attendance date: {date}")

        binary = self.image_processor.process(image_path)
        results = self.signature_detector.detect(binary, image_name)
        students = self.student_repository.load_students()

        records = [
            AttendanceRecord(student_id=s.student_id, name=s.name, date=date, status=results[i])
            for i, s in enumerate(students)
        ]
        self.database.save_records(records)

        cv2.destroyAllWindows()
        print("Program completed successfully!")

    @staticmethod
    def _resolve_date(image_name: str) -> str:
        # Try to parse date from filename (e.g. 10.07.2019.png)
        match = re.search(r"(\d{2}\.\d{2}\.\d{4})", image_name)
        if match:
            return match.group(1).replace(".", "-")

        # Fallback to config mapping
        date = config.IMAGE_DATES.get(image_name)
        if not date or date.startswith("TODO"):
            print(f"ERROR: No date set for {image_name} in config.IMAGE_DATES and not found in filename. "
                  f"Open the sheet, read the date in the header, and add it there.")
            sys.exit(1)
        return date


def main():
    if len(sys.argv) != 3:
        print("Usage: python sams.py <image_path> <xml_path>")
        print("Example: python sams.py 10.07.2019.png data/info.xml")
        sys.exit(1)

    image_path = sys.argv[1]
    xml_path = sys.argv[2]

    AttendanceSystem(xml_path).process_sheet(image_path)


if __name__ == "__main__":
    main()