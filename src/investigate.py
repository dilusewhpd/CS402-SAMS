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
                    
                    processed = self._prepare_signature(crop)
                    if np.count_nonzero(processed) >= 50:
                        samples.append((image_date, image_name, processed))
            
            print(f"Preprocessed {len(samples)} valid signatures.")
        except Exception as exc:
            print(f"Investigation failed: {exc}")

    def _prepare_signature(self, crop: np.ndarray) -> np.ndarray:
        TARGET_W, TARGET_H = 300, 150
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cleaned = cv2.morphologyEx(cv2.bitwise_not(binary), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        
        coords = cv2.findNonZero(cleaned)
        if coords is None: return np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
        
        x, y, w, h = cv2.boundingRect(coords)
        tight = cleaned[y : y + h, x : x + w]
        canvas = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
        
        if w > TARGET_W or h > TARGET_H:
            scale = min(TARGET_W / float(w), TARGET_H / float(h))
            new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
            tight = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_AREA)
            w, h = new_w, new_h

        x_off, y_off = (TARGET_W - w) // 2, (TARGET_H - h) // 2
        canvas[y_off : y_off + h, x_off : x_off + w] = tight
        return cv2.GaussianBlur(canvas, (3, 3), 0)

def main() -> None:
    if len(sys.argv) != 2: sys.exit(1)
    SignatureInvestigator(StudentRepository(config.XML_PATH), AttendanceDatabase(config.DB_PATH)).investigate(sys.argv[1])

if __name__ == "__main__":
    main()