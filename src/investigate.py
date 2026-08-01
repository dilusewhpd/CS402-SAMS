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
DATA_DIR = os.path.join(ROOT_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

class SignatureInvestigator:
    """Compare a student's signatures across multiple signing sheets."""
    SIMILARITY_THRESHOLD = 0.60

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

            print(f"Found {len(present_dates)} 'Present' records. Extracting signatures...")
            samples: List[Tuple[str, str, np.ndarray]] = []
            students = self.repository.load_students()

            row_index = next((i for i, s in enumerate(students) if s.student_id == student_id), -1)
            student = next((s for s in students if s.student_id == student_id), None)

            if row_index == -1 or student is None:
                print("Student not found in XML.")
                return

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
                    
                    if np.count_nonzero(processed) < 50: continue
                    samples.append((image_date, image_name, processed))

            if not samples: return
            self._report_similarity(student.name, student_id, samples)

        except Exception as exc:
            print(f"Investigation failed: {exc}")

    def _prepare_signature(self, crop: np.ndarray) -> np.ndarray:
        TARGET_W, TARGET_H = 300, 150
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_inv = cv2.bitwise_not(binary)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

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

    def _report_similarity(self, student_name: str, student_id: str, samples: List[Tuple[str, str, np.ndarray]]) -> None:
        print(f"Comparing {len(samples)} signature samples for {student_name} ({student_id})")
        print("\nDate | Similarity vs. first/reference signature | Status\n" + "-" * 70)

        reference_date, _, reference_sig = samples[0]
        scores: List[float] = []
        summary_rows: List[Tuple[str, float, str]] = []

        for date, _, sample_sig in samples:
            score = self._similarity(reference_sig, sample_sig)
            scores.append(score)
            summary_rows.append((date, score, "unknown"))

        nonref_scores = [s for i, s in enumerate(scores) if i != 0]
        if len(nonref_scores) >= 2:
            mean, std = float(np.mean(nonref_scores)), float(np.std(nonref_scores))
            dynamic_cutoff = max(0.0, mean - 1.0 * std)
        elif len(nonref_scores) == 1:
            mean, std, dynamic_cutoff = nonref_scores[0], 0.0, self.SIMILARITY_THRESHOLD
        else:
            mean, std, dynamic_cutoff = 0.0, 0.0, self.SIMILARITY_THRESHOLD

        updated_rows: List[Tuple[str, float, str]] = []
        for date, score, _ in summary_rows:
            if date == reference_date:
                updated_rows.append((date, score, "reference"))
                print(f"{date:<15} | {'1.000':>8} | reference")
            else:
                status = "match" if score >= dynamic_cutoff else "mismatch"
                updated_rows.append((date, score, status))
                print(f"{date:<15} | {score:>8.3f} | {status}")

        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                score = self._similarity(samples[i][2], samples[j][2])
                if len(nonref_scores) >= 1 and score < max(0.0, mean - 1.5 * std):
                    print(f"Possible signature mismatch detected between {samples[i][0]} and {samples[j][0]} (similarity: {score:.3f})")

        self._save_comparison_image(student_id, updated_rows, samples)

    def _save_comparison_image(self, student_id: str, summary_rows: List[Tuple[str, float, str]], samples: List[Tuple[str, str, np.ndarray]]) -> None:
        panels = []
        sample_map = {date: img for (date, _, img) in samples}
        for date, score, status in summary_rows:
            panel = np.full((220, 220, 3), 255, dtype=np.uint8)
            sig = sample_map.get(date)
            if sig is not None:
                sig_rgb = cv2.cvtColor(sig, cv2.COLOR_GRAY2BGR) if len(sig.shape) == 2 else sig
                h, w = sig_rgb.shape[:2]
                scale = min(200 / float(w), 120 / float(h), 1.0)
                new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
                sig_resized = cv2.resize(sig_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                panel[40 : 40 + new_h, 10 : 10 + new_w] = sig_resized

            cv2.putText(panel, date, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(panel, f"score: {score:.2f}", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(panel, status, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            panels.append(panel)

        if panels:
            output_path = os.path.join(ROOT_DIR, f"investigate_{student_id}.png")
            cv2.imwrite(output_path, cv2.hconcat(panels))
            print(f"Saved comparison image to {output_path}")

    def _similarity(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        target_h, target_w = 150, 300
        if sig_a.shape != (target_h, target_w): sig_a = cv2.resize(sig_a, (target_w, target_h))
        if sig_b.shape != (target_h, target_w): sig_b = cv2.resize(sig_b, (target_w, target_h))
        if sig_a.size == 0 or sig_b.size == 0: return 0.0

        img1, img2 = cv2.GaussianBlur(sig_a.astype(np.float64), (3, 3), 0), cv2.GaussianBlur(sig_b.astype(np.float64), (3, 3), 0)
        C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        kernel_size, sigma = (11, 11), 1.5

        mu1, mu2 = cv2.GaussianBlur(img1, kernel_size, sigma), cv2.GaussianBlur(img2, kernel_size, sigma)
        mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 * img1, kernel_size, sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, kernel_size, sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, kernel_size, sigma) - mu1_mu2

        denom = np.where((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) == 0, 1e-9, (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return max(0.0, min(1.0, float(np.mean(((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / denom))))

def main() -> None:
    if len(sys.argv) != 2: sys.exit(1)
    SignatureInvestigator(StudentRepository(config.XML_PATH), AttendanceDatabase(config.DB_PATH)).investigate(sys.argv[1])

if __name__ == "__main__":
    main()