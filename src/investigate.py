"""Investigate a student's signature consistency across signing sheets.

This script reuses the project’s existing student roster and attendance database
logic, crops the same signature cell used by the attendance pipeline, prepares
those crops for comparison, and reports whether the student’s signatures look
consistent across dates.
"""

import os
import sys
from typing import List, Tuple

import cv2
import numpy as np

import config
from database import AttendanceDatabase
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

    def investigate(self, student_id: str) -> None:
        """Load the student, attendance records, and signature crops for comparison."""
        try:
            students = self.repository.load_students()
            student = next((s for s in students if s.student_id == student_id), None)
            if student is None:
                try:
                    row_index = int(student_id) - 1
                    if 0 <= row_index < len(students):
                        student = students[row_index]
                    else:
                        print(f"Student ID not found: {student_id}")
        self.student_repo = student_repo
        self.image_proc = ImageProcessor(show_steps=False)

    def investigate(self, student_id: str):
        # 1. Fetch only "Present" records for the student
        records = self.database.get_records_for_student(student_id)
        present_dates = [r[0] for r in records if r[1] == "Present"]

        if not present_dates:
            print(f"No present attendance records found for student {student_id}.")
            return

        print(f"Found {len(present_dates)} 'Present' records. Extracting signatures...")
        
        # 2. Extract signatures from the images on those dates
        signatures = []
        students = self.student_repo.load_students()
        
        # Find which row corresponds to this student
        row_index = next((i for i, s in enumerate(students) if s.student_id == student_id), -1)
        if row_index == -1:
            print("Student not found in XML.")
            return

        # Scan all images in data/images/
        image_files = glob.glob(os.path.join(config.IMAGE_DIR, "*.jpeg")) + \
                      glob.glob(os.path.join(config.IMAGE_DIR, "*.png"))
                      
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            
            # Try to parse date from filename first, then config
            match = re.search(r"(\d{2}\.\d{2}\.\d{4})", image_name)
            if match:
                image_date = match.group(1).replace(".", "-")
            else:
                image_date = config.IMAGE_DATES.get(image_name)

            if image_date in present_dates:
                # Run image processing to get binary
                binary = self.image_proc.process(image_path)
                
                # Fetch hardcoded box for this image
                coords = config.CELL_COORDINATES.get(image_name)
                if not coords or row_index >= len(coords):
                    print(f"Skipping {image_date}: no coordinates configured for {image_name}")
                    continue
                    
                y1, y2, x1, x2 = coords[row_index]
                crop = binary[y1:y2, x1:x2]
                
                processed = self._prepare_signature(crop)
                if np.count_nonzero(processed) < 50:
                    print(f"Skipping {image_date}: signature not detected in {image_name}")
                    continue
            self._report_similarity(student.name, student_id, samples)
        except Exception as exc:  # pragma: no cover - defensive reporting for the report
            print(f"Investigation failed: {exc}")

    def _prepare_signature(self, crop: np.ndarray) -> np.ndarray:
        """Convert a signature crop into a normalized grayscale image.

        Steps:
        - convert to gray and binarize with Otsu (consistent for all images)
        - invert so signature strokes are white on black
        - remove small noise via morphological open
        - compute tight bounding box around strokes and crop
        - center the tight crop on a fixed-size canvas (300x150)
        - apply a small Gaussian blur to tolerate stroke/photographic noise
        """
        TARGET_W, TARGET_H = 300, 150

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Consistent binarization using Otsu. Foreground (ink) will be dark (0).
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert so ink/strokes are white (255) on black (0) which is easier to process.
        binary_inv = cv2.bitwise_not(binary)

        # Remove small noise while preserving the main signature strokes.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel, iterations=1)

        # Find tight bounding rect around the signature strokes.
        coords = cv2.findNonZero(cleaned)
        if coords is None:
            # No ink found — return an empty canvas of the target size.
            return np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)

        x, y, w, h = cv2.boundingRect(coords)
        tight = cleaned[y : y + h, x : x + w]

        # Center the tight crop on a fixed-size black canvas.
        canvas = np.zeros((TARGET_H, TARGET_W), dtype=np.uint8)
        # If a signature is larger than the canvas, resize it preserving aspect ratio.
        if w > TARGET_W or h > TARGET_H:
            scale = min(TARGET_W / float(w), TARGET_H / float(h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            tight = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_AREA)
            w, h = new_w, new_h

        x_off = (TARGET_W - w) // 2
        y_off = (TARGET_H - h) // 2
        canvas[y_off : y_off + h, x_off : x_off + w] = tight

        # Small blur to be tolerant of minor stroke/pixel differences.
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0)
        return canvas

    def _save_debug_samples(self, student_id: str, samples: List[Tuple[str, str, np.ndarray]]) -> None:
        """Save each processed signature crop for manual inspection."""
        os.makedirs("cells/investigate", exist_ok=True)
        for date, image_name, sample in samples:
            output_path = os.path.join("cells", "investigate", f"{student_id}_{date}_{image_name}.png")
            cv2.imwrite(output_path, sample)

    def _report_similarity(self, student_name: str, student_id: str, samples: List[Tuple[str, str, np.ndarray]]) -> None:
        """Print a comparison report and save a side-by-side visualization image."""
        print(f"Comparing {len(samples)} signature samples for {student_name} ({student_id})")
        print("\nDate | Similarity vs. first/reference signature | Status")
        print("-" * 70)

        reference_date, _, reference_sig = samples[0]
        # Compute SSIM scores of every sample against the reference.
        scores: List[float] = []
        summary_rows: List[Tuple[str, float, str]] = []

        for date, _, sample_sig in samples:
            score = self._similarity(reference_sig, sample_sig)
            scores.append(score)
            # We'll decide match/mismatch after computing class-level statistics.
            summary_rows.append((date, score, "unknown"))

        # Print raw SSIMs for manual inspection (reference is first sample).
        for date, score, _ in summary_rows:
            if date == reference_date:
                print(f"{date:<15} | {'1.000':>8} | reference")
            else:
                print(f"{date:<15} | {score:>8.3f} |")

        # Decide mismatches by checking for outliers relative to the student's own scores.
        nonref_scores = [s for i, s in enumerate(scores) if i != 0]
        if len(nonref_scores) >= 2:
            mean = float(np.mean(nonref_scores))
            std = float(np.std(nonref_scores))
            # Dynamic cutoff: flag as mismatch if score is more than ~1 std below mean.
            dynamic_cutoff = max(0.0, mean - 1.0 * std)
        elif len(nonref_scores) == 1:
            # Fallback to global threshold when only one comparison exists.
            mean = nonref_scores[0]
            std = 0.0
            dynamic_cutoff = self.SIMILARITY_THRESHOLD
        else:
            mean = 0.0
            std = 0.0
            dynamic_cutoff = self.SIMILARITY_THRESHOLD

        # Update statuses with the dynamic cutoff and print final table lines.
        print("\nDate | Similarity vs. first/reference signature | Status")
        print("-" * 70)
        updated_rows: List[Tuple[str, float, str]] = []
        for date, score, _ in summary_rows:
            if date == reference_date:
                updated_rows.append((date, score, "reference"))
            else:
                status = "match" if score >= dynamic_cutoff else "mismatch"
                updated_rows.append((date, score, status))
                print(f"{date:<15} | {score:>8.3f} | {status}")

        # Check every pair of signatures for mismatches below the threshold.
        # Also check pairwise comparisons and report any clear outliers.
        for i in range(len(samples)):
            for j in range(i + 1, len(samples)):
                date_a, _, sig_a = samples[i]
                date_b, _, sig_b = samples[j]
                score = self._similarity(sig_a, sig_b)
                # Use the same dynamic logic: if this pair is well below the student's mean, report it.
                if len(nonref_scores) >= 1 and score < max(0.0, mean - 1.5 * std):
                    print(
                        f"Possible signature mismatch detected between {date_a} and {date_b} "
                        f"(similarity: {score:.3f})"
                    )

        # Save a side-by-side visualization for the report.
        self._save_comparison_image(student_id, updated_rows, samples)

        if any(status == "mismatch" for _, _, status in updated_rows[1:]):
            print("\nInvestigation result: possible mismatch detected.")
        else:
            print("\nInvestigation result: signatures appear consistent.")

    def _save_comparison_image(self, student_id: str, summary_rows: List[Tuple[str, float, str]],
                               samples: List[Tuple[str, str, np.ndarray]]) -> None:
        """Create a side-by-side image showing signature crops and their scores."""
        panels = []
        # Map dates to processed sample images for display
        sample_map = {date: img for (date, _, img) in samples}
        for date, score, status in summary_rows:
            # Create a panel with white background
            panel = np.full((220, 220, 3), 255, dtype=np.uint8)
            # Draw signature image if available
            sig = sample_map.get(date)
            if sig is not None:
                # sig is single-channel; convert to BGR and resize to fit
                sig_rgb = cv2.cvtColor(sig, cv2.COLOR_GRAY2BGR)
                h, w = sig_rgb.shape[:2]
                max_w, max_h = 200, 120
                scale = min(max_w / float(w), max_h / float(h), 1.0)
                new_w = max(1, int(w * scale))
                new_h = max(1, int(h * scale))
                sig_resized = cv2.resize(sig_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                x_off = 10
                y_off = 40
                panel[y_off : y_off + new_h, x_off : x_off + new_w] = sig_resized

            cv2.putText(panel, date, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(panel, f"score: {score:.2f}", (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(panel, status, (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            panels.append(panel)

        if not panels:
            return

        output_path = os.path.join(ROOT_DIR, f"investigate_{student_id}.png")
        canvas = cv2.hconcat(panels)
        cv2.imwrite(output_path, canvas)
        print(f"Saved comparison image to {output_path}")

    def _similarity(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Compare two preprocessed signature images and return a similarity score in [0, 1]."""
        # Ensure both images are the same target size.
        target_h, target_w = 150, 300
        if sig_a.shape != (target_h, target_w):
            sig_a = cv2.resize(sig_a, (target_w, target_h), interpolation=cv2.INTER_AREA)
        if sig_b.shape != (target_h, target_w):
            sig_b = cv2.resize(sig_b, (target_w, target_h), interpolation=cv2.INTER_AREA)

        if sig_a.size == 0 or sig_b.size == 0:
            return 0.0

        # Convert to float64 for SSIM math and normalize to [0,255].
        img1 = sig_a.astype(np.float64)
        img2 = sig_b.astype(np.float64)

        # Apply a small Gaussian blur to tolerate small pixel/pen-width differences.
        img1 = cv2.GaussianBlur(img1, (3, 3), 0)
        img2 = cv2.GaussianBlur(img2, (3, 3), 0)

        # Compute SSIM (following the typical formula used by scikit-image).
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        # Gaussian kernel for local statistics
        kernel_size = (11, 11)
        sigma = 1.5

        mu1 = cv2.GaussianBlur(img1, kernel_size, sigma)
        mu2 = cv2.GaussianBlur(img2, kernel_size, sigma)

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(img1 * img1, kernel_size, sigma) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 * img2, kernel_size, sigma) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, kernel_size, sigma) - mu1_mu2

        # SSIM map
        denom = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        # Avoid division by zero
        denom = np.where(denom == 0, 1e-9, denom)
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / denom

        # Mean SSIM over the image
        score = float(np.mean(ssim_map))
        # Clip to [0, 1]
        return max(0.0, min(1.0, score))


def main() -> None:
    """CLI entry point for the investigator."""
    if len(sys.argv) != 2:
        print("Usage: python investigate.py <student_id>")
        print("Example: python investigate.py 10000409")
        sys.exit(1)

    student_id = sys.argv[1]
    investigator = SignatureInvestigator(
        StudentRepository(config.XML_PATH),
        AttendanceDatabase(config.DB_PATH),
    )
    investigator.investigate(student_id)


if __name__ == "__main__":
    main()
