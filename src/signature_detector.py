
import os
import cv2
import numpy as np
import pytesseract


class SignatureDetector:
    def __init__(self, coordinates: dict, ink_threshold_pct: float, blob_area_threshold: int):
        self.coordinates = coordinates
        self.ink_threshold_pct = ink_threshold_pct
        self.blob_area_threshold = blob_area_threshold

    def detect(self, binary, image_name: str) -> list[str]:
        """Returns a list of "Present"/"Absent" strings, one per cell, in row order."""
        cells = self.coordinates.get(image_name)
        if not cells:
            raise ValueError(f"No coordinates configured for {image_name}")

        os.makedirs("cells", exist_ok=True)
        print(f"Binary image shape: {binary.shape}")

        return [self._classify_cell(binary, box, i) for i, box in enumerate(cells)]

    def _classify_cell(self, binary, box: tuple, index: int) -> str:
        y1, y2, x1, x2 = box
        cell = binary[y1 + 10:y2 - 10, x1 + 10:x2 - 10]
        cell = self._remove_grid_lines(cell)
        cv2.imwrite(f"cells/cell_{index + 1}.jpg", cell)

        bridged = self._bridge_strokes(cell)
        ink_pct = self._ink_percentage(cell)
        largest_blob, blob_count = self._largest_blob(bridged)

        print(f"Student {index + 1}: ink%={ink_pct:.2f}%, "
              f"blobs={blob_count}, largest_blob={largest_blob}")

        status = "Present" if (largest_blob > self.blob_area_threshold
                                and ink_pct > self.ink_threshold_pct) else "Absent"

        if self._looks_like_ab_mark(cell):
            status = "Absent"

        print(f"Student {index + 1}: {status}")
        return status

    @staticmethod
    def _remove_grid_lines(cell):
        """Strips long near-horizontal lines (table borders) from a cell crop.
        Uses HoughLinesP rather than a strict horizontal kernel because phone
        photos are rarely perfectly level."""
        h, w = cell.shape
        out = cell.copy()
        lines = cv2.HoughLinesP(cell, 1, np.pi / 180, threshold=int(w * 0.4),
                                 minLineLength=int(w * 0.5), maxLineGap=10)
        if lines is not None:
            for l in lines:
                x1, y1, x2, y2 = l[0]
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                if angle < 10 or angle > 170:
                    cv2.line(out, (x1, y1), (x2, y2), 0, thickness=5)
        return out

    @staticmethod
    def _bridge_strokes(cell):
        """Dilates ink slightly so a thin/faint signature broken into several
        disconnected strokes is treated as one connected blob."""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
        return cv2.dilate(cell, kernel, iterations=1)

    @staticmethod
    def _ink_percentage(cell) -> float:
        ink_pixels = cv2.countNonZero(cell)
        area = cell.shape[0] * cell.shape[1]
        return (ink_pixels / area) * 100

    @staticmethod
    def _largest_blob(bridged) -> tuple[int, int]:
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
        if num_labels <= 1:
            return 0, 0
        areas = [stats[j, cv2.CC_STAT_AREA] for j in range(1, num_labels)]
        largest = max(areas)
        significant = sum(1 for a in areas if a > 50)
        return largest, significant

    @staticmethod
    def _looks_like_ab_mark(cell) -> bool:
        """OCRs the cell restricted to only 'a'/'b' characters (the only text
        convention we expect), since unrestricted OCR misreads handwritten
        "ab" as "ak" (b/k look similar in cursive at this size)."""
        inv = cv2.bitwise_not(cell)
        inv = cv2.resize(inv, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        text = pytesseract.image_to_string(
            inv, config="--psm 7 -c tessedit_char_whitelist=ab"
        ).lower()
        return "ab" in text