import os
import cv2
import numpy as np


OUTPUT_DIR = os.path.join("output", "detected_tables")

MIN_H_LINE_RATIO = 1 / 25
MIN_V_LINE_RATIO = 1 / 25
MIN_LINE_LENGTH = 20
MIN_TABLE_AREA_RATIO = 0.02


class TableNotFoundError(ValueError):
    pass


def check_image(img):
    if img is None:
        raise ValueError("image is None")
    if not isinstance(img, np.ndarray) or img.size == 0:
        raise ValueError("image is empty")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _ink_mask(binary_img):
    return cv2.bitwise_not(binary_img)


def detect_horizontal_lines(binary_img, min_length_ratio=MIN_H_LINE_RATIO):
    binary_img = check_image(binary_img)
    height, width = binary_img.shape

    ink = _ink_mask(binary_img)
    kernel_len = max(MIN_LINE_LENGTH, int(width * min_length_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)

    return cv2.bitwise_not(horizontal)


def detect_vertical_lines(binary_img, min_length_ratio=MIN_V_LINE_RATIO):
    binary_img = check_image(binary_img)
    height, width = binary_img.shape

    ink = _ink_mask(binary_img)
    kernel_len = max(MIN_LINE_LENGTH, int(height * min_length_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)

    return cv2.bitwise_not(vertical)


def combine_grid_lines(horizontal_img, vertical_img):
    horizontal_img = check_image(horizontal_img)
    vertical_img = check_image(vertical_img)
    if horizontal_img.shape != vertical_img.shape:
        raise ValueError("horizontal and vertical images must be the same size")

    h_ink = _ink_mask(horizontal_img)
    v_ink = _ink_mask(vertical_img)
    combined_ink = cv2.bitwise_or(h_ink, v_ink)
    combined_ink = cv2.dilate(combined_ink, np.ones((3, 3), np.uint8))
    return cv2.bitwise_not(combined_ink)


def detect_table_boundary(grid_img, min_area_ratio=MIN_TABLE_AREA_RATIO):
    grid_img = check_image(grid_img)
    height, width = grid_img.shape
    image_area = float(height * width)

    ink = _ink_mask(grid_img)
    ink = cv2.dilate(ink, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise TableNotFoundError(
            "No attendance table found: no grid lines detected in the image.")

    largest = max(contours, key=cv2.contourArea)
    area_ratio = cv2.contourArea(largest) / image_area
    if area_ratio < min_area_ratio:
        raise TableNotFoundError(
            f"No attendance table found: largest grid region only covers "
            f"{area_ratio:.1%} of the image (need at least {min_area_ratio:.0%}).")

    x, y, w, h = cv2.boundingRect(largest)
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    return x, y, w, h


def _line_positions(line_ink, axis, min_gap):
    projection = line_ink.sum(axis=1 - axis).astype(np.float64)
    threshold = projection.max() * 0.4
    if threshold <= 0:
        return []

    on_line = projection >= threshold
    positions = []
    start = None
    for i, val in enumerate(on_line):
        if val and start is None:
            start = i
        elif not val and start is not None:
            positions.append((start + i - 1) // 2)
            start = None
    if start is not None:
        positions.append((start + len(on_line) - 1) // 2)

    merged = []
    for pos in positions:
        if merged and pos - merged[-1] < min_gap:
            continue
        merged.append(pos)
    return merged


def detect_row_boundaries(horizontal_img, table_bbox):
    horizontal_img = check_image(horizontal_img)
    x, y, w, h = table_bbox
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid table bounding box: {table_bbox}")

    cropped = horizontal_img[y:y + h, x:x + w]
    ink = _ink_mask(cropped)

    line_ys = _line_positions(ink, axis=0, min_gap=max(5, h // 50))
    if len(line_ys) < 2:
        raise TableNotFoundError(
            "Could not find enough horizontal lines to determine table rows.")

    line_ys = sorted(line_ys)
    rows = []
    for top, bottom in zip(line_ys[:-1], line_ys[1:]):
        rows.append((top + y, bottom + y))
    return rows


def detect_signature_column(vertical_img, table_bbox):
    vertical_img = check_image(vertical_img)
    x, y, w, h = table_bbox
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid table bounding box: {table_bbox}")

    cropped = vertical_img[y:y + h, x:x + w]
    ink = _ink_mask(cropped)

    line_xs = _line_positions(ink, axis=1, min_gap=max(5, w // 50))
    if len(line_xs) < 2:
        raise TableNotFoundError(
            "Could not find enough vertical lines to determine the signature column.")

    line_xs = sorted(line_xs)
    col_left, col_right = line_xs[-2], line_xs[-1]
    return col_left + x, col_right + x


def process_table_detection(binary_image, source_name="table"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("[1/4] Detecting horizontal lines")
    horizontal = detect_horizontal_lines(binary_image)

    print("[2/4] Detecting vertical lines")
    vertical = detect_vertical_lines(binary_image)

    print("[3/4] Combining grid lines and locating the table")
    grid = combine_grid_lines(horizontal, vertical)
    bbox = detect_table_boundary(grid)

    print("[4/4] Detecting row boundaries and the signature column")
    rows = detect_row_boundaries(horizontal, bbox)
    signature_column = detect_signature_column(vertical, bbox)

    table_preview = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR) \
        if binary_image.ndim == 2 else binary_image.copy()
    x, y, w, h = bbox
    cv2.rectangle(table_preview, (x, y), (x + w, y + h), (0, 255, 0), 3)
    for top, bottom in rows:
        cv2.line(table_preview, (x, top), (x + w, top), (255, 0, 0), 1)
        cv2.line(table_preview, (x, bottom), (x + w, bottom), (255, 0, 0), 1)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{source_name}_horizontal_lines.png"), horizontal)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{source_name}_vertical_lines.png"), vertical)
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{source_name}_detected_table.png"), table_preview)
    print(f"Saved horizontal/vertical/table images to {OUTPUT_DIR}")

    return {
        "table_boundary": bbox,
        "rows": rows,
        "signature_column": signature_column,
    }


def make_test_table_image(rows=4, cols=3, cell_w=140, cell_h=60, margin=40):
    width = margin * 2 + cell_w * cols
    height = margin * 2 + cell_h * rows
    img = np.full((height, width), 255, dtype=np.uint8)

    for r in range(rows + 1):
        y = margin + r * cell_h
        cv2.line(img, (margin, y), (margin + cell_w * cols, y), 0, 2)
    for c in range(cols + 1):
        x = margin + c * cell_w
        cv2.line(img, (x, margin), (x, margin + cell_h * rows), 0, 2)

    cv2.putText(img, "sign", (margin + cell_w * (cols - 1) + 15, margin + 40),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, 0, 2)
    return img


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        from thresholding import otsu_threshold
        gray = cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)
        binary = otsu_threshold(gray)
        name = os.path.splitext(os.path.basename(sys.argv[1]))[0]
    else:
        binary = make_test_table_image()
        name = "demo"
        print("no image given, using a synthetic test table image instead")

    result = process_table_detection(binary, source_name=name)
    print("table boundary:", result["table_boundary"])
    print("rows:", result["rows"])
    print("signature column:", result["signature_column"])

def _select_student_rows(rows: list[tuple[int, int]], student_count: int) -> list[tuple[int, int]]:
    """Select exactly the student rows from Member 4's raw row detections.

    The supplied sheets can include the lecturer table/header and some signatures
    create false horizontal lines. Student rows are the lower block of similarly
    sized rows. Tiny split rows are merged until exactly student_count rows remain.
    """
    if student_count < 1:
        raise ValueError("XML must contain at least one student.")
    if len(rows) < student_count:
        raise ValueError(
            f"Detected only {len(rows)} row regions, but XML contains {student_count} students."
        )

    rows = sorted((int(t), int(b)) for t, b in rows if int(b) > int(t))
    heights = np.array([b - t for t, b in rows], dtype=float)
    median_h = float(np.median(heights))
    normal_heights = heights[heights >= max(4.0, median_h * 0.75)]
    typical_h = float(np.median(normal_heights)) if normal_heights.size else median_h

    bottom = rows[-1][1]
    target_top = bottom - (student_count * typical_h)
    start_idx = min(range(len(rows)), key=lambda i: abs(rows[i][0] - target_top))
    selected = rows[start_idx:]

    # If we still have too many segments, merge the adjacent pair whose combined
    # height is closest to a normal student-row height. This repairs rows split by
    # a false horizontal line from handwriting.
    while len(selected) > student_count:
        best_i = None
        best_cost = None
        for i in range(len(selected) - 1):
            combined_h = selected[i + 1][1] - selected[i][0]
            gap = selected[i + 1][0] - selected[i][1]
            cost = abs(combined_h - typical_h) + abs(gap) * 3
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_i = i
        i = best_i
        selected[i:i + 2] = [(selected[i][0], selected[i + 1][1])]

    # If nearest-top selection produced too few, move the start upward.
    while len(selected) < student_count and start_idx > 0:
        start_idx -= 1
        selected.insert(0, rows[start_idx])

    if len(selected) != student_count:
        raise ValueError(
            f"Could not isolate exactly {student_count} student rows from {len(rows)} raw rows."
        )

    return selected


