import cv2
import numpy as np
import xml.etree.ElementTree as ET
import sqlite3
import sys
import os
import pytesseract   
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

#load image 
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Image not found! Please check the path.")
        sys.exit(1)
    print(f"Image loaded successfully. Size: {img.shape}")
    display = cv2.resize(img, (800, 1000))
    cv2.imshow("Step 1 - Original Image", display)
    cv2.waitKey(0)
    return img

#convert to grayscale
def convert_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output_gray.jpg", gray)
    display = cv2.resize(gray, (800, 1000))
    cv2.imshow("Step 2 - Grayscale Image", display)
    cv2.waitKey(0)
    return gray

#binarize the image
def binarize(gray):
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    cv2.imwrite("output_binary.jpg", binary)
    display = cv2.resize(binary, (800, 1000))
    cv2.imshow("Step 3 - Binarized Image", display)
    cv2.waitKey(0)
    return binary

#remove grid lines from the cell
def remove_grid_lines(cell):
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

#check if the cell contains "ab" mark using OCR
def looks_like_ab_mark(cell_no_lines):
    inv = cv2.bitwise_not(cell_no_lines)
    inv = cv2.resize(inv, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(
        inv, config="--psm 7 -c tessedit_char_whitelist=ab"
    ).lower()
    return "ab" in text

IMAGE_DATES = {
    "1.jpeg": "31-05-2019",
    "2.jpeg": "21-06-2019",
    "3.jpeg": "28-06-2019",
    "4.jpeg": "05-07-2019",
    "5.jpeg": "12-07-2019",
}

#def get_date_for_image(image_path):
def get_date_for_image(image_path):
    image_name = os.path.basename(image_path)
    date = IMAGE_DATES.get(image_name)
    if not date or date.startswith("TODO"):
        print(f"ERROR: No date set for {image_name} in IMAGE_DATES. "
              f"Open the sheet, read the date in the header, and add it to the dict.")
        sys.exit(1)
    return date

#detect signatures in the binary image and determine attendance
def detect_signatures(binary, image_path):
    print(f"Binary image shape: {binary.shape}")

    coordinates = {
        "1.jpeg": [
            (1439, 1568, 2113, 2555),
            (1673, 1770, 2105, 2555),
            (1761, 1854, 2097, 2555),
            (1842, 1943, 2101, 2547),
            (1935, 2028, 2101, 2547),
            (2024, 2116, 2097, 2547),
        ],
        "2.jpeg": [
            (1419, 1483, 1965, 2400),
            (1507, 1564, 1973, 2407),
            (1588, 1649, 1976, 2407),
            (1665, 1729, 1973, 2415),
            (1720, 1820, 1950, 2426),
            (1820, 1920, 1950, 2426),
        ],
        "3.jpeg": [
            (1528, 1588, 1920, 2343),
            (1604, 1681, 1920, 2354),
            (1689, 1761, 1920, 2362),
            (1766, 1838, 1920, 2358),
            (1858, 1927, 1924, 2362),
            (1931, 2016, 1935, 2362),
        ],
        "4.jpeg": [
            (1584, 1669, 1905, 2351),
            (1673, 1745, 1912, 2358),
            (1757, 1834, 1908, 2358),
            (1842, 1923, 1912, 2362),
            (1923, 1995, 1920, 2358),
            (2007, 2080, 1924, 2362),
        ],
        "5.jpeg": [
            (1572, 1645, 2060, 2521),
            (1665, 1733, 2071, 2513),
            (1745, 1822, 2075, 2525),
            (1834, 1911, 2071, 2525),
            (1923, 1999, 2075, 2536),
            (2007, 2084, 2082, 2540),
        ],
    }

    image_name = os.path.basename(image_path)
    signature_cells = coordinates.get(image_name)
    if not signature_cells:
        print(f"ERROR: No coordinates found for {image_name}")
        sys.exit(1)

    os.makedirs("cells", exist_ok=True)
    results = []

    for i, (y1, y2, x1, x2) in enumerate(signature_cells):
        cell = binary[y1 + 10:y2 - 10, x1 + 10:x2 - 10]
        cell = remove_grid_lines(cell)
        cv2.imwrite(f"cells/cell_{i+1}.jpg", cell)

        bridge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 3))
        bridged = cv2.dilate(cell, bridge_kernel, iterations=1)

        ink_pixels = cv2.countNonZero(cell)
        cell_area = cell.shape[0] * cell.shape[1]
        ink_percentage = (ink_pixels / cell_area) * 100

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bridged, connectivity=8)
        if num_labels > 1:
            largest_blob = max(stats[j, cv2.CC_STAT_AREA] for j in range(1, num_labels))
        else:
            largest_blob = 0
        significant_blobs = sum(1 for j in range(1, num_labels) if stats[j, cv2.CC_STAT_AREA] > 50)

        print(f"Student {i+1}: ink%={ink_percentage:.2f}%, "
              f"blobs={significant_blobs}, largest_blob={largest_blob}")

        if largest_blob > 800 and ink_percentage > 2:
            status = "Present"
        else:
            status = "Absent"

        if looks_like_ab_mark(cell):
            status = "Absent"

        results.append(status)
        print(f"Student {i+1}: {status}")

    return results

#parse XML file to get student info
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    students = []
    for s in root.findall("student"):
        students.append({
            "no": s.find("no").text,
            "id": s.find("id").text,
            "name": s.find("name").text
        })
    print(f"Total students loaded from XML: {len(students)}")
    return students

#save attendance results to SQLite database
def save_to_db(students, results, date):
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect("db/attendance.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
        student_id TEXT,
        name TEXT,
        date TEXT,
        status TEXT
    )''')
    cursor.execute("DELETE FROM attendance WHERE date=?", (date,))
    for i, student in enumerate(students):
        cursor.execute(
            "INSERT INTO attendance VALUES (?, ?, ?, ?)",
            (student["id"], student["name"], date, results[i])
        )
        print(f"Saved: {student['name']} - {results[i]} - {date}")
    conn.commit()
    conn.close()
    print("Attendance records saved to database successfully!")

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print("Usage: python sams.py <image_path> [date]")
        print("Example: python sams.py data/images/2.jpeg")
        print("(date is optional - if omitted, it's looked up from IMAGE_DATES)")
        sys.exit(1)

    image_path = sys.argv[1]
    # CLI date argument still works if you want to override; otherwise it's
    # looked up automatically from the IMAGE_DATES table above.
    date = sys.argv[2] if len(sys.argv) == 3 else get_date_for_image(image_path)
    xml_path = "data/info.xml"

    print(f"Processing image: {image_path}")
    print(f"Attendance date: {date}")

    img = load_image(image_path)
    gray = convert_grayscale(img)
    binary = binarize(gray)
    results = detect_signatures(binary, image_path)
    students = parse_xml(xml_path)
    save_to_db(students, results, date)

    cv2.destroyAllWindows()
    print("Program completed successfully!")