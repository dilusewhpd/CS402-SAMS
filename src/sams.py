import cv2
import numpy as np
import xml.etree.ElementTree as ET
import sqlite3
import sys
import os

def load_image(image_path):
    # Load the image from the given path
    img = cv2.imread(image_path)
    if img is None:
        print("ERROR: Image not found! Please check the path.")
        sys.exit(1)
    print(f"Image loaded successfully. Size: {img.shape}")
    
    # Resize for display only (original used for processing)
    display = cv2.resize(img, (800, 1000))
    cv2.imshow("Step 1 - Original Image", display)
    cv2.waitKey(0)
    return img

def convert_grayscale(img):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("output_gray.jpg", gray)
    
    # Resize for display only
    display = cv2.resize(gray, (800, 1000))
    cv2.imshow("Step 2 - Grayscale Image", display)
    cv2.waitKey(0)
    return gray

def binarize(gray):
    # Apply Otsu thresholding to convert grayscale to binary image
    # Black pixels represent ink, white pixels represent background
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    cv2.imwrite("output_binary.jpg", binary)
    
    # Resize for display only
    display = cv2.resize(binary, (800, 1000))
    cv2.imshow("Step 3 - Binarized Image", display)
    cv2.waitKey(0)
    return binary

def detect_signatures(binary):
    # Print binary image dimensions for debugging
    print(f"Binary image shape: {binary.shape}")

    # Define pixel coordinates for each student signature cell
    # Format: (y_start, y_end, x_start, x_end)
    signature_cells = [
        (1600, 1720, 2200, 3000),  # Student 1
        (1720, 1840, 2200, 3000),  # Student 2
        (1840, 1960, 2200, 3000),  # Student 3
        (1960, 2080, 2200, 3000),  # Student 4
        (2080, 2200, 2200, 3000),  # Student 5
        (2200, 2320, 2200, 3000),  # Student 6
    ]

    # Save a test cell image for debugging
    test = binary[1600:1720, 2200:3000]
    cv2.imwrite("test_cell.jpg", test)
    print("Test cell image saved as test_cell.jpg")

    results = []
    for i, (y1, y2, x1, x2) in enumerate(signature_cells):
        # Crop the signature cell from the binary image
        cell = binary[y1:y2, x1:x2]
        # Count ink (non-zero) pixels in the cell
        ink_pixels = cv2.countNonZero(cell)
        # If ink pixels exceed threshold, student is present
        status = "Present" if ink_pixels > 100 else "Absent"
        results.append(status)
        print(f"Student {i+1}: {status} (ink pixels: {ink_pixels})")

    return results

def parse_xml(xml_path):
    # Parse the XML file to extract student information
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

def save_to_db(students, results, date):
    # Create database directory if it does not exist
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect("db/attendance.db")
    cursor = conn.cursor()

    # Create attendance table if it does not exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
        student_id TEXT,
        name TEXT,
        date TEXT,
        status TEXT
    )''')

    # Insert attendance records for each student
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
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python sams.py <image_path> <date>")
        print("Example: python sams.py data/images/1.jpeg 12-07-2019")
        sys.exit(1)

    image_path = sys.argv[1]
    date = sys.argv[2]
    xml_path = "data/info.xml"

    print(f"Processing image: {image_path}")
    print(f"Attendance date: {date}")

    # Run all processing steps
    img = load_image(image_path)
    gray = convert_grayscale(img)
    binary = binarize(gray)
    results = detect_signatures(binary)
    students = parse_xml(xml_path)
    save_to_db(students, results, date)

    cv2.destroyAllWindows()
    print("Program completed successfully!")