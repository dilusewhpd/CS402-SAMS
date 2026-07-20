# CS402-SAMS (Student Attendance Management System)

A fully automated, image-processing based Student Attendance Management System developed for the CS402.3 coursework. 

This project uses OpenCV to analyze smartphone photos of physical attendance sheets, dynamically crop student signatures, and track daily attendance in a database. It also includes data visualization and automated forgery investigation tools.

---

## 🛠️ Project Setup

### 1. Prerequisites
- **Python 3.7+** must be installed on your machine.
- **Tesseract OCR (Optional but recommended)**: Used for reading "ab" (absent) marks. 
  - Download from: [UB-Mannheim Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
  - Ensure it is installed at the default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - *Note: If you skip this, the project will safely fall back to ink-blob size checks.*

### 2. Install Dependencies
Open a terminal in the root of the project directory and run the following command to install all required libraries (OpenCV, Numpy, PyTesseract, Matplotlib, Pytest):

```bash
pip install -r requirements.txt
```
*(If you encounter permission errors, run `pip install --user -r requirements.txt`)*

---

## 🚀 How to Run the Project

The system is broken down into three main executable scripts to cover the entire coursework requirements pipeline.

### Step 1: Process Attendance Sheets (`sams.py`)
This script analyzes an image of an attendance sheet, determines which students signed in, and saves the data to the SQLite database.

**Command Format:**
```bash
python src/sams.py <image_path> <xml_path>
```

**Example Commands (Run these one by one):**
```bash
python src/sams.py data/images/1.jpeg data/info.xml
python src/sams.py data/images/2.jpeg data/info.xml
python src/sams.py data/images/3.jpeg data/info.xml
python src/sams.py data/images/4.jpeg data/info.xml
python src/sams.py data/images/5.jpeg data/info.xml
```
*Note: A popup window will appear during processing to show you the step-by-step image thresholds. **Press any key on your keyboard** to advance past these windows.*

---

### Step 2: Generate Attendance Visualizations (`infovis.py`)
Once you have processed the images in Step 1, you can generate a bar chart visualizing the attendance history of any specific student.

**Command Format:**
```bash
python src/infovis.py <student_id>
```

**Example Command:**
```bash
python src/infovis.py 10000409
```
*A Matplotlib window will open showing the chart, and the chart will also be saved locally as `attendance_chart.png`.*

---

### Step 3: Investigate Signatures (`investigate.py`)
This script cross-references the database and the original images to automatically extract and compare a student's signature across all the days they were marked present. It flags signatures that are suspiciously identical (indicating a photocopy forgery).

**Command Format:**
```bash
python src/investigate.py <student_id> <xml_path>
```

**Example Command:**
```bash
python src/investigate.py 10000409 data/info.xml
```

---

## 🧪 Running Automated Tests
The project includes a professional test suite using `pytest` to ensure core components (like the database models and the signature detector) are functioning correctly.

Run the test suite using:
```bash
pytest tests/
```

---

## 📂 Project Structure

```text
CS402-SAMS/
├── data/
│   ├── images/              # Raw JPEG/PNG signing sheets
│   └── info.xml             # XML database of enrolled students
├── db/
│   └── attendance.db        # Automatically generated SQLite database
├── src/
│   ├── sams.py              # Entry point: Image processing pipeline
│   ├── infovis.py           # Entry point: Data visualization
│   ├── investigate.py       # Entry point: Forgery investigation
│   ├── config.py            # Global thresholds and cell coordinates
│   ├── database.py          # SQLite database connection layer
│   ├── image_processor.py   # OpenCV binarization and thresholding
│   ├── models.py            # Data classes (AttendanceRecord)
│   ├── signature_detector.py# Signature ink blob analysis
│   └── student_repository.py# XML parsing logic
├── tests/                   # Pytest automated unit test suite
└── requirements.txt         # Project dependency list
```
