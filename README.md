# CS402-SAMS
Student Attendance Management System

## Usage

1. Process a signing sheet and save attendance records:
   ```bash
   python src/sams.py data/images/1.jpeg
   ```

2. View attendance visualization for a student:
   ```bash
   python src/infovis.py 10000409
   ```

3. Investigate signature consistency for a student across sheets:
   ```bash
   python src/investigate.py 10000409
   ```

## Files

- `src/sams.py`: main pipeline to process sheet images and save attendance.
- `src/infovis.py`: generates a bar chart of attendance for a student.
- `src/investigate.py`: compares the same student's signatures across sheets.
- `data/info.xml`: student roster and metadata.
- `data/images/1.jpeg` ... `5.jpeg`: signing sheet images.
- `db/attendance.db`: SQLite attendance records.
