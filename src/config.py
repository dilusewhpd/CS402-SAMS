# Path to the Tesseract-OCR executable (Windows). Adjust if installed elsewhere.
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

DB_PATH = "db/attendance.db"
XML_PATH = "data/info.xml"

# Decision thresholds used by SignatureDetector
INK_THRESHOLD_PERCENT = 2
BLOB_AREA_THRESHOLD = 800

# Each signing sheet photo has its signature cells at different pixel
# coordinates (y1, y2, x1, x2), because each photo was taken independently.
CELL_COORDINATES = {
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

# Actual date printed on each signing sheet's header (read by hand once).
IMAGE_DATES = {
    "1.jpeg": "31-05-2019",
    "2.jpeg": "21-06-2019",
    "3.jpeg": "28-06-2019",
    "4.jpeg": "05-07-2019",
    "5.jpeg": "12-07-2019",
}