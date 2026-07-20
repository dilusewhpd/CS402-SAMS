import sys
import os
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from signature_detector import SignatureDetector

def test_signature_detector_init():
    detector = SignatureDetector(ink_threshold_pct=2.0, blob_area_threshold=800)
    assert detector.ink_threshold_pct == 2.0
    assert detector.blob_area_threshold == 800

def test_ink_percentage():
    detector = SignatureDetector(ink_threshold_pct=2.0, blob_area_threshold=800)
    # Create a 100x100 image (10000 pixels)
    img = np.zeros((100, 100), dtype=np.uint8)
    # Fill 10x10 with white (100 pixels) -> 1% ink
    img[0:10, 0:10] = 255
    pct = detector._ink_percentage(img)
    assert pct == 1.0

def test_largest_blob():
    detector = SignatureDetector(ink_threshold_pct=2.0, blob_area_threshold=800)
    img = np.zeros((100, 100), dtype=np.uint8)
    # Create two blobs
    img[10:20, 10:20] = 255 # 100 area
    img[50:70, 50:70] = 255 # 400 area
    largest, significant = detector._largest_blob(img)
    assert largest == 400
    assert significant == 2
