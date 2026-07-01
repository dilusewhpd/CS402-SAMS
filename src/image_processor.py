import sys
import cv2


class ImageProcessor:
    def __init__(self, show_steps: bool = True):
        self.show_steps = show_steps

    def load(self, image_path: str):
        img = cv2.imread(image_path)
        if img is None:
            print("ERROR: Image not found! Please check the path.")
            sys.exit(1)
        print(f"Image loaded successfully. Size: {img.shape}")
        self._display("Step 1 - Original Image", img)
        return img

    def to_grayscale(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("output_gray.jpg", gray)
        self._display("Step 2 - Grayscale Image", gray)
        return gray

    def binarize(self, gray):
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        cv2.imwrite("output_binary.jpg", binary)
        self._display("Step 3 - Binarized Image", binary)
        return binary

    def process(self, image_path: str):
        """Runs the full load -> grayscale -> binarize pipeline."""
        img = self.load(image_path)
        gray = self.to_grayscale(img)
        return self.binarize(gray)

    def _display(self, title: str, img):
        if not self.show_steps:
            return
        display = cv2.resize(img, (800, 1000))
        cv2.imshow(title, display)
        cv2.waitKey(0)