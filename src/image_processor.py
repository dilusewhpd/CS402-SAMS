import sys
import cv2


class ImageProcessor:
    def _init_(self, show_steps: bool = True):
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
        print("Progress: Applying grayscale conversion...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite("output_gray.jpg", gray)
        self._display("Step 2 - Grayscale Image", gray)
        return gray