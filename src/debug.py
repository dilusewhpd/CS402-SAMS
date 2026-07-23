import cv2

# Load image
img = cv2.imread("data/images/2.jpeg")
print(f"Image size: {img.shape}")  # height, width, channels

# Draw grid lines to find correct coordinates
def mouse_click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Clicked position - X: {x}, Y: {y}")
        # Convert display coordinates to original coordinates
        orig_x = int(x * (img.shape[1] / 800))
        orig_y = int(y * (img.shape[0] / 1000))
        print(f"Original image position - X: {orig_x}, Y: {orig_y}")