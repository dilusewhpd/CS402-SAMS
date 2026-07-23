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

    # Display resized image
display = cv2.resize(img, (800, 1000))
cv2.imshow("Click on signature cells to find coordinates", display)
cv2.setMouseCallback("Click on signature cells to find coordinates", mouse_click)

print("Instructions:")
print("1. The image window will open. ")
print("2. Click the TOP-LEFT corner of Student 1's signature box. ")
print("3. Click the BOTTOM-RIGHT corner of Student 1's signature box. ")
print("4. Repeat the same steps for all 6 students. ")
print("5. Press Q to close the window. ")  

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()