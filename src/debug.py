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
print("1. Image window open වෙනවා")
print("2. Student 1 signature box TOP-LEFT corner click කරන්න")
print("3. Student 1 signature box BOTTOM-RIGHT corner click කරන්න")
print("4. ඒ විදිහටම students 6 දෙනාටම click කරන්න")
print("5. Q press කරන්න close කරන්න")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()