import cv2
import numpy as np

selected_points = []

def select_point(event, x, y, flags, param):
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        print(f"Selected point: ({x}, {y})")
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image)

image = cv2.imread("resources/localization/nothing_2a/lane.png")

cv2.imshow("Image", image)
cv2.setMouseCallback("Image", select_point)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Selected points:", selected_points)