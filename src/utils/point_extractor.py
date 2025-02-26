import cv2
import numpy as np

selected_points = []
scale = 0.5
upscale = 1/scale

def select_point(event, x, y, flags, param):
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Image", image)
        x = x*upscale
        y = y*upscale
        selected_points.append([x, y])
        print(f"Selected point: ({x}, {y})")

camera = "lumix"
image = cv2.imread(f"resources/localization/{camera}/lane.png")

print(image.shape)
image = cv2.resize(image, None, fx=scale, fy=scale)
print(image.shape)
cv2.imshow("Image", image)
cv2.setMouseCallback("Image", select_point)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Selected points:", selected_points)