import cv2 as cv

selected_points = []
scale = 0.5
upscale = 1/scale

def select_point(event, x, y, flags, param):
    global selected_points
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(image, (x, y), 5, (0, 0, 255), -1)
        cv.imshow("Image", image)
        x = x*upscale
        y = y*upscale
        selected_points.append([x, y])
        print(f"Selected point: ({x}, {y})")

camera = "lumix"
image = cv.imread(f"resources/localization/{camera}/lane.png")

print(image.shape)
image = cv.resize(image, None, fx=scale, fy=scale)
print(image.shape)
cv.imshow("Image", image)
cv.setMouseCallback("Image", select_point)

cv.waitKey(0)
cv.destroyAllWindows()

print("Selected points:", selected_points)