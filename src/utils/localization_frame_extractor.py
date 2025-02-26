import cv2 as cv 

video="opt_7.MP4"
camera = "nothing_2a"
video_path = f"resources/video/{camera}/{video}"
out_path = f"resources/localization/{camera}/lane.png"

capture = cv.VideoCapture(video_path)
while(capture.isOpened()):
    ret, frame = capture.read()
    cv.imshow('frame', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        res = frame
        break

capture.release()
cv.destroyAllWindows()

cv.imshow('Resulting Frame', res)
if cv.waitKey(0) & 0xFF == ord('s'):
    print(f"Saving the frame into {out_path}")
    cv.imwrite(out_path, res)
else : 
    print("File not saved")
