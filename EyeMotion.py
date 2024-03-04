import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    roi = frame[269:795, 537:1416]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 10, 60], dtype="uint8")  # ค่า HSV สำหรับสีน้ำตาลเข้ม
    upper = np.array([20, 150, 255], dtype="uint8")  # ค่า HSV สำหรับสีน้ำตาลเข้ม
    mask = cv2.inRange(hsv_roi, lower, upper)
    threshold = cv2.bitwise_and(gray_roi, gray_roi, mask=mask)
    contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

    rows, cols, _ = roi.shape  # สร้างตัวแปร rows และ cols จากขนาดของ roi

    for cnt in contour:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
        cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)

    cv2.imshow("GRAY", gray_roi)
    cv2.imshow("THRES", threshold)
    cv2.imshow("DETECT-EYE", roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
