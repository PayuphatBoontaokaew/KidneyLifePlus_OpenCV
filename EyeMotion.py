import cv2

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if ret is False:
        break

    roi = frame[269:795, 537:1416]
    rows, cols, _ = roi.shape
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    black_threshold = 30
    black_mask = cv2.inRange(gray_roi, 0, black_threshold)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if 3000 < area < 3500:            
            cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.line(roi, (x + int(w/2), 0), (x + int(w/2), rows), (0, 255, 0), 2)
            cv2.line(roi, (0, y + int(h/2)), (cols, y + int(h/2)), (0, 255, 0), 2)
            # print("Area :",area)
            


    cv2.imshow("DETECT-BLACKEYE", roi)

    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
