import cv2

def draw_red_box(frame):
    height, width, _ = frame.shape
    box_size = 150
    x = int((width - box_size) / 2)
    y = int((height - box_size) / 2)
    cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (0, 0, 255), 2)
    cv2.line(frame, (x + int(box_size/2), 0), (x + int(box_size/2), height), (0, 0, 255), 2)
    cv2.line(frame, (0, y + int(box_size/2)), (width, y + int(box_size/2)), (0, 0, 255), 2)
    return x, y, box_size


cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    x, y, box_size = draw_red_box(frame)
    roi = frame[y:y+box_size, x:x+box_size]    
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_roi = cv2.GaussianBlur(gray_roi, (7, 7), 0)
    black_threshold = 30
    black_mask = cv2.inRange(gray_roi, 0, black_threshold)
    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if   area >= 1000 and area <= 1500:            
            cv2.rectangle(roi, (x, y), (x+w, y+h), (0, 255, 0), 2)  
            cv2.putText(roi, 'Eye', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  
            print("Area : " , area)
            
            
    resized_frame = cv2.resize(frame, (1280, 720))   
    cv2.imshow("DETECT-BLACKEYE", resized_frame)
    

    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
