import cv2

# ฟังก์ชันสำหรับวาดสี่เหลี่ยมตรงกลางจอ
def draw_center_rectangle(frame, eye_x, eye_y, rect_width=100, rect_height=100):
    height, width, _ = frame.shape
    center_x = int(width / 2)
    center_y = int(height / 2)
    top_left_x = center_x - int(rect_width / 2)
    top_left_y = center_y - int(rect_height / 2)
    bottom_right_x = center_x + int(rect_width / 2)
    bottom_right_y = center_y + int(rect_height / 2)
    if top_left_x < eye_x < bottom_right_x and top_left_y < eye_y < bottom_right_y:
        color = (0, 255, 0)  # เปลี่ยนสีเป็นเขียวเมื่อตาอยู่ในกรอบ
    else:
        color = (0, 0, 255)  # คงสีแดงเมื่อตาไม่อยู่ในกรอบ
    cv2.rectangle(frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), color, 2)

# เปิดกล้อง
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if ret is False:
        break

    # ข้อมูลตำแหน่งตาจากตัวอย่าง
    eye_x = 300
    eye_y = 200

    # วาดสี่เหลี่ยมตรงกลางจอ
    draw_center_rectangle(frame, eye_x, eye_y)

    # แสดงภาพ
    cv2.imshow("DETECT-BLACKEYE", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

# ปิดกล้องและปิดหน้าต่าง
cam.release()
cv2.destroyAllWindows()
