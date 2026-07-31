import cv2
from picamera2 import Picamera2
import os

name = 'Member_Name' 
dataset_path = "dataset/" + name

# Tu dong tao thu muc neu chua co de tranh loi khi luu anh
if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)

# Khoi tao camera bang Picamera2
picam2 = Picamera2()

# --- CAU HINH QUAN TRONG DE MAU SAC GIONG THUC TE ---
# Cau hinh do phan giai cho luong anh (640x480 la du net cho nhan dien)
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)

picam2.start()

# Ep he thong bat tu dong can bang trang (AWB) va tu dong phoi sang (AEC)
picam2.set_controls({
    "AwbEnable": True,
    "AeEnable": True
})
# ----------------------------------------------------

cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
cv2.resizeWindow("press space to take a photo", 500, 300)

img_counter = 0

print("[INFO] Camera dang chay... (Cho khoang 2 giay de AWB on dinh mau sac)")

while True:
    # Lay anh tu camera (Picamera2 tra ve mang Numpy)
    frame = picam2.capture_array()
    
    # Chuyen mau tu RGB sang BGR vi OpenCV mac dinh dung BGR
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Lat nguoc anh de de canh goc chup
    frame = cv2.flip(frame, 1)

    cv2.imshow("press space to take a photo", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        img_name = "{}/image_{}.jpg".format(dataset_path, img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

# 4. Giai phong tai nguyen
picam2.stop()
cv2.destroyAllWindows()