import os
import time
import threading
import pickle
import cv2
import numpy as np
import imutils
from imutils.video import FPS
import face_recognition
from picamera2 import Picamera2
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory

# ==========================================
# 1. HAM TINH TOAN MAT (EAR) CHONG GIA MAO
# ==========================================
def calculate_ear(eye):
    A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
    B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
    C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
    return (A + B) / (2.0 * C)

EYE_AR_CLOSED_THRESH = 0.22 # Nguong xac dinh da nham mat (ep phai nham that ro)
EYE_AR_OPEN_THRESH = 0.3   # Nguong xac dinh dang mo mat

# Bien cho State Machine (May trang thai)
# 0: Cho xac nhan mat mo
# 1: Dang cho ho nham mat de tinh gio
# 2: Xac thuc thanh cong
anti_spoof_state = 0 
eyes_closed_start_time = None 

# ==========================================
# 2. CAU HINH SERVO MG996R (HARDWARE PWM)
# ==========================================
SERVO_PIN = 17

print("[INFO] Dang khoi tao module pigpio cho Servo...")
try:
    factory = PiGPIOFactory()
    servo = Servo(SERVO_PIN, pin_factory=factory, min_pulse_width=0.0005, max_pulse_width=0.0025)
except Exception as e:
    print("[ERROR] Khong the khoi tao pigpio! Hay mo terminal va chay 'sudo pigpiod'.")
    exit()

def set_servo_angle(action):
    def move_servo():
        if action == "lock":
            servo.min()  # Xoay ve 0 do
        elif action == "unlock":
            servo.mid()  # Xoay ve 90 do
        time.sleep(0.5)  # Thoi gian cho servo chay het hanh trinh
        servo.detach()   # Ngat xung dien, chong e e va chay motor
    threading.Thread(target=move_servo, daemon=True).start()

print("[INFO] Thiet lap Servo ve trang thai mac dinh (KHOA)...")
set_servo_angle("lock") 

# ==========================================
# 3. KHOI TAO BIEN VA DU LIEU NHAN DIEN
# ==========================================
currentname = "unknown"
encodingsP = "encodings.pickle"
TOLERANCE = 0.45 
doorUnlock = False
prevTime = 0

print("[INFO] Dang tai du lieu encodings...")
try:
    data = pickle.loads(open(encodingsP, "rb").read())
except FileNotFoundError:
    print("[ERROR] Khong tim thay file encodings.pickle! Hay chay file train truoc.")
    exit()

# ==========================================
# 4. CAU HINH CAMERA OV5647
# ==========================================
print("[INFO] Dang khoi dong camera...")
picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "BGR888"})
picam2.configure(config)
picam2.start()

picam2.set_controls({"AwbEnable": True, "AeEnable": True})
time.sleep(2.0) 

fps = FPS().start()

process_this_frame = 0
dynamic_skip = 4 # Toc do bo qua khung hinh mac dinh

AI_PROCESS_WIDTH = 240 
scale = 640 / AI_PROCESS_WIDTH  
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# ==========================================
# 5. VONG LAP CHINH
# ==========================================
while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.flip(frame, 1)

    if process_this_frame % (dynamic_skip + 1) == 0:
        small_frame = imutils.resize(frame, width=AI_PROCESS_WIDTH)
        
        # Loc chong nguoc sang
        ycrcb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = clahe.apply(y) 
        small_frame_clahe = cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

        rgb_small = cv2.cvtColor(small_frame_clahe, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_small, model="hog")
        
        waiting_for_blink = False 
        
        if len(boxes) > 0:
            encodings = face_recognition.face_encodings(rgb_small, boxes)
            names = []
            
            for i, encoding in enumerate(encodings):
                face_distances = face_recognition.face_distance(data["encodings"], encoding)
                original_name = "Unknown"
                display_name = "Unknown"
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    
                    if face_distances[best_match_index] < TOLERANCE:
                        original_name = data["names"][best_match_index]
                        display_name = original_name
                        
                        # --- THUAT TOAN CHONG GIA MAO (STATE MACHINE) ---
                        if not doorUnlock:
                            waiting_for_blink = True 
                            
                            landmarks_list = face_recognition.face_landmarks(rgb_small, [boxes[i]])
                            
                            if len(landmarks_list) > 0:
                                landmarks = landmarks_list[0]
                                leftEAR = calculate_ear(landmarks['left_eye'])
                                rightEAR = calculate_ear(landmarks['right_eye'])
                                ear = (leftEAR + rightEAR) / 2.0
                                
                                # TRANG THAI 0: Bat buoc phai thay mat dang mo
                                if anti_spoof_state == 0:
                                    if ear > EYE_AR_OPEN_THRESH:
                                        anti_spoof_state = 1 # Chuyen sang trang thai cho nham mat
                                        display_name = f"{original_name} (Da xac nhan mat mo)"
                                    else:
                                        display_name = f"{original_name} (Vui long mo mat truoc!)"
                                        
                                # TRANG THAI 1: Yeu cau nham mat du 2 giay
                                elif anti_spoof_state == 1:
                                    if ear < EYE_AR_CLOSED_THRESH:
                                        if eyes_closed_start_time is None:
                                            eyes_closed_start_time = time.time()
                                        
                                        closed_duration = time.time() - eyes_closed_start_time
                                        display_name = f"{original_name} (Dang giu: {closed_duration:.1f}s)"
                                        
                                        # Du 2.0 giay -> Thanh cong
                                        if closed_duration >= 2.0:
                                            anti_spoof_state = 2
                                    else:
                                        # Ho mo mat ra truoc khi du 2 giay
                                        eyes_closed_start_time = None
                                        display_name = f"{original_name} (Hay nham mat 2s de mo!)"
                                
                                # TRANG THAI 2: Mo khoa
                                if anti_spoof_state == 2:
                                    set_servo_angle("unlock")
                                    doorUnlock = True
                                    print(f">>> XAC THUC NGUOI THAT THANH CONG. KHOA MO CHO: {original_name}")
                                    prevTime = time.time()
                                    anti_spoof_state = 0
                                    eyes_closed_start_time = None 
                        else:
                            # Cua dang mo, reset trang thai
                            anti_spoof_state = 0
                            eyes_closed_start_time = None
                            prevTime = time.time()
                
                names.append(display_name)
                
                if original_name != "Unknown" and currentname != original_name:
                    currentname = original_name
                    print(f"[INFO] Phat hien nguoi quen: {currentname}")
        else:
            names = [] 
            anti_spoof_state = 0 # Khong thay ai thi reset luon
            eyes_closed_start_time = None

        # --- TINH NANG TANG TOC KHUNG HINH KHI CHO XAC THUC ---
        if waiting_for_blink:
            dynamic_skip = 0 
        else:
            dynamic_skip = 4 

    process_this_frame += 1

    # ==========================================
    # 6. LOGIC KHOA CUA TU DONG
    # ==========================================
    if doorUnlock and (time.time() - prevTime > 5):
        doorUnlock = False
        set_servo_angle("lock") 
        currentname = "unknown" 
        anti_spoof_state = 0
        eyes_closed_start_time = None 
        print(">>> CUA DA KHOA TU DONG")

    # ==========================================
    # 7. VE KET QUA LEN MAN HINH
    # ==========================================
    for ((top, right, bottom, left), name) in zip(boxes, names):
        top = int(top * scale)
        right = int(right * scale)
        bottom = int(bottom * scale)
        left = int(left * scale)
        
        # Doi mau khung tuy theo trang thai chong gia mao
        if name == "Unknown":
            color = (0, 0, 255) # Do
        elif "(Vui long mo" in name or "(Hay nham" in name or "(Da xac nhan" in name:
            color = (0, 255, 255) # Vang (Dang cho thuc hien lenh)
        elif "(Dang giu" in name:
            color = (255, 165, 0) # Cam (Dang bam gio)
        else:
            color = (0, 255, 0) # Xanh la (Da mo cua thanh cong)
            
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        y_text = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imshow("He Thong Nhan Dien Nguoi That", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    fps.update()

# ==========================================
# 8. DON DEP TAI NGUYEN
# ==========================================
fps.stop()
print("="*40)
print(f"[INFO] Tong thoi gian chay: {fps.elapsed():.2f} giay")
print(f"[INFO] Toc do khung hinh trung binh (FPS): {fps.fps():.2f}")
print("="*40)

cv2.destroyAllWindows()
picam2.stop()
