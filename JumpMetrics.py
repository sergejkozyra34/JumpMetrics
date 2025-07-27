import cv2
import mediapipe as mp
import numpy as np
import math
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

USER_HEIGHT = 1.75

def calculate_distance(shoulder_width_px, focal_length=1000):
    avg_shoulder_width = 0.4
    distance = (avg_shoulder_width * focal_length) / shoulder_width_px
    return distance

def calculate_jump_height(start_height, max_height, pixel_per_meter):
    return (start_height - max_height) / pixel_per_meter

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

calibrated = False
measuring = False
jump_detected = False
jump_start_time = 0
jump_timeout = 3.0

positions = []
timestamps = []
pixel_per_meter = 0
start_height = 0
jump_height = 0
jump_length = 0
distance_to_user = 0

CALIBRATION_COLOR = (0, 255, 255)
READY_COLOR = (0, 255, 0)
JUMP_COLOR = (0, 0, 255)
RESULT_COLOR = (255, 0, 0)
TRAJECTORY_COLOR = (0, 165, 255)

print("Инструкция:")
print("1. Встаньте в полный рост перед камерой на расстоянии 2-3 метров")
print("2. Нажмите 'c' для калибровки")
print("3. Приготовьтесь к прыжку")
print("4. Нажмите 's' чтобы начать измерение прыжка")
print("5. Совершите прыжок")
print("6. Нажмите 'q' для выхода")

cv2.namedWindow('Jump Measurement System', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Jump Measurement System', 1280, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    
    display_frame = frame.copy()
    h, w, _ = frame.shape
    
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
        right_ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
        
        shoulder_mid = (
            int((left_shoulder.x + right_shoulder.x) * 0.5 * w),
            int((left_shoulder.y + right_shoulder.y) * 0.5 * h)
        
        ankle_mid = (
            int((left_ankle.x + right_ankle.x) * 0.5 * w),
            int((left_ankle.y + right_ankle.y) * 0.5 * h)
        
        shoulder_width_px = abs(left_shoulder.x - right_shoulder.x) * w
        
        if not calibrated:
            distance_to_user = calculate_distance(shoulder_width_px)
            head = landmarks[mp_pose.PoseLandmark.NOSE.value]
            height_px = abs(head.y * h - ankle_mid[1])
            pixel_per_meter = height_px / USER_HEIGHT
            
            cv2.putText(display_frame, f"Calibrating... Distance: {distance_to_user:.2f}m", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CALIBRATION_COLOR, 2)
            cv2.putText(display_frame, f"Press 's' when ready to jump", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CALIBRATION_COLOR, 2)
        
        if measuring:
            if start_height == 0:
                start_height = ankle_mid[1]
                jump_start_time = time.time()
            
            positions.append(ankle_mid)
            timestamps.append(time.time())
            
            current_height = ankle_mid[1]
            if current_height < start_height - 0.05 * pixel_per_meter:
                jump_detected = True
                cv2.putText(display_frame, "JUMPING!", (w//2-100, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, JUMP_COLOR, 3)
            
            if jump_detected:
                if current_height > start_height - 0.02 * pixel_per_meter:
                    min_height = min(pos[1] for pos in positions)
                    jump_height = calculate_jump_height(start_height, min_height, pixel_per_meter)
                    start_x = positions[0][0]
                    end_x = positions[-1][0]
                    jump_length = abs(end_x - start_x) / pixel_per_meter
                    measuring = False
                    jump_detected = False
                elif time.time() - jump_start_time > jump_timeout:
                    cv2.putText(display_frame, "Jump timeout!", (w//2-150, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, JUMP_COLOR, 3)
                    measuring = False
                    jump_detected = False
        
        mp_drawing.draw_landmarks(
            display_frame, 
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=JUMP_COLOR if jump_detected else READY_COLOR, thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=JUMP_COLOR if jump_detected else READY_COLOR, thickness=2)
        )
        
        cv2.circle(display_frame, ankle_mid, 8, (0, 0, 255), -1)
        cv2.putText(display_frame, f"Distance: {distance_to_user:.2f}m", 
                   (w - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CALIBRATION_COLOR, 2)
    
    if len(positions) > 1:
        for i in range(1, len(positions)):
            cv2.line(display_frame, positions[i-1], positions[i], TRAJECTORY_COLOR, 2)
    
    info_panel = np.zeros((h, 300, 3), dtype=np.uint8)
    
    if calibrated:
        if measuring:
            status = "MEASURING JUMP"
            color = JUMP_COLOR
        else:
            status = "READY TO JUMP"
            color = READY_COLOR
        cv2.putText(info_panel, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        cv2.putText(info_panel, "CALIBRATION NEEDED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CALIBRATION_COLOR, 2)
    
    cv2.putText(info_panel, "CONTROLS:", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(info_panel, "c - Calibrate", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, CALIBRATION_COLOR, 1)
    cv2.putText(info_panel, "s - Start measurement", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, READY_COLOR, 1)
    cv2.putText(info_panel, "r - Reset", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(info_panel, "q - Quit", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    if jump_height > 0:
        cv2.putText(info_panel, "JUMP RESULTS:", (20, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Height: {jump_height:.2f} m", (20, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RESULT_COLOR, 2)
        cv2.putText(info_panel, f"Length: {jump_length:.2f} m", (20, 310), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RESULT_COLOR, 2)
    
    cv2.putText(info_panel, f"User height: {USER_HEIGHT} m", (20, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
    
    combined = np.hstack((display_frame, info_panel))
    cv2.imshow('Jump Measurement System', combined)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibrated = True
    elif key == ord('s') and calibrated:
        measuring = True
        jump_detected = False
        start_height = 0
        positions = []
        timestamps = []
        jump_height = 0
        jump_length = 0
    elif key == ord('r'):
        calibrated = False
        measuring = False
        jump_detected = False
        positions = []
        timestamps = []
        pixel_per_meter = 0
        start_height = 0
        jump_height = 0
        jump_length = 0
        distance_to_user = 0

cap.release()
cv2.destroyAllWindows()