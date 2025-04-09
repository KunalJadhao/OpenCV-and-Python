import cv2 as cv
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_face = mp.solutions.face_detection
face = mp_face.FaceDetection(min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv.VideoCapture(0)

finger_colors = {
    'thumb': (0, 255, 0),
    'index': (128, 128, 128),
    'middle': (0, 255, 255),
    'ring': (255, 0, 0),
    'little': (0, 0, 255)
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
        
    frame = cv.flip(frame, 1)
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    
    hand_results = hands.process(rgb_frame)
    face_results = face.process(rgb_frame)
    
    output_frame = frame.copy()
    
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(output_frame, detection)
    
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(output_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            little_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            fingers = [
                (thumb_tip.y, 'thumb'),
                (index_tip.y, 'index'),
                (middle_tip.y, 'middle'),
                (ring_tip.y, 'ring'),
                (little_tip.y, 'little')
            ]
            
            fingers.sort()
            extended_finger = fingers[0][1]
            
            if extended_finger == 'thumb':
                output_frame[:, :, 0] = 0
                output_frame[:, :, 2] = 0
                cv.putText(output_frame, 'Green Channel', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, finger_colors['thumb'], 2)
            elif extended_finger == 'index':
                output_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                output_frame = cv.cvtColor(output_frame, cv.COLOR_GRAY2BGR)
                cv.putText(output_frame, 'Grayscale', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, finger_colors['index'], 2)
            elif extended_finger == 'middle':
                output_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                cv.putText(output_frame, 'HSV Color Space', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, finger_colors['middle'], 2)
            elif extended_finger == 'ring':
                output_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                cv.putText(output_frame, 'RGB Color Space', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, finger_colors['ring'], 2)
            elif extended_finger == 'little':
                output_frame = cv.bitwise_not(frame)
                cv.putText(output_frame, 'Inverted Colors', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, finger_colors['little'], 2)

    cv.imshow('Finger Color Mapping with Face Detection', output_frame)
    
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()