import math
import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
count = 0
dir = 0
dir2 = 0
count2 = 0

  
def findAngle(p1, p2, p3):
    # Get the landmarks
    x1, y1 = lmList[p1][1:]
    x2, y2 = lmList[p2][1:]
    x3, y3 = lmList[p3][1:]

    # Calculate the Angle
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(image, (x3, y3), (x2, y2), (255, 255, 255), 3)
        cv2.circle(image, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x1, y1), 15, (0, 0, 255), 2)
        cv2.circle(image, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x2, y2), 15, (0, 0, 255), 2)
        cv2.circle(image, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
        cv2.circle(image, (x3, y3), 15, (0, 0, 255), 2)
        cv2.putText(image, str(int(angle)), (x2 - 50, y2 + 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

    return angle


cap = cv2.VideoCapture(0)
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(image, results.pose_landmarks)
        lmList = []

        if results.pose_landmarks:
            for id, lm in enumerate(results.pose_landmarks.landmark):
                h, w, c = image.shape
                # print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            if len(lmList) != 0:
                #                 RIGHT
                angle = findAngle(12, 14, 16)
                per = np.interp(angle, (210, 310), (0, 100))
                bar = np.interp(angle, (220, 310), (650, 100))
                color = (255, 0, 255)
                if per == 100:
                    color = (0, 255, 0)
                    if dir == 0:
                        count += 0.5
                        dir = 1
                if per == 0:
                    color = (0, 255, 0)
                    if dir == 1:
                        count += 0.5
                        dir = 0
                cv2.putText(image, "RIGHT", (450, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)
                cv2.putText(image, "CURLS", (450, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)
                cv2.putText(image, str(int(count)), (550, 200), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 4)
                cv2.putText(image, f'{int(per)} %', (550, 470), cv2.FONT_HERSHEY_PLAIN, 3, color, 4)
                
                #                   LEFT

                angle2 = findAngle(15, 13, 11)
                per2 = np.interp(angle2, (210, 310), (0, 100))
                bar2 = np.interp(angle2, (220, 310), (650, 100))
                # print(angle, per)
                color = (255, 0, 255)
                if per2 == 100:
                    color = (0, 255, 0)
                    if dir2 == 0:
                        count2 += 0.5
                        dir2 = 1
                if per2 == 0:
                    color = (0, 255, 0)
                    if dir2 == 1:
                        count2 += 0.5
                        dir2 = 0
                cv2.putText(image, "LEFT", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)
                cv2.putText(image, "CURLS", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 4)
                cv2.putText(image, str(int(count2)), (30, 200), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 4)
                cv2.putText(image, f'{int(per2)} %', (10, 470), cv2.FONT_HERSHEY_PLAIN, 3, color, 4)

        cv2.imshow('MediaPipe Pose', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
cap.release()
