import cv2
import mediapipe as mp
import time
import PoseModule

cap = cv2.VideoCapture('y.mp4')  # For video file
# cap = cv2.VideoCapture(0)  # For webcam

detect = PoseModule.PomseDetector()
leftElbow = 0
rightElbow = 0
leftHip = 0
RightHip = 0
leftKnee = 0
rightKnee = 0
prevTime = 0
while True:
    success, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (1280, 720))
    detect.DetectKero(frame, draw=False)
    lmlist = detect.GetPose(frame, log=False)
    if len(lmlist) != 0:
        # Left Arm
        leftElbow = detect.FindAngle(frame, 11, 13, 15, draw=True)
        # Right Arm
        rightElbow = detect.FindAngle(frame, 12, 14, 16, draw=True)
        # Left Hip
        leftHip = detect.FindAngle(frame, 11, 23, 25, draw=True)
        # Right Hip
        RightHip = detect.FindAngle(frame, 12, 24, 26, draw=True)
        # Left Knee
        leftKnee = detect.FindAngle(frame, 23, 25, 27, draw=True)
        # Right Knee
        rightKnee = detect.FindAngle(frame, 24, 26, 28, draw=True)

        # Neck
        # neck = detect.FindAngle(frame, 24, 26, 28, draw=True)

    # print(rightKnee)

    detect.ClassifyPose(frame, leftElbow, rightElbow, leftHip, RightHip, leftKnee, rightKnee)
    # print(lmlist)
    currentTime = time.time()
    fps = 1 / (currentTime - prevTime)
    prevTime = currentTime

    cv2.putText(frame, 'FPS: ' + str((int(fps))), (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
    cv2.imshow('Video', frame)

    # cv2.waitKey(1)
    # Press Q to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Close window
cap.release()

cv2.destroyAllWindows()
