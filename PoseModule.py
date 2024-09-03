import cv2
import mediapipe as mp
import time
import math
import numpy as np


class PomseDetector:
    def __init__(self, mode=False, upperBody=False, smooth=False, enable_segmentation=False, smooth_segmentation=False,
                 detectionConf=0.5, trackingConf=0.5):

        self.results = None
        self.lmlist = []  # Initialize lmlist as an instance variable
        self.smooth_segmentation = smooth_segmentation
        self.enable_segmentation = enable_segmentation
        self.upperBody = upperBody
        self.trackingConf = trackingConf
        self.smooth = smooth
        self.mode = mode
        self.detectionConf = detectionConf

        # Mediapipe pose model
        self.poseModel = mp.solutions.pose
        # Mediapipe landmark utility
        self.draw = mp.solutions.drawing_utils
        # Object Instantiation
        self.pose = self.poseModel.Pose(self.mode, self.upperBody, self.smooth, self.enable_segmentation,
                                        self.smooth_segmentation, self.detectionConf, self.trackingConf)

    def DetectKero(self, feed, draw=True):
        # Convert Input to Mediapipe color pattern
        vidRGB = cv2.cvtColor(feed, cv2.COLOR_BGR2RGB)

        # Feed processed stream to Pose Model
        self.results = self.pose.process(vidRGB)

        # Reset lmlist for each new frame
        self.lmlist = []

        # Draw landmarks if detected
        if self.results.pose_landmarks:
            if draw:
                self.draw.draw_landmarks(feed, self.results.pose_landmarks, self.poseModel.POSE_CONNECTIONS)
            # self.GetPose(feed, log=False)

        return feed

    def GetPose(self, img, log=True):
        h, w, _ = img.shape
        self.lmlist = []

        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmlist.append([id, cx, cy])
                if log:
                    print(f"Landmark {id}: x={cx}, y={cy}")

        return self.lmlist

    def FindAngle(self, img, p1, p2, p3, draw=True):
        angles=[]
        # Get landmarks
        x1, y1 = self.lmlist[p1][1:]
        x2, y2 = self.lmlist[p2][1:]
        x3, y3 = self.lmlist[p3][1:]
        # n1 = self.lmlist[0][:1]

        vector1 = np.array((x1, y1)) - np.array((x2, y2))  # Vector from point2 to point1
        vector2 = np.array((x3, y3)) - np.array((x2, y2))  # Vector from point2 to point3

        # Calculate norms (lengths) of vectors
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)

        if norm1 == 0 or norm2 == 0:
            return None

        # Calculate the angle between the two vectors
        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle = np.arccos(cosine_angle)  # Get the angle in radians


        # Convert the angle from radians to degrees
        angle = round(np.degrees(angle))

        if angle < 0:
            angle += 360

        #print(angle)

        if draw:
            cv2.circle(img, (x1, y1), 5, (255, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 255, 0), cv2.FILLED)
            cv2.circle(img, (x3, y3), 5, (255, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        return angle

    def ClassifyPose(self, img, leftElbow, rightElbow, leftHip, rightHip, leftKnee, rightKnee):
        tree = [129, 128, 169, 103, 167, 24]
        plank = [96, 89, 171, 173, 180, 180]
        pose = [0, 0, 0, 0, 0]
        plank = False

        none = True
        Pose = False

        if (leftElbow <= tree[0] + 10 or leftElbow >= tree[0] - 10) and (
                rightElbow <= tree[1] + 10 or rightElbow >= tree[1] - 10) and (
                leftHip <= tree[2] + 10 or leftHip >= tree[2] - 10) and (
                rightHip <= tree[3] + 10 or rightHip >= tree[3] - 10) and (
                leftKnee <= tree[4] + 10 or leftKnee >= tree[4] - 10) and (
                rightKnee <= tree[5] + 10 or rightKnee >= tree[5] - 10):
            print("Tree")
            pose[0] = 1
            Pose = True
        else:
            print("Not tree")
            Pose = False

        return pose
    # def Classify(self, img, pic):


def main():
    cap = cv2.VideoCapture('y.mp4')  # For video file
    # cap = cv2.VideoCapture(0)  # For webcam

    # Error control
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    detect = PomseDetector()

    prevTime = 0
    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame.")
            break

        frame = detect.DetectKero(frame, draw=True)
        currentTime = time.time()
        fps = 1 / (currentTime - prevTime)
        prevTime = currentTime

        cv2.putText(frame, 'FPS: ' + str(int(fps)), (50, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Close window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
