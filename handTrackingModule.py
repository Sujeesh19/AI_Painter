import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHands, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img,
                              cv2.COLOR_BGR2RGB)  # bcoz this class only accepts RGB images, so converting it into RGB images
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLns in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLns, self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):

        self.lm = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, ln in enumerate(myHand.landmark):

                h, w, c = img.shape
                cx, cy = int(ln.x * w), int(ln.y * h)

                self.lm.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)

        return self.lm

    def fingersUp(self):
        fingers = []
        # THUMB
        if self.lm[self.tipIds[0]][1] < self.lm[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # FOUR FINGERS
        for id in range(1, 5):
            if self.lm[self.tipIds[id]][2] < self.lm[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

def main():
    wCam, hCam = 1080, 720
    cap = cv2.VideoCapture(1)

    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = HandDetector()
    ptime = 0
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lm = detector.findPosition(img)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(img, f'FPS: {int(fps)}', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()