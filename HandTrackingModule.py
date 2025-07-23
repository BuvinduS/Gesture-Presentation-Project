import cv2
import mediapipe as mp
import time


class HandDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackingCon=0.5, complexity=1):    # Constructor for the class
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackingCon = trackingCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackingCon)  # We can change the parameters if needed
        self.mpDraw = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # Hands class only uses RGB images. So we need to convert from BGR to RGB
        self.results = self.hands.process(imgRGB)

        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img



    def findPosition(self, img, handNo = 0, draw=True):
        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo] # Gets the relevant hand. First for handNo = 0

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape # height, width, columns
                cx, cy = int(lm.x*w), int(lm.y*h) # pixel positions of each landmark on the hand
                #print(id, cx, cy)       # find out the coordinates for the 21 different landmarks, id represented the landmark

                lmList.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx,cy),5, (253, 255, 57),cv2.FILLED)

        return lmList

def main():
    prevTime = 0
    currTime = 0

    cap = cv2.VideoCapture(0)

    detector = HandDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        lmList = detector.findPosition(img)

        if lmList != []:
            print(lmList[4])

        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 3)

        cv2.imshow('Image', img)
        if cv2.waitKey(1) == ord('q'):
            break




if __name__ == "__main__":
    main()