from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import HandTrackingModule as htm
import math
import time
import pyautogui

lastSwipeTime = 0
cooldownTime = 1
displayGesture = None
displayTime = 0
lastGestureDirection = None
lastGestureTime = 0
gestureResetDelay = 1

def getCamera(wCam=1280, hCam=760):
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    return cap

detector = htm.HandDetector(detectionCon=0.7)
cap = getCamera()
xPosHistory = deque(maxlen=10)

def detectSwipe(posHistory, distanceThreshold = 100, timeThreshold = 0.5):
    if len(posHistory) < 2:
        return None

    xEnd, timeEnd = posHistory[-1]
    xStart, timeStart = posHistory[0]

    timeTaken = timeEnd - timeStart
    distanceTravelled = xEnd - xStart

    if abs(distanceTravelled) > distanceThreshold and timeTaken < timeThreshold:
        if distanceTravelled > 0:
            return "Right"
        else:
            return "Left"

    return None



def controlPresentation(xPosHistory, lmList, img, indicate=True):
    global lastSwipeTime, displayTime, displayGesture, lastGestureDirection, lastGestureTime
    currTime = time.time()
    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]     # x coordinate of the index fingertip
        cv2.circle(img, (x1, y1), 5, (0, 0, 0), cv2.FILLED)
        xPosHistory.append((x1, currTime))

        if currTime - lastSwipeTime > cooldownTime:
            gesture = detectSwipe(xPosHistory)

            if gesture:
                if lastGestureDirection == None or gesture == lastGestureDirection:
                    print(f"Detected {gesture} swipe")
                    lastSwipeTime = currTime
                    displayGesture = gesture
                    lastGestureTime = currTime
                    displayTime = currTime
                    lastGestureDirection = gesture
                    xPosHistory.clear()

                    if gesture == "Right":
                        pyautogui.press("right")
                    elif gesture == "Left":
                        pyautogui.press("left")

                else:
                    pass
        if lastGestureDirection is not None and (currTime - lastGestureTime) > gestureResetDelay:
            lastGestureDirection = None




while True:
    success, img = cap.read()
    img = detector.findHands(img)       # Find the hand in the video
    lmList = detector.findPosition(img, draw=False)

    controlPresentation(xPosHistory, lmList, img)


    if displayGesture and time.time() - displayTime < 1.5:
        # Shows the gesture for 1.5 seconds
        cv2.putText(img, f"{displayGesture} swipe", (10, 70), cv2.FONT_ITALIC, 1, (255, 0, 0), 3)
    else:
        displayGesture = None       # Clean up when the time is up

    cv2.imshow("Presentation control", img)

    if cv2.waitKey(1) == ord('q'):
        break