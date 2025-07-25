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
gestureResetDelay = 0.5

def getCamera(wCam=1280, hCam=760):
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)
    return cap

detector = htm.HandDetector(detectionCon=0.7)
cap = getCamera()
xPosHistoryLeft = deque(maxlen=10)
xPosHistoryRight = deque(maxlen=10)

def detectSwipe(posHistory, depth, baseThreshold = 100, timeThreshold = 0.8):
    if len(posHistory) < 2:
        return None

    xEnd, timeEnd = posHistory[-1]
    xStart, timeStart = posHistory[0]

    timeTaken = timeEnd - timeStart
    distanceTravelled = xEnd - xStart

    scalingFactor = 1 + abs(depth) * 2
    scaledThreshold = baseThreshold * scalingFactor

    # print(f'travelled a distance of  {distanceTravelled} in {timeTaken} ,while the threshold is {scaledThreshold}')

    if abs(distanceTravelled) > scaledThreshold and timeTaken < timeThreshold:
        if distanceTravelled > 0:
            return "Right"
        else:
            return "Left"

    return None


def controlPresentation(xPosHistory, lmList, img, handLabel, indicate=True):
    global lastSwipeTime, displayTime, displayGesture, lastGestureDirection, lastGestureTime
    currTime = time.time()
    if len(lmList) != 0:
        x1, y1 = lmList[8][1], lmList[8][2]     # x coordinate of the index fingertip
        depth = lmList[8][3] if len(lmList[8]) > 3 else 0 # lmList[8][3] is the Z-depth (if available) Relative depth from the wrist
        cv2.circle(img, (x1, y1), 5, (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"Current depth : {depth}", (750, 70), cv2.FONT_ITALIC, 1, (255, 0, 0), 3)
        xPosHistory.append((x1, currTime))

        if currTime - lastSwipeTime > cooldownTime:
            gesture = detectSwipe(xPosHistory, depth=depth)

            if gesture == handLabel:
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
    img = cv2.flip(img, 1)
    # img = detector.findHands(img)       # Find the hand in the video
    # lmList = detector.findPosition(img, draw=False)

    img, hands = detector.findTwoHands(img)

    for label, lmList in hands:
        if label == "Right":
            controlPresentation(xPosHistoryRight, lmList, img, handLabel="Right")
            # print("Right Hand!")
        elif label == "Left":
            controlPresentation(xPosHistoryLeft, lmList, img, handLabel="Left")
            # print("Left Hand!")


    if displayGesture and time.time() - displayTime < 1.5:
        # Shows the gesture for 1.5 seconds
        cv2.putText(img, f"{displayGesture} swipe", (10, 70), cv2.FONT_ITALIC, 1, (255, 0, 0), 3)
    else:
        displayGesture = None       # Clean up when the time is up

    cv2.imshow("Presentation control", img)

    if cv2.waitKey(1) == ord('q'):
        break