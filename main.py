import cv2
import numpy as np
import os
import hand_tracking_module as htm

# ================= SETTINGS =================
brushThickness = 10
eraserThickness = 80

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)
cap.set(10, 200)

detector = htm.HandDetector(detectionConf=0.85)

# ================= LOAD HEADER =================
folderPath = "Header"
overlayList = []

for imgPath in os.listdir(folderPath):
    img = cv2.imread(os.path.join(folderPath, imgPath))
    overlayList.append(img)

headerIndex = 0
drawColor = (20, 40, 100)

xp, yp = 0, 0
imgCanvas = None

# ================= FINGER FUNCTION (FIXED CORE LOGIC) =================
def fingersUp(lmList):
    fingers = []

    # Thumb
    if lmList[4][1] > lmList[3][1]:
        fingers.append(1)
    else:
        fingers.append(0)

    # 4 fingers
    tipIds = [8, 12, 16, 20]

    for tip in tipIds:
        if lmList[tip][2] < lmList[tip - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers

# ================= MAIN LOOP =================
while True:

    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)

    if imgCanvas is None:
        imgCanvas = np.zeros_like(img)

    img = detector.findHands(img)
    lmList, _ = detector.findPosition(img, draw=False)

    if len(lmList) != 0:

        fingers = fingersUp(lmList)

        x1, y1 = lmList[8][1], lmList[8][2]

        # ================= SELECTION MODE =================
        if fingers[1] == 1 and fingers[2] == 1:

            xp, yp = 0, 0

            if y1 < 125:

                if 100 < x1 < 150:
                    drawColor = (20, 40, 100)
                    headerIndex = 0

                elif 200 < x1 < 250:
                    drawColor = (128, 0, 0)
                    headerIndex = 1

                elif 300 < x1 < 350:
                    drawColor = (0, 0, 255)
                    headerIndex = 2

                elif 400 < x1 < 450:
                    drawColor = (0, 165, 255)
                    headerIndex = 3

                elif 500 < x1 < 550:
                    drawColor = (0, 0, 0)
                    headerIndex = 4

        # ================= DRAW MODE (FIXED) =================
        elif fingers[1] == 1 and fingers[2] == 0:

            cv2.circle(img, (x1, y1), 10, drawColor, cv2.FILLED)

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness

            cv2.line(img, (xp, yp), (x1, y1), drawColor, thickness)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)

            xp, yp = x1, y1

    # ================= MERGE CANVAS =================
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    imgBg = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(imgBg, imgCanvas)

    # ================= HEADER =================
    h, w, _ = img.shape
    img[0:125, 0:w] = cv2.resize(overlayList[headerIndex], (w, 125))

    # ================= SHOW =================
    cv2.imshow("AI Drawing Board", img)

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()