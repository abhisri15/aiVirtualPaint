import cv2
import time
import os
import numpy as np
import HandTrackingModule as htm


####################
brushThickness = 15
eraserThickness = 100
xp, yp = 0, 0
####################


folderPath = "Header"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))
header = overlayList[0]
drawColor=(255,0,255)

cap = cv2.VideoCapture(0)
cap.set(3,1280) # we want exact same size, 720,1280
cap.set(4,720)

detector = htm.handDetector(detectionCon=0.85)

imgCanvas = np.zeros((720,1280,3),np.uint8)

while True:
    # 1. Import image
    success,img = cap.read()
    img = cv2.resize(img, (1280, 720))
    img = cv2.flip(img,1)
    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList,bb = detector.findPosition(img, draw=False) # landmark list

    if len(lmList)!=0:
        # print(lmList)

        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # 3. Check which fingers are up

        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection mode - Two fingers are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            print('Selection Mode')
            # checking for click
            if y1<125:
                if 250<x1<450:
                    header = overlayList[0]
                    drawColor=(220,50,200)
                elif 550<x1<750:
                    header = overlayList[1]
                    drawColor=(255,50,30)
                elif 800<x1<950:
                    header = overlayList[2]
                    drawColor = (100, 255, 50)
                elif 1050<x1<1200:
                    header = overlayList[3]
                    drawColor = (0, 0, 0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

        # 5. If Drawing mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,drawColor,cv2.FILLED)
            print('Drawing Mode')
            if xp==0 and yp==0:
                xp, yp = x1, y1

            if drawColor ==(0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp,yp),(x1,y1),drawColor,brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

            xp,yp = x1,y1

    imgGray = cv2.cvtColor(imgCanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,50,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img,imgCanvas)


    # setting the header image
    img[0:125, 0:1280] = header  # only in this specific region our camera will be overlapped by the image (Header)


    # TRY THIS! LOOKS COOL
    # img = cv2.addWeighted(img,0.5,imgCanvas,0.5,0)

    cv2.imshow('Image',img)
    # cv2.imshow('Canvas', imgCanvas)
    # cv2.imshow('Inverse', imgInv)
    cv2.waitKey(1)