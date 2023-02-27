# import numpy as np 
# # import cv2
# import hand_tracking_module as htm
# import cv2
# import time
# import mediapipe as mp
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_hands = mp.solutions.hands
# prev_time = 0

# cap = cv2.VideoCapture(0)
# with mp_hands.Hands(
#     model_complexity=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as hands:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue
#     # current_time = time.time() 
#     # fps = 1 / (current_time - prev_time)
#     # prev_time = current_time
#     # cv2.putText(image , str(int(fps)) , (10 , 50) , cv2.FONT_HERSHEY_PLAIN , 3 , (255 ,0 ,0) , 3)
#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = hands.process(image)

#     # Draw the hand annotations on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#       for hand_landmarks in results.multi_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             image,
#             hand_landmarks,
#             mp_hands.HAND_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(255 , 0 ,150 ) , thickness= 2 , circle_radius= 2 ) ,mp_drawing.DrawingSpec(color= (255 , 150 , 0), thickness= 2 , circle_radius= 2 ) )
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#       break
# cap.release()


"""
Hand Tracing Module
By: Murtaza Hassan
Youtube: http://www.youtube.com/c/MurtazasWorkshopRoboticsandAI
Website: https://www.computervision.zone/
"""

import cv2
import mediapipe as mp
import time
import math
import numpy as np
import os
from datetime import datetime

class face_detector():
    def __init__(self, mode= 0, detection_conf =0.5 ):
        
        self.mode = mode
        self.detection_conf = detection_conf   
        self.IS_FACE = False

        self.mp_face_detection = mp.solutions.face_detection
        self.face = self.mp_face_detection.FaceDetection(self.mode,self.detection_conf)
        self.mpDraw = mp.solutions.drawing_utils

    def detect_face(self, img , draw=True):
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face.process(img_rgb)

        # Draw the face detection annotations on the image.
        
        img_gbr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if results.detections:
            self.IS_FACE = True
            for detection in results.detections:
                if draw:
                    self.mpDraw.draw_detection(img_gbr, detection)
            # Flip the image horizontally for a selfie-view display.
        else:
            self.IS_FACE = False
        return img_gbr

    def is_face(self):
        if self.IS_FACE == True:
            return True

    def take_picture(self , img):
        
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # os.chdir(directory)
        datetime_ = datetime.now()
        time_ = datetime_.strftime("%X").replace(":","-")
        date_ = datetime_.strftime("%x").replace("/","-")
        date_time = "{}.png".format(datetime_.strftime("%c")).replace(" " , "-").replace(":" , "-")
        # directory = rf"C:\Users\Morvarid\Desktop\python\open_cv\hand_tracking\saved_pictures{date_time}"
        directory = rf"C:\Users\Morvarid\Documents\saved_images\_date={date_}_time={time_}.jpg"
        cv2.imwrite(directory , img)
        if not cv2.imwrite(directory , img):
            print("the shit didn't save...")
        else:    
            print("the image is saved")

        



class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity =1 , trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = modelComplexity

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex ,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)

        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        all_hands = []
        self.lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for handtype , handlms in zip(self.results.multi_handedness , self.results.multi_hand_landmarks):
                my_hand = {}
                
                for id, lm in enumerate(myHand.landmark):
                    # print(id, lm)
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    # print(id, cx, cy)
                    
                    self.lmList.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

                my_hand["lm_list"] = self.lmList

                if handtype.classification[0].label == "Right":
                    my_hand["type"] = "Left"
                else:
                    my_hand["type"] = "Right"

                all_hands.append(my_hand)
                
            # print(all_hands)
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)

        return self.lmList, bbox

    def find_Hands(self, img, draw=True, flipType=True):
        """
        Finds hands in a BGR image.
        :param img: Image to find the hands in.
        :param draw: Flag to draw the output on the image.
        :return: Image with or without drawings
        """
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        allHands = []
        h, w, c = img.shape
        if self.results.multi_hand_landmarks:
            for handType, handLms in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                myHand = {}
                ## lmList
                mylmList = []
                xList = []
                yList = []
                for id, lm in enumerate(handLms.landmark):
                    px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                    mylmList.append([px, py, pz])
                    xList.append(px)
                    yList.append(py)

                ## bbox
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                boxW, boxH = xmax - xmin, ymax - ymin
                bbox = xmin, ymin, boxW, boxH
                cx, cy = bbox[0] + (bbox[2] // 2), \
                         bbox[1] + (bbox[3] // 2)

                myHand["lmList"] = mylmList
                myHand["bbox"] = bbox
                myHand["center"] = (cx, cy)

                if flipType:
                    if handType.classification[0].label == "Right":
                        myHand["type"] = "Left"
                    else:
                        myHand["type"] = "Right"
                else:
                    myHand["type"] = handType.classification[0].label
                allHands.append(myHand)

                ## draw
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms,
                                               self.mpHands.HAND_CONNECTIONS)
                    cv2.rectangle(img, (bbox[0] - 20, bbox[1] - 20),
                                  (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                  (255, 0, 255), 2)
                    cv2.putText(img, myHand["type"], (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                2, (255, 0, 255), 2)
        if draw:
            return allHands, img
        else:
            return allHands

    def fingersUp(self):
        fingers = []
        # Thumb
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Fingers
        for id in range(1, 5):

            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        # totalFingers = fingers.count(1)

        return fingers

    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()
    # print("the handDetector class contains : " + "\n" + str(dir(handDetector)))
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if len(lmList) != 0:
            print(" y = " + str(lmList[8][2]))
            print(" x = "+ str(lmList[8][1]))

        # cTime = time.time()
        # fps = 1 / (cTime - pTime)
        # pTime = cTime

        # cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
        #             (255, 0, 255), 3)

        cv2.imshow("Image", img)
        if cv2.waitKey(1)== ord('q'):
            break



if __name__ == "__main__":
    main()


    # caeri