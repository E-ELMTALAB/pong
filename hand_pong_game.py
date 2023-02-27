import cv2
import numpy as np
import time 
import hand_tracking_module as htm


##################################################
#### variables
##################################################
y = 50
mag_pad_y = 50
cyn_pad_y = 300
mag_score , cyn_score = 0 ,0
speedx , speedy = -40 , 30
ball_posx , ball_posy = 700 , 400
lastpos_mag , lastpos_cyn = 10 , 10
game_over = False

#### initializing and setting the width and height of the input frame
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH , 1500)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT , 700)

#### object from hand detector to detect hands
hand_detector = htm.handDetector()

##################################################
#### overlaying pad images on the background
##################################################
def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv2.split(imgFront)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    maskBGR = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
    imgRGB = cv2.cvtColor(imgRGBA, cv2.COLOR_BGRA2BGR)

    imgMaskFull = np.zeros((hb, wb, cb), np.uint8)
    imgMaskFull[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = imgRGB
    imgMaskFull2 = np.ones((hb, wb, cb), np.uint8) * 255
    maskBGRInv = cv2.bitwise_not(maskBGR)
    imgMaskFull2[pos[1]:hf + pos[1], pos[0]:wf + pos[0], :] = maskBGRInv

    imgBack = cv2.bitwise_and(imgBack, imgMaskFull2)
    imgBack = cv2.bitwise_or(imgBack, imgMaskFull)

    return imgBack


#### read the images of the background and the ball and the pads
cyan_pad = cv2.imread(r"C:\python\open_cv\hand_tracking\rect.png")
magenta_pad = cv2.imread(r"C:\python\open_cv\hand_tracking\rect2.png")
white_background = cv2.imread(r"C:\python\open_cv\hand_tracking\white.jpg")
white_background = cv2.resize(white_background ,(0,0), fx=4 , fy=2)
mag_shape = magenta_pad.shape
cyn_shape = cyan_pad.shape
white_shape = white_background.shape
white_background[50: 50+mag_shape[0] , 50:50+mag_shape[1]] = magenta_pad


##################################################
#### the main loopo
##################################################
while cv2.waitKey(1) != ord("q"):

    #### getting the background image
    white_background = cv2.imread(r"C:\python\open_cv\hand_tracking\white.jpg")
    white_background = cv2.resize(white_background ,(0,0), fx=3 , fy=1.6)

    ret , frame = cap.read()

    #### finding the hands and their positions
    hands , frame = hand_detector.find_Hands(frame)

    #### the score of the players 
    cv2.putText(white_background , "magenta player:" + str(mag_score), ( 140 , 100) , cv2.FONT_HERSHEY_PLAIN , 2 , (0,0,0) , 2)
    cv2.putText(white_background , "cyan player:" + str(cyn_score), ( 990 , 100) , cv2.FONT_HERSHEY_PLAIN , 2 , (0,0,0) , 2)

    ##################################################
    #### find two hands and changes the ball pos according to the positions of the hands
    ##################################################
    for hand in hands:

        if hand["type"] == "Left":

            if hand["lmList"][8][1] < 550 and hand["lmList"][8][1] > 10:
                white_background[hand["lmList"][8][1]: hand["lmList"][8][1]+cyn_shape[0] , 1280:1280 +cyn_shape[1]] = cyan_pad

                if ball_posx > 1250 and (hand["lmList"][8][1] <= ball_posy <= hand["lmList"][8][1] + cyn_shape[0]):
                    speedx = -speedx
                lastpos_cyn = hand["lmList"][8][1]
            else :
                white_background[lastpos_cyn: lastpos_cyn +cyn_shape[0] , 1280:1280 +cyn_shape[1]] = cyan_pad
                if ball_posx > 1250 and (hand["lmList"][8][1] <= ball_posy <= hand["lmList"][8][1] + cyn_shape[0]):
                    speedx = -speedx

        if hand["type"] == "Right":

            if hand["lmList"][8][1] < 550 and hand["lmList"][8][1] > 10:
                white_background[hand["lmList"][8][1]: hand["lmList"][8][1]+mag_shape[0] , 50:50 +mag_shape[1]] = magenta_pad
                lastpos_mag = hand["lmList"][8][1]

                if ball_posx <131 and (hand["lmList"][8][1]<=ball_posy<=hand["lmList"][8][1] + mag_shape[0]):
                    speedx = -speedx
            else :
                white_background[lastpos_mag: lastpos_mag +mag_shape[0] , 50:50 +mag_shape[1]] = magenta_pad
                if ball_posx <131 and (hand["lmList"][8][1]<=ball_posy<=hand["lmList"][8][1] + mag_shape[0]):
                    speedx = -speedx

    ##################################################
    #### move the ball
    ##################################################
    ball_posx += speedx
    ball_posy += speedy

    ball = cv2.circle(white_background , (ball_posx ,ball_posy) , 26 , (255 , 255 , 0) , -1 )

    if ball_posy > 650 or ball_posy < 90:
        speedy = -speedy
    if ball_posx > 1350 :
       mag_score +=1 
       game_over = True
    if ball_posx < 40 :
        cyn_score += 1
        game_over = True


    if game_over:

        ball_posx , ball_posy = 700 , 400
        game_over = False
        speedx = -speedx
        speedy = -speedy

    cv2.imshow("the hands" , white_background)
