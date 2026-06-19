import cv2
import os
# direcory path
os.chdir(r"D:\\Meltshop")
#import datetime
# video path
cap = cv2.VideoCapture(
    r"D:\\Meltshop\\118_fullladle.mkv")
# ret,frame = cap.read()
framerate = int(cap.get(cv2.CAP_PROP_FPS))
framecount = 0
count = 0
while True:
    success, frame = cap.read()
    # show the output frame
    frame = cv2.resize(frame, (1280, 720))
    #key = cv2.waitKey(1) & 0xFF

    # cv2.rectangle(frame, (0, 0), (2800, 1000), (0, 255, 0), 3)
    # frame1 = frame[0:1000, 0:2800
    # cv2.imshow("crop", frame1)
    framecount += 1
    # # #
    if framecount == (framerate * 1//4):
        print('hi', count)
        framecount = 0
        # output folder to store images example(img) and image names also here 152C
        cv2.imwrite('118_frames%d.jpg' % count, frame)
        count += 1
        
    if success == 0:
        break

        # show the output frame
    #cv2.imshow("Frame", cv2.resize(frame, (1000, 800)))
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
