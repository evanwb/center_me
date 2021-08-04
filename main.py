import cv2
import numpy as np
import math
import os


casacde_path = os.path.dirname(os.path.abspath(__file__)) + '/cascades/haarcascade_frontalface_default.xml'
print(casacde_path)

cascade = cv2.CascadeClassifier(casacde_path)

cap = cv2.VideoCapture(0)

resolution = (1280,720)

center = (int(resolution[0]/2) , int(resolution[1]/2))
radius = 10
center_color = (255,0,0)
box_color = (0,0,255)
line_color = (0,255,0)
font = cv2.FONT_HERSHEY_SIMPLEX
thickness = 2
org = (50,50)


while(cap.isOpened()):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray,(5,5), .3)
    faces = cascade.detectMultiScale(gray, 1.4, 4)
    #frame = cv2.circle(frame, center, radius, center_color, thickness)

    

    for (x,y,w,h) in faces:
        
        box_center = (int(x+(w/2)),int(y+(h/2)))
        distance = math.sqrt(((box_center[0]-center[0])**2)+((box_center[1]-center[1])**2))
        distance = 'Pixels from center: '+str(int(distance))
        #cv2.circle(frame, box_center, radius, box_color, thickness)
        #cv2.line(frame, box_center, center, line_color, thickness)
        if (box_center[0] - center[0]) > 60:
            cv2.rectangle(frame, (x,y), (x+w, y+h), box_color, thickness)
            cv2.putText(frame, "You are not centered", org, font, 1, box_color, thickness, cv2.LINE_AA)
            cv2.putText(frame, "Move left", (50,80), font, 1, box_color, thickness, cv2.LINE_AA)
            cv2.putText(frame, distance, (50,120), font, 1, box_color, thickness, cv2.LINE_AA)
        elif (box_center[0] - center[0]) < -60:
            cv2.rectangle(frame, (x,y), (x+w, y+h), box_color, thickness)
            cv2.putText(frame, "You are not centered", org, font, 1, box_color, thickness, cv2.LINE_AA)
            cv2.putText(frame, "Move right", (50,80), font, 1, box_color, thickness, cv2.LINE_AA)
            cv2.putText(frame, distance, (50,120), font, 1, box_color, thickness, cv2.LINE_AA)    
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), line_color, thickness)
            cv2.putText(frame, "You're all centered up", org, font, 1, line_color, thickness, cv2.LINE_AA)

        

    cv2.imshow('image', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
