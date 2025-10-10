import mediapipe as media 
import time 
import cv2 as cv
import handtraking as htk
import keyboard 

vidcap = cv.VideoCapture(0)
track=htk.handtracking()
ptime=0
handlmk=[]
ctime=0
final_list=[]
while True :
    sucess,img = vidcap.read()
    ctime=time.time()
    fps= 1/(ctime-ptime)
    ptime=ctime
    track.findhands(img)
    handlmk=track.findposition(img,[0,4])
    cv.putText(img,f"fps:{int(fps)}",(50,50),cv.FONT_HERSHEY_COMPLEX,1,(225,0,255),2)
    cv.imshow("vid",img)
    if cv.waitKey(10)&0xFF==ord('d'):
        break
    print(handlmk)

