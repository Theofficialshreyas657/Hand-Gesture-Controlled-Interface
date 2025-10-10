import cv2 as cv
import mediapipe as media 
import time

class handtracking:
    def __init__(self, mode=False ,maxHands=2, modelComplexity=1,detectionCon=0.5, trackCon=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionCon=detectionCon
        self.trackCon=trackCon
        self.modelComplexity=1
        self.vidcap = cv.VideoCapture(0)
        self.obj_hands = media.solutions.hands
        self.hand = self.obj_hands.Hands(self.mode,self.maxHands,self.modelComplexity,
                                         self.detectionCon,self.trackCon)
        self.mpdraw= media.solutions.drawing_utils
    def findhands(self, img,Draw=True):
        imgrgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results=self.hand.process(imgrgb)
        if self.results.multi_hand_landmarks:
            if Draw :
                for handlms in self.results.multi_hand_landmarks:
                    self.mpdraw.draw_landmarks(img,handlms,self.obj_hands.HAND_CONNECTIONS)

    def findposition(self,img ,positions=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], hand_no=0, draw= True):
        handlmk=[]
        if self.results.multi_hand_landmarks:
            myhand= self.results.multi_hand_landmarks[hand_no]
            for id,lm in enumerate(myhand.landmark):
                if id in positions:
                    h,w,c = img.shape
                    cx,cy= int(lm.x*w),int(lm.y*h)
                    print(id,cx,cy)
                    handlmk.append((id,cx,cy))
                    if draw:
                        cv.circle(img,(cx,cy),4,(225,0,125),cv.FILLED)
        return handlmk
                
            


    #adding findlandmark function that prints the excat location of landmarks
def main():
    vidcap = cv.VideoCapture(0)
    track=handtracking()
    ptime=0
    ctime=0
    while True :
        sucess,img = vidcap.read()
        ctime=time.time()
        fps= 1/(ctime-ptime)
        ptime=ctime
        track.findhands(img)
        cv.putText(img,f"fps:{int(fps)}",(50,50),cv.FONT_HERSHEY_COMPLEX,1,(225,0,255),2)
        
        cv.imshow("vid",img)
        if cv.waitKey(10)&0xFF==ord('d'):
            break

if __name__=="main":
    main()