import cv2
import os
import webbrowser
from playsound import playsound
import time
video_capture=cv2.VideoCapture(0)

facecascade=cv2.CascadeClassifier(r".\haarcascade_frontalface_default.xml")
clf=cv2.face.LBPHFaceRecognizer_create()
clf.read('face_clasifier.yml')
coord=[]
i=1 #initialized for jarvis sound
def facedetect(img,scalefactor,minneighbors):
    global coord
    global i
    imggray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   #convert image to rgb
    features=facecascade.detectMultiScale(img,scalefactor,minneighbors)  #provide 2D array of x,y,w,h(dimension of face coordinates)
    for (x,y,w,h) in features:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        if len(coord)==4:
            idd,_=clf.predict(imggray[coord[1]:coord[1]+coord[3],coord[0]:coord[0]+coord[2]])   #facerecognize
            print(idd)
            if idd==1:
                cv2.putText(img,"Jay",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2,cv2.LINE_4)
                if i==14:
                    playsound(r"C:\Users\Jay Bhosale\Downloads\d554ef73-b9d4-43be-bff5-93cfcad9f9ee.wav")
                    webbrowser.open(r"https://calendar.google.com/calendar/r/day")
                    time.sleep(2.5)
                    playsound(r"C:\Users\Jay Bhosale/Downloads/a10852b7-5249-4709-b2b1-6038485206a0.wav")

                i=i+1 
        
            elif idd==2:
                
                cv2.putText(img,"Chitesh",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2,cv2.LINE_4)
                if i==14:
                    playsound(r".\d554ef73-b9d4-43be-bff5-93cfcad9f9ee.wav")
                    webbrowser.open(r"https://calendar.google.com/calendar/r/day")
                    time.sleep(2.5)
                    playsound(r"./a10852b7-5249-4709-b2b1-6038485206a0.wav")

                i=i+1    
            elif idd==3:
                cv2.putText(img,"Shubham",(x,y-6),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,255),2,cv2.LINE_4)
            
        coord=[x,y,w,h]
    return img,coord

                    

##idd=1
##img_id=1
def Onlyfacedetect(img):
    img,coord = facedetect(img,1.1,10)
    
##    global img_id    
##    if len(coord)==4:
##        img_id=img_id+1
##        small_img=img[coord[1]:coord[1]+coord[3],coord[0]:coord[0]+coord[2]]
##        cv2.imwrite(r"C:\Users\Jay Bhosale\PycharmProjects\Regression\venv\Lib\site-packages\Dataset_face_recognization\jay."+str(idd)+"."+str(img_id)+".jpg",small_img)   #use to create dataset of face from live camera
##        
    return img


    


while True:
    TorF,img=video_capture.read() 
    img=Onlyfacedetect(img)
    cv2.imshow("JB",img)   #used with wait
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
   
video_capture.release()
cv2.destroyAllWindows()    
#cv2.imshow('aaa',b)     
#Scalefactor is percentage change to upscale or downscale image(1.01=1.01,2.02,3.03) (2=2,4,6,8)  Its just a example
