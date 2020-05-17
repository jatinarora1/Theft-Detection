import numpy as np
import cv2 
import telegram
TOKEN = '1193023091:AAEl9eLOZ6Q0PdDRXF07TprHDXt9tEGuclo'
bot = telegram.Bot(TOKEN)
from datetime import datetime
from activity_captioning import images
from Telegram_Bot import bot2
from keras.models import load_model
import matplotlib.pyplot as plt
model = load_model("models/best_model.h5")
class_to_label = {0 :'Angry', 1 : 'Disgust', 2:'Fear', 3 :'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

face_cascade = cv2.CascadeClassifier("facial_expression/haarcascade_frontalface_default.xml")
first_frame=None
status_list=[None,None]
times=[]
chat_id = 1213182814
currentframe=0

video  = cv2.VideoCapture(0)
ret,frame1 = video.read()
ret,frame2 = video.read()
while video.isOpened():
    text = 'Unoccupied'
    status=1
    timestamp = datetime.now()
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,thresh = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh,None,iterations = 3)
    (cnts,_)=cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        
        text = "Occupied"
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 2)
    
    cv2.imshow("feed",frame1)
    
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame2, "Room Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # cv2.putText(frame1,(10, int(frame1.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.35, (0, 0, 255), 1)

    status_list.append(status)

    status_list=status_list[-2:]


           
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())


    gray_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(frame2,1.2,5)



    if len(faces)  == 0:
        for (x,y,w,h) in faces:
            cv2.putText(frame2, "Processing", (x,y-10), cv2.FONT_HERSHEY_COMPLEX,1,(255,30,0),2,cv2.LINE_AA)

            cv2.rectangle(frame2,(x,y),(x+w,y+h),(255,0,0),2)


    label = []
    for face in faces:
        x,y,w, h = face
        
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        
        if(np.all(np.array(face_section.shape))):
            face_section = cv2.resize(face_section,(48,48))

            pred = np.argmax(model.predict(face_section.reshape(1,48,48,1)))
            label = class_to_label[pred]

            cv2.putText(frame2, label, (x,y-10), cv2.FONT_HERSHEY_COMPLEX,1,(255,30,0),2,cv2.LINE_AA)
            cv2.rectangle(frame2,(x,y),(x+w,y+h), (0,255,255),2)


    frame1 = frame2
    cv2.imshow("color",frame2)
    cv2.imshow("Guassian Blur",blur)
    cv2.imshow("dilate",dilated)
    ret,frame2 = video.read()

    key=cv2.waitKey(1)

    if key & ord('q') == 0xFF:
        if status==1:
            times.append(datetime.now())
        break
#Taking the image of the frame-----------------------------------------------------------------------
    
    if(text=="Occupied" and currentframe%200==0):
        name = 'images/'+str(currentframe) + '.jpg'
        cv2.imwrite(name,frame2)
        currentframe += 1
        emotions = str(label)
        caption = images.caption_this_image(name) + "\n" +"person in image seems : " + emotions + "."
        
        bot2.tasveer(name,caption)
        
    else:
        currentframe+=1
        

print(currentframe)
video.release()
cv2.destroyAllWindows
