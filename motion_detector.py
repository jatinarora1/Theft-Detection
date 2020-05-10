import cv2
import telegram
TOKEN = '1193023091:AAEl9eLOZ6Q0PdDRXF07TprHDXt9tEGuclo'
bot = telegram.Bot(TOKEN)
from datetime import datetime
import images 
import bot2
from keras.layers import *
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model("models/best_model.h5")
class_to_label = {0 :'Angry', 1 : 'Disgust', 2:'Fear', 3 :'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}
# cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
first_frame=None
status_list=[None,None]
times=[]
chat_id = 1213182814
#df=pandas.DataFrame(columns=["Start","End"])

video=cv2.VideoCapture(0)
currentframe=0
while True:
    check, frame = video.read()
    status=1
    timestamp = datetime.now()
    text = "Unoccupied"
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(31,31),0)

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)
    thresh_frame=cv2.threshold(delta_frame, 21, 255, cv2.THRESH_BINARY)[1]
    thresh_frame=cv2.dilate(thresh_frame, None, iterations=1)

    (cnts,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        status=1
        (x, y, w, h)=cv2.boundingRect(contour)
        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 3)
        text = "Occupied"


    # draw the text and timestamp on the frame
    ts = timestamp.strftime("%A %d %B %Y %I:%M:%S%p")
    cv2.putText(frame, "Room Status: {}".format(text), (10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, ts, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,0.35, (0, 0, 255), 1)
    
    status_list.append(status)

    status_list=status_list[-2:]

    
           
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())

    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(frame,1.2,5)
    

    
    if len(faces)  == 0:
        for (x,y,w,h) in faces:
            cv2.putText(frame, "Processing", (x,y-10), cv2.FONT_HERSHEY_COMPLEX,1,(255,30,0),2,cv2.LINE_AA)

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

  
    label = []
    for face in faces:
        x,y,w, h = face
        
        offset = 10
        face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
        
        if(np.all(np.array(face_section.shape))):
            face_section = cv2.resize(face_section,(48,48))

            pred = np.argmax(model.predict(face_section.reshape(1,48,48,1)))
            label = class_to_label[pred]

            cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_COMPLEX,1,(255,30,0),2,cv2.LINE_AA)
            cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,255),2)

    
    # cv2.imshow("Emotion", frame)
    

    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",thresh_frame)
    cv2.imshow("Color Frame",frame)

    key=cv2.waitKey(1)

    if key==ord('q'):
        if status==1:
            times.append(datetime.now())
        break
#Taking the image of the frame-----------------------------------------------------------------------
    while(True):
        if(text=="Occupied" and currentframe%200==0):
            name = 'images/'+str(currentframe) + '.jpg'
            cv2.imwrite(name,frame)
            currentframe += 1
            emotions = str(label)
            caption = images.caption_this_image(name) + "\n" +"person in image seems : " + emotions + "."
            
            bot2.tasveer(name,caption)
            break
        else:
            currentframe+=1
            break

print(currentframe)
video.release()
cv2.destroyAllWindows
