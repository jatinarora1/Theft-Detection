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
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
import imutils
import time
from imutils.video import VideoStream
import os
model = load_model("models/best_model.h5")
class_to_label = {0 :'Angry', 1 : 'Disgust', 2:'Fear', 3 :'Happy', 4:'Sad', 5:'Surprise', 6:'Neutral'}

face_cascade = cv2.CascadeClassifier("facial_expression/haarcascade_frontalface_default.xml")
first_frame=None
status_list=[None,None]
times=[]
chat_id = 1213182814
currentframe=0
label1 = ''
label2 = ''
def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        preds = maskNet.predict(faces)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


# load our serialized face detector model from disk
print("[INFO] loading face detector model...")
prototxtPath = 'face_detector/deploy.prototxt'
weightsPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
print("[INFO] loading face mask detector model...")
maskNet = tf.keras.models.load_model("model.h5")

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
time.sleep(2.0)

frame = 0

video  = cv2.VideoCapture(0)
ret,frame1 = video.read()
ret,frame2 = video.read()
print("Enter the type of mode: ")
mode = input()
while video.isOpened():
    text = 'Unoccupied'
    status=1
    timestamp = datetime.now()
    if mode == "night":
    
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


        
        for face in faces:
            x,y,w, h = face
            
            offset = 10
            face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
            
            if(np.all(np.array(face_section.shape))):
                face_section = cv2.resize(face_section,(48,48))

                pred = np.argmax(model.predict(face_section.reshape(1,48,48,1)))
                label1 = class_to_label[pred]

                cv2.putText(frame2, label1, (x,y-10), cv2.FONT_HERSHEY_COMPLEX,1,(255,30,0),2,cv2.LINE_AA)
                cv2.rectangle(frame2,(x,y),(x+w,y+h), (0,255,255),2)
        
        frame = imutils.resize(frame2, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label2 = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label2 == "Mask" else (0, 0, 255)

            # include the probability in the label
            label2 = "{}: {:.2f}%".format(label2, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label2, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)


        cv2.imshow("color",frame2)
        cv2.imshow("Guassian Blur",blur)
        cv2.imshow("dilate",dilated)
        cv2.imshow("Mask/NoMask",frame)
    
    else:
        gray_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(frame2,1.2,5)



        if len(faces)  == 0:
            for (x,y,w,h) in faces:
                cv2.putText(frame2, "Processing", (x,y-10), cv2.FONT_HERSHEY_COMPLEX,1,(255,30,0),2,cv2.LINE_AA)

                cv2.rectangle(frame2,(x,y),(x+w,y+h),(255,0,0),2)


        label1 = []
        for face in faces:
            x,y,w, h = face
            
            offset = 10
            face_section = gray_frame[y-offset:y+h+offset,x-offset:x+w+offset]
            
            if(np.all(np.array(face_section.shape))):
                face_section = cv2.resize(face_section,(48,48))

                pred = np.argmax(model.predict(face_section.reshape(1,48,48,1)))
                label1 = class_to_label[pred]

                cv2.putText(frame2, label1, (x,y-10), cv2.FONT_HERSHEY_COMPLEX,1,(255,30,0),2,cv2.LINE_AA)
                cv2.rectangle(frame2,(x,y),(x+w,y+h), (0,255,255),2)
        label2 = []
        frame = imutils.resize(frame2, width=400)

        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label2 = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label2 == "Mask" else (0, 0, 255)

            # include the probability in the label
            label2 = "{}: {:.2f}%".format(label2, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label2, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.imshow("Mask/NoMask",frame)


    frame1 = frame2
    
    
    ret,frame2 = video.read()

    key=cv2.waitKey(1)

    if key & ord('q') == 0xFF:
        if status==1:
            times.append(datetime.now())
        break
        print(times)
#Taking the image of the frame-----------------------------------------------------------------------
    if mode == "night":
        if(text=="Occupied" and currentframe%200==0):
            name = 'images/'+str(currentframe) + '.jpg'
            
            cv2.imwrite(name,frame2)
            
            currentframe += 1
            emotions = str(label1)
            Mask = str(label2)

            caption = images.caption_this_image(name) + "\n" +"person in image seems : " + emotions +"."+ "\n"+Mask+"."
            print(caption)
            bot2.tasveer(name,caption)
            
        else:
            currentframe+=1
    else:
        if(currentframe%200 == 0):
            name = 'images/'+str(currentframe) + '.jpg'
            
            cv2.imwrite(name,frame2)
            
            currentframe += 1
            emotions = str(label1)
            Mask = str(label2)

            caption = images.caption_this_image(name) + "\n" +"person in image seems : " + emotions +"."+ "\n"+Mask+"."
            print(caption)
            bot2.tasveer(name,caption)
            
        else:
            currentframe += 1


print(currentframe)
video.release()
cv2.destroyAllWindows
