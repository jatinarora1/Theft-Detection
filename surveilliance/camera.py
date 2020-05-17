import urllib.request
import cv2
import numpy as np
url = 0
ds_factor=0.6
cap = cv2.VideoCapture(url)
class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(url)
      
    def __del__(self):
        #releasing camera
        self.video.release()
    def get_frame(self):
      #extracting frames
      _, frame = self.video.read()
      frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,
      interpolation=cv2.INTER_AREA)                    
      gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      # encode OpenCV raw frame to jpg and displaying it

      blur=cv2.GaussianBlur(gray,(21,21),0)
      first_frame=None
      if first_frame is None:
          first_frame=gray
          

      delta_frame=cv2.absdiff(first_frame,blur)
      thresh_frame=cv2.threshold(delta_frame, 21, 255, cv2.THRESH_BINARY)[1]
      thresh_frame=cv2.dilate(thresh_frame, None, iterations=2)

      (_,_)=cv2.findContours(thresh_frame.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

      
      #     text = "Occupied"
      ret, jpeg= cv2.imencode('.jpg', gray)
      ret, jpeg2 = cv2.imencode('.jpg',frame)
      ret, jpeg3 = cv2.imencode('.jpg',blur)
      ret, jpeg4 = cv2.imencode('.jpg',thresh_frame)
      return jpeg.tobytes(),jpeg2.tobytes(),jpeg3.tobytes(),jpeg4.tobytes()
 