import urllib.request
import cv2
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
      _, frame1 = self.video.read()
      _, frame2 = self.video.read()
      frame1=cv2.resize(frame1,None,fx=ds_factor,fy=ds_factor,
      interpolation=cv2.INTER_AREA)
      frame2=cv2.resize(frame2,None,fx=ds_factor,fy=ds_factor,
      interpolation=cv2.INTER_AREA)

      diff = cv2.absdiff(frame1,frame2)
      gray = cv2.cvtColor(diff,cv2.COLOR_RGB2GRAY)
      blur = cv2.GaussianBlur(gray,(5,5),0)
      blurred = cv2.GaussianBlur(cv2.cvtColor(frame1,cv2.COLOR_RGB2GRAY),(5,5),0)
      _,thresh = cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
      dilated = cv2.dilate(thresh,None,iterations = 3)
      (cnts,_)=cv2.findContours(dilated.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      

      for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue

        
        (x, y, w, h)=cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), 2)
      
      frame1 = frame2
      #     text = "Occupied"
      ret, jpeg= cv2.imencode('.jpg', frame1)
      ret, jpeg2 = cv2.imencode('.jpg',gray)
      ret, jpeg3 = cv2.imencode('.jpg',blurred)
      ret, jpeg4 = cv2.imencode('.jpg',dilated)

      _,frame2 = self.video.read()
      return jpeg.tobytes(),jpeg2.tobytes(),jpeg3.tobytes(),jpeg4.tobytes()
 