import urllib.request
import cv2
url = 'http://10.78.95.57:8080/video'
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
      ret, frame = self.video.read()
      frame=cv2.resize(frame,None,fx=ds_factor,fy=ds_factor,
      interpolation=cv2.INTER_AREA)                    
      gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      # encode OpenCV raw frame to jpg and displaying it
      ret, jpeg = cv2.imencode('.jpg', gray)
      return jpeg.tobytes()
