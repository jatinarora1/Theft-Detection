
import cv2
import telegram
TOKEN = '1193023091:AAEl9eLOZ6Q0PdDRXF07TprHDXt9tEGuclo'
bot = telegram.Bot(TOKEN)
from activity_captioning import images 
from Telegram_Bot import bot2
from keras.layers import *
from keras.models import load_model
import matplotlib.pyplot as plt
video = cv2.VideoCapture("test_input/Little Boy Playing at Toys R Us Superstore Fun For Kids.mp4")
count = 0
model = load_model("models/best_model.h5")
chat_id = 1213182814
while(video.isOpened()):
	ret, frame = video.read()
	if ret == False:
		break

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	cv2.imshow('frame',gray)

	if count%500 == 0:
		name = 'images/kang'+str(count) + '.jpg'
		cv2.imwrite('images/kang'+str(count)+'.jpg',frame)
		count += 1
		caption = images.caption_this_image(name) + " ."
		print(caption)
		bot2.tasveer(name,caption)
	count+=1
	key=cv2.waitKey(1)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
video.release()
cv2.destroyAllWindows()
