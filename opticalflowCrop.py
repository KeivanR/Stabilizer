import numpy as np
import cv2
import sys
from functions import *
import gui
gui.plt.show()
cap = gui.cap
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT)) 
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('C:/Users/Diana/Desktop/Data Analysis/try.avi',fourcc, frame_rate, (frame_width,frame_height))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	


	#img = cv2.circle(img,(int(frame_width/2),int(frame_height/2)),5,color[i-1].tolist(),-1)
	#img = cv2.circle(img,(int(frame_width/2+tx),int(frame_height/2+ty)),5,color[i].tolist(),-1)



while(cap.isOpened()):
	ret,frame = cap.read()
	try:
		frame1 = frame[0:(int(frame_height/2)-1),]
		frame2 = frame[(int(frame_height/2)):frame_height,]
	except:
		break
	old_frame1 = old_frame[0:(int(frame_height/2)-1),]
	old_frame2 = old_frame[(int(frame_height/2)):frame_height,]

	img = frame

	img[0:(int(frame_height/2)-1),] = stab(frame1,old_frame1,lk_params)
	img[int(frame_height/2):frame_height,] = stab(frame2,old_frame2,lk_params)
	
	out.write(img)
	cv2.imshow('frame',img)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
#test2
    # Now update the previous frame and previous points
    #old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).copy()
    #p0 = good_new.reshape(-1,1,2)
cap.release()
out.release()
cv2.destroyAllWindows()

