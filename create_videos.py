import numpy as np
import cv2
import sys
def stripy_frame(w,s):
	frame = np.zeros((512,1024))
	for i in range(w):
		frame[i::s,] = 255
		
	return (frame)
def point_frame(w,x,y):
	frame = np.zeros((512,1024))
	for i in range(w):
		for j in range(w):
			frame[x+i,y+j] = 255
		
	return (frame)
def translate(old_frame,speed_vector):
	dX = old_frame.shape[0]
	dY = old_frame.shape[1]
	if (speed_vector[0]>0):
		frame = np.concatenate((old_frame[speed_vector[0]:dX,:],old_frame[0:(speed_vector[0]),:]), axis=0)
	else:
		frame = np.concatenate((old_frame[(dX+speed_vector[0]):dX,:],old_frame[0:(dX+speed_vector[0]),:]), axis=0)
	if (speed_vector[1]>0):
		frame = np.concatenate((frame[:,speed_vector[1]:dY],frame[:,0:(speed_vector[1])]), axis=1)
	else:
		frame = np.concatenate((frame[:,(dY+speed_vector[1]):dY],frame[:,0:(dY+speed_vector[1])]), axis=1)
	return (frame)
	
def animate(frame,func,speed_vector):
	frame_width = frame.shape[0]
	frame_height = frame.shape[1]
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('C:/Users/Diana/Desktop/Data Analysis/pattern.avi',fourcc, 20.0, (1024,512), False)
	t=0
	while(t<100):
		t+=1
		if (t%3==0):
			speed_vector[0] = -speed_vector[0]
			speed_vector[1] = -speed_vector[1]
		print(speed_vector)
		frame = func(frame,speed_vector)
		frame2 = np.uint8(frame)
		cv2.imshow('frame',frame2)
		out.write(frame2)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
	
	out.release()
	cv2.destroyAllWindows()	
	
animate(point_frame(20,200,500),translate,[5,5])

	
