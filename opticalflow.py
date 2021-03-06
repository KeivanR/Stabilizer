import numpy as np
import cv2
import sys


cap = cv2.VideoCapture('C:/Users/Diana/Desktop/Data Analysis/l2fast.avi')
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT)) 
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('C:/Users/Diana/Desktop/Data Analysis/output28.avi',fourcc, frame_rate, (frame_width,frame_height))
out2 = cv2.VideoWriter('C:/Users/Diana/Desktop/Data Analysis/coutput28.avi',fourcc, frame_rate, (frame_width,2*frame_height))
p0 = None
qlevel = 1
while p0 is None or len(p0)<1:
	qlevel -= 0.1
	# params for ShiTomasi corner detection
	feature_params = dict( maxCorners = 100,
						   qualityLevel = qlevel,
						   minDistance = 7,
						   blockSize = 7 )

	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (15,15),
					  maxLevel = 2,
					  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

	# Create some random colors
	color = np.random.randint(0,255,(200,3))

	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print(len(p0))
print(qlevel)
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(cap.isOpened()):
	ret,frame = cap.read()
	
	try: 
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	except:
		break
    
    

    # calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]

    # draw the tracks
	tx = 0
	ty = 0 
	frame2=frame
	for i,(new,old) in enumerate(zip(good_new,good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
		#frame2 = cv2.circle(frame2,(c,d),10,color[i].tolist(),-1)
		#frame2 = cv2.circle(frame2,(a,b),10,color[2*i].tolist(),-1)
	tx += c-a
	ty += d-b
	
   # img = cv2.add(frame,mask)
	img = frame2
	tx = 1*tx/(i+1)
	ty = 1*ty/(i+1)
	num_rows, num_cols = img.shape[:2]
	translation_matrix = np.float32([ [1,0,tx], [0,1,ty] ])
	img = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
	combined = np.concatenate((frame,img), axis=0)
		
	out.write(img)
	out2.write(combined)
	cv2.imshow('frame',combined)
	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break
#test2
    # Now update the previous frame and previous points
    #old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).copy()
    #p0 = good_new.reshape(-1,1,2)
cap.release()
out.release()
out2.release()
cv2.destroyAllWindows()

