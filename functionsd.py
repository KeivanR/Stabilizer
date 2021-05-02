import cv2
import sys
import numpy as np
def stab(frame_piece, old_piece, lk_params):
# Create some random colors
	color = np.random.randint(0,255,(200,3))
	old_gray = cv2.cvtColor(old_piece, cv2.COLOR_BGR2GRAY)
	p0 = None
	qlevel = 1
	while p0 is None or len(p0)<1:
		qlevel -= 0.1
		# params for ShiTomasi corner detection
		feature_params = dict( maxCorners = 100,qualityLevel = qlevel, minDistance = 7, blockSize = 7 )
		p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
	#print(len(p0))
	#print(qlevel)
	# Create a mask image for drawing purposes
	mask = np.zeros_like(old_piece)
	 
	frame_gray = cv2.cvtColor(frame_piece, cv2.COLOR_BGR2GRAY)
	
	

	# calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

	# Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]

	# draw the tracks
	tx = 0
	ty = 0 
	for i,(new,old) in enumerate(zip(good_new,good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
		frame_piece = cv2.circle(frame_piece,(c,d),1,color[i].tolist(),-1)
		frame_piece = cv2.circle(frame_piece,(a,b),1,color[2*i].tolist(),-1)
	tx += c-a
	ty += d-b
	
   # img = cv2.add(frame,mask)
	img = frame_piece
	tx = 1*tx/(i+1)
	ty = 1*ty/(i+1)
	num_rows, num_cols = img.shape[:2]
	translation_matrix = np.float32([ [1,0,tx], [0,1,ty] ])
	img = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
	return (img)