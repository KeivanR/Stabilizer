import numpy as np
import cv2
import sys
def stab_frame(frame_piece, old_piece, lk_params,x1,x2,y1,y2,old_tx,old_ty,out_trans):
	# Create some random colors
	color = np.random.randint(0,255,(200,3))
	old_gray = cv2.cvtColor(old_piece, cv2.COLOR_BGR2GRAY)
	p0 = None
	min_features = 1
	qlevel = 1
	#p0 is for the old frame. If oldframe is the saame (as in the first frame), then it is useless to recalculate p0 all the time
	while p0 is None or len(p0)<min_features:
		qlevel -= 0.1
		# params for ShiTomasi corner detection
		feature_params = dict( maxCorners = 100,qualityLevel = qlevel, minDistance = 15, blockSize = 5 )
		p0 = cv2.goodFeaturesToTrack(old_gray[int(y1):int(y2),int(x1):int(x2)], mask = None, **feature_params)
	# Create a mask image for drawing purposes
	mask = np.zeros_like(old_piece)
	 
	frame_gray = cv2.cvtColor(frame_piece, cv2.COLOR_BGR2GRAY)
	
	frame_piece2 = np.copy(frame_piece)

	# calculate optical flow
	p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray[int(y1):int(y2),int(x1):int(x2)], frame_gray[int(y1):int(y2),int(x1):int(x2)], p0, None, **lk_params)
	# Select good points
	good_new = p1[st==1]
	good_old = p0[st==1]

	# draw the tracks
	tx = 0
	ty = 0 
	#print(good_new,good_old)
	for i,(new,old) in enumerate(zip(good_new,good_old)):
		a,b = new.ravel()
		c,d = old.ravel()
		mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
		frame_piece = cv2.circle(frame_piece,(int(c+x1),int(d+y1)),1,color[i].tolist(),-1)
		frame_piece = cv2.circle(frame_piece,(int(a+x1),int(b+y1)),1,color[2*i].tolist(),-1)
		tx += c-a
		ty += d-b
	
   # img = cv2.add(frame,mask)
	img = frame_piece
	tx = 1*tx/(i+1)
	ty = 1*ty/(i+1)
	out_trans.write(str(tx)+' '+str(ty))
	out_trans.write("\n")
	num_rows, num_cols = img.shape[:2]
	translation_matrix = np.float32([ [1,0,int(tx+old_tx)], [0,1,int(ty+old_ty)] ])
	img = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
	img2 = cv2.warpAffine(frame_piece2, translation_matrix, (num_cols, num_rows))
	old_tx += tx
	old_ty += ty
	return {'img':img, 'img2':img2, 'old_tx':old_tx,'old_ty':old_ty}
	
def stab_video(cap,x1,x2,y1,y2,filename,method = 'frame_to_frame',window_frac = 1,time_frame0 = 0):
	frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT)) 
	frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
	fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
	x1 = int(0+window_frac*frame_width)
	x2 = int(frame_width-window_frac*frame_width)
	y1 = int(0+window_frac*frame_height)
	y2 = int(frame_height-window_frac*frame_height)
	out = cv2.VideoWriter('C:/Users/Diana/Desktop/Data Analysis/'+filename.split('/')[-1].split('.')[0]+'_stab.avi',fourcc, frame_rate, (frame_width,frame_height))
	out2 = cv2.VideoWriter('C:/Users/Diana/Desktop/Data Analysis/'+filename.split('/')[-1].split('.')[0]+'_stabcomb.avi',fourcc, frame_rate, (frame_width,2*frame_height))
	# Take first frame and find corners in it
	ret, old_frame = cap.read()
	old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
	# Parameters for lucas kanade optical flow
	lk_params = dict( winSize  = (frame_width,frame_height),maxLevel = 0, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10000, 0.03))

		


		#img = cv2.circle(img,(int(frame_width/2),int(frame_height/2)),5,color[i-1].tolist(),-1)
		#img = cv2.circle(img,(int(frame_width/2+tx),int(frame_height/2+ty)),5,color[i].tolist(),-1)


	i = 0
	old_tx = 0
	old_ty = 0
	mean_tx = 0
	mean_ty = 0
	txs = []
	tys = []
	out_trans = open('C:/Users/Diana/Desktop/Data Analysis/'+filename.split('/')[-1].split('.')[0]+'_trans.txt', "w")
	if method == 'from_maxframe':
		old_frame = max_frame(get_frames(filename))['frame']
	if method == 'from_timeframe0':
		frames = get_frames(filename)
		old_frame = frames[int(time_frame0*frame_rate)]
		print('frame0 :',int(time_frame0*frame_rate))
	while(cap.isOpened()):
		ret,frame = cap.read()
		if frame is None:
			break
		sf = stab_frame(frame,old_frame,lk_params,x1,x2,y1,y2,old_tx,old_ty,out_trans)
		mean_tx+=sf['old_tx']
		mean_ty+=sf['old_ty']
		txs.append(sf['old_tx'])
		tys.append(sf['old_ty'])
		if method == 'frame_to_frame':
			old_tx = sf['old_tx']
			old_ty = sf['old_ty']
			old_frame = frame
		img = sf['img']
		img2 = sf['img2']
		combined = np.concatenate((frame,img), axis=0)
		i = i+1
		out.write(img2)
		out2.write(combined)
		cv2.imshow('frame',combined)
		k = cv2.waitKey(30) & 0xff
		if k == 27:
			break
	#test2
		# Now update the previous frame and previous points
		#old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).copy()
		#p0 = good_new.reshape(-1,1,2)
	mean_tx = mean_tx/len(txs)
	mean_ty = mean_ty/len(tys)
	index = np.argmin(np.abs(np.array(txs)**2+np.array(tys)**2-mean_tx**2-mean_ty**2))
	print('time for next frame: ',index/frame_rate)
	out_trans.close()
	cap.release()
	out.release()
	cv2.destroyAllWindows()
def get_frames(neuron):
	cap = cv2.VideoCapture(neuron)
	frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	frame_height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
	nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frames = []
	while(cap.isOpened()):
		
		ret, frame = cap.read()
		try: 
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		except:
			break
		
		frames.append(frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	cap.release()
	cv2.destroyAllWindows()
	return(frames)

def max_frame(frames):
	max_brightness = 0
	max_index = 0
	for i in range(0,len(frames)):
		gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
		if max_brightness<gray.sum():
			max_brightness = gray.sum()
			max_index = i
	return {'frame':frames[max_index],'index':max_index}