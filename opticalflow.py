import numpy as np
import cv2
import sys

cap = cv2.VideoCapture(str(sys.argv[1]))
frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))

frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT)) 
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 30.0, (frame_width,frame_height))

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.2,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
print(len(p0))
# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(cap.isOpened()):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    

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
        
	tx += c-a
	ty += d-b
	
   # img = cv2.add(frame,mask)
    img = frame
    tx = tx/(i+1)
    ty = ty/(i+1)
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([ [1,0,tx], [0,1,ty] ])
    img = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))
    img = cv2.circle(img,(int(frame_width/2),int(frame_height/2)),5,color[i-1].tolist(),-1)
    img = cv2.circle(img,(int(frame_width/2+tx),int(frame_height/2+ty)),5,color[i].tolist(),-1)
    out.write(img)
    cv2.imshow('frame',img)
    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break
#test
    # Now update the previous frame and previous points
    #old_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).copy()
    #p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
out.release()
