from tkinter import *
import tkinter.filedialog
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.widgets as widgets
import numpy as np
import cv2
from functions import *

root = Tk()
root.withdraw()

filename = tkinter.filedialog.askopenfilename(initialdir='C:/Users/Diana/Desktop/Data Analysis/',title='Open video for stabilization')
cap = cv2.VideoCapture(filename)
ret,frame = cap.read()

x1,x2,y1,y2=0,0,0,0

def onselect(eclick, erelease):
	global x1,x2,y1,y2
	if eclick.ydata>erelease.ydata:
		eclick.ydata,erelease.ydata=erelease.ydata,eclick.ydata
	if eclick.xdata>erelease.xdata:
		eclick.xdata,erelease.xdata=erelease.xdata,eclick.xdata
	x1,y1,x2,y2=eclick.xdata,eclick.ydata,erelease.xdata,erelease.ydata


def handle_close(evt):
	print(x1,x2,y1,y2)
	print('Closed Figure!')
	stab_video(cap,x1,x2,y1,y2)
	

fig = plt.figure()
ax = fig.add_subplot(111)
fig.canvas.mpl_connect('close_event', handle_close)
#im = Image.open(filename)
arr = np.asarray(frame)
plt_image=plt.imshow(arr)
rs=widgets.RectangleSelector(
    ax, onselect, drawtype='box',
    rectprops = dict(facecolor='red', edgecolor = 'black', alpha=0.5, fill=True))
plt.show()


