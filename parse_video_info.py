import sys
import os
import cv2
import subprocess

file_ = sys.argv[1]+".txt"
video_ = sys.argv[1]+".mp4"


# base and high of the two images
def get_ratio(b1, h1, b2, h2):
	a1 = b1*h1
	a2 = b2*h2

	if(a1 > a2):
		return a2/a1
	return a1/a2

# not working with cv2
# use extract_frames.sh script instead
def extract_frame(frame_n):
	#vidcap = cv2.VideoCapture(video_)
	#print(video_)
	#success,image = vidcap.read()
	#count = 0
	#while success and count < frame_n:
  	#	success,image = vidcap.read()
  	#	print('Read a new frame: ', success)
  	#	count += 1
	#if(success):
  	#	cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
	#return image
	pass	




# main

# read video dimensions first
cmd_ = "./get_video_resolution.sh "+video_
p = subprocess.Popen(cmd_ , shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = p.communicate()
dim = stdout.strip().split("x") # dim[0] = b, dim[1] = h

# parse text file
frames = {}
with open(file_, "r") as f:
	for line in f:
		words = line.split()
		
		frame = words[0]
		x_top = int(words[1])
		y_top = int(words[2])
		x_down = int(words[3])
		y_down = int(words[4])
		
		ratio = get_ratio(abs(x_top-x_down), (y_top-y_down), dim[0], dim[1])
		if(ratio > .8):
			frames[frame] += 1
		
	
