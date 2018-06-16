import numpy as np
import os, sys
import cv2
import csv

# base directory for videos
#basedir = "../Data/Videos/"
basedir = "/Users/paolomoriello/Downloads/"

# csv
videoframes = {}
with open(sys.argv[1], 'r') as f:
	reader = csv.reader(f, delimiter =';')
	for row in reader:
		path = row[0]
		prediction = row[1]
		
		filename, fileextension = os.path.splitext(path)
		video = os.path.basename(filename).split('_')
		videoname = video[0]+".mp4"
		frame = int(video[1])

		if videoname in videoframes:
			videoframes[videoname][frame] = prediction
		else:
			videoframes[videoname] = {frame: prediction}

for k, v in videoframes.items():
	# video in
	inputfile = basedir+k
	vid_in = cv2.VideoCapture(inputfile)
	he = int(vid_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
	wi = int(vid_in.get(cv2.CAP_PROP_FRAME_WIDTH))
	vid_frames = int(vid_in.get(cv2.CAP_PROP_FRAME_COUNT))
	vid_fps = int(vid_in.get(cv2.CAP_PROP_FPS))
	print('width=', wi, ' height=', he, ' frames=', vid_frames, ' fps=', vid_fps)

	# video out
	filename, fileextension = os.path.splitext(inputfile)
	outputfile = filename+"_withsubs"+fileextension
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	vid_out = cv2.VideoWriter(outputfile,fourcc,vid_fps,(wi,he),1)

	# execution
	frame = 0
	frame_count = 25
	old_label = None
	while(vid_in.isOpened() & frame<vid_frames):
		ret, image = vid_in.read()
		if ret==True:
			if frame in v:
				print("Writing", v[frame], "on frame", frame, "for", k)
				font  = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(image, v[frame], (10, he-20), font, 1, (255,255,255), 2)
				frame_count = 0
				old_label = v[frame]
			# write old label for 25 frames
			elif frame_count < 25:
				cv2.putText(image, old_label, (10, he-20), font, 1, (255,255,255), 2)
				frame_count += 1
			vid_out.write(image)
			frame = frame+1
		else:
			break

	vid_out.release()
	vid_in.release()
