import sys
import os
import subprocess

file_ = sys.argv[1]+"_faces.txt"
video_ = sys.argv[1]+"_faces.MP4"


# base and high of the two images
def get_ratio(b1, h1, b2, h2):
	a1 = b1*h1
	a2 = b2*h2

	if(a1 > a2):
		return a2/a1
	return a1/a2

# not working with cv2
# use extract_frames.sh script instead
def extract_frames(frame_string):
	#cmd_ = 'ffmpeg -i '+video_+' -vf select=eq(n\,'+frame_string+')'+  ' -vsync 0 "/datas/teaching/projects/spring2018/ps34/Data/Videos/extracted_frames/'+sys.argv[1].split("/")[-1]+'_'+frame_string+'.jpg"' 
	#print("CMD:", cmd_)
	#string = ''
	#for frame in frames:
#		string += ' '+str(frame)
	cmd_ = './extract_frames.sh '+video_+' '+frame_string
	#print(cmd_)
	#os.system(cmd_)
	p = subprocess.check_call(cmd_, shell=True)
	#p.wait()
	#p = subprocess.Popen(cmd_ , shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	#stdout, stderr = p.communicate()
	return



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
		ratio = get_ratio(abs(x_down-x_top), abs(y_top-y_down), int(dim[0]), int(dim[1]))
		if frame not in frames.keys():
			frames[frame] = [1, [ratio]]
		else:
			frames[frame][0] += 1
			frames[frame][1].append(ratio)
	ratios = []
	#frame_string = ""
	frms = []
	for key, elem in frames.items():
		if elem[0] == 1 and elem[1][0] > 0.25:
			ratios.append(elem[1][0])
			frms.append(key)
			#extract_frames(key)
			#frame_string += "eq(n\,"+key+")+" 	
	#print("Mean:", sum(ratios)/len(ratios))
	print(" ".join(str(x) for x in frms))
	#ratios.sort()
	#print(ratios)	
	#print("Frame String:", frame_string[:-1])
	#extract_frames(frame_string[:-1])
	#extract_frames(frms)

