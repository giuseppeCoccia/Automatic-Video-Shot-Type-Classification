import sys
import os
import subprocess
import argparse


if(len(sys.argv) < 3):
	print("Usage: python3 parse_video_info.py path_to_video(no extention) mode")
	print("Mode: -gros_plan (face area > 0.25% of frame area)")
	print("Mode: -plan_moyen (face on the frame's head)")
	exit()

file_ = sys.argv[1]+"_faces.txt"
video_ = sys.argv[1]+".mp4"
mode = sys.argv[2]

#### UTILS FUNCTIONS
# base and high of the two images
def area_ratio(b1, h1, b2, h2):
	a1 = b1*h1
	a2 = b2*h2
	if(a1 == 0 or a2 == 0):
		return 0
	if(a1 > a2):
		return a2/a1
	return a1/a2


# coordinates is an array of 4 points: (x_top, y_top, x_bottom, y_bottom)
def headbody_ratio(coordinates):
	pass	


# coordinates is an array of 4 points: (x_top, y_top, x_bottom, y_bottom)
def on_head(coordinates, h, n_split=3, h_ratio_limit=0.7):
	lower_limit_h = h/n_split
	y1_down = coordinates[3]
	h_ratio = (coordinates[3]-coordinates[1])/(lower_limit_h)
	return (y1_down <= lower_limit_h and h_ratio >= h_ratio_limit)

# coordinates is an array of 4 points: (x_top, y_top, x_bottom, y_bottom)
# b and h are the dimensions of th original frame
# n_split is the number of "split" on which the frame will be divided into
# large_split is to decide if we want only the center split, or to esclude only the first and last
def on_center(coordinates, b, h, n_split=3, large_split=True, h_ratio_limit=0.4):
	if large_split:
		upper_limit_h = h/n_split
		lower_limit_h = h*(n_split-1)/n_split
	else:
		if(n_split % 2 == 0):
			mid = n_split / 2
			upper_limit_h = h*(mid-1)/n_split
			lower_limit_h = h*(mid+1)/n_split
		else:
			mid = n_split // 2
			upper_limit_h = h*mid/n_split
			lower_limit_h = h*(mid+1)/n_split
	h_ratio = (coordinates[3]-coordinates[1])/(lower_limit_h-upper_limit_h)
	return (coordinates[1] >= upper_limit_h and coordinates[3] <= lower_limit_h and h_ratio >= h_ratio_limit)



#### MAIN ###

# read video dimensions first
cmd_ = "./get_video_resolution.sh "+video_
p = subprocess.Popen(cmd_ , shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
stdout, stderr = p.communicate()
dim = stdout.strip().split("x") # dim[0] = b, dim[1] = h

# parse input file
frames = {}
with open(file_, "r") as f:
	for line in f:
		words = line.split()
		frame = int(words[0])
		x_top = int(words[1])
		#if(x_top < 0): x_top = 0
		y_top = int(words[2])
		#if(y_top < 0): y_top = 0
		x_down = int(words[3])
		#if(x_down > int(dim[0])): x_down = int(dim[0])
		y_down = int(words[4])
		#if(y_down > int(dim[1])): y_down = int(dim[1])
		
		if frame not in frames.keys():
			frames[frame] = [(x_top, y_top, x_down, y_down)]
		else:
			frames[frame].append((x_top, y_top, x_down, y_down))

old_key = -1
frms = {'plan_moyen':[], 'plan_rapproche':[], 'plan_americain':[], 'gros_plan':[], 'plan_large':[]}
for key, value in sorted(frames.items()):
	if(old_key == -1): old_key = key
	elif(key < old_key+20): continue
	if(len(value) == 1):
		coordinates = value[0]
		ratio = area_ratio(abs(coordinates[2]-coordinates[0]), abs(coordinates[1]-coordinates[3]), int(dim[0]), int(dim[1]))
		hb_ratio = headbody_ratio(coordinates)
		# GROS PLAN
		if ratio > 0.25: #and ratio < 0.35
			frms['gros_plan'].append(key)
		# PLAN MOYEN (0.006 0.01) -> 0 < h < 154
		elif ratio > 0.006 and ratio < 0.007 and coordinates[3] < 384-230 and coordinates[3] > 384-340:
			frms['plan_moyen'].append(key)
		# PLAN RAPPROCHE (0.03 and 0.08) -> 164 < h < 254
		elif ratio > 0.05 and ratio < 0.06 and coordinates[3] < 384-130 and coordinates[3] > 384-220:
			frms['plan_rapproche'].append(key)
		# PLAN AMERICAIN (0.01 and 0.03) -> 
		elif ratio > 0.02 and ratio < 0.03 and coordinates[3] < 384-160:# and coordinates[3] > 384-220:
			frms['plan_americain'].append(key)
		# PLAN LARGE (0 and 0.006)
		elif ratio < 0.006 and ratio >= 0:
			frms['plan_large'].append(key)
		else: continue #if not chosen, do not update key
	else:
		coordinates = value[0]
		ratio = area_ratio(abs(coordinates[2]-coordinates[0]), abs(coordinates[1]-coordinates[3]), int(dim[0]), int(dim[1]))
		hb_ratio = headbody_ratio(coordinates)
		if ratio >= 0 and ratio < 0.006:
			frms['plan_larg'].append(key)
	old_key = key
print(" ".join(str(x) for x in frms[mode]))
