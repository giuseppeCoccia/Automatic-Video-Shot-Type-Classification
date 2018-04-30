import sys
import csv
from os import listdir
from os.path import isfile, join


class ImgInfo:
	def __init__(self, id, video, doc_start, doc_end, title):
		self.id, self.video, self.doc_start, self.doc_end, self.title, self.frames = id, video, doc_start, doc_end, title, []
	
	def add_frame(self, frame):
		self.frames.append(frame)

if(len(sys.argv) != 4):
	print("Usage: python3 _.py csvpath image_plan_folder path_to_video_directory (No slash at the end of paths)")
	exit(1)

filename = sys.argv[1]

infos = {}
with open(filename, 'r') as csvfile:
	reader = csv.reader(csvfile, delimiter=';', quotechar='|')
	next(reader, None)  # skip the header
	for row in reader:
		id_notice = row[0]
		video = row[2]
		doc_start = row[3]
		doc_end = row[4]
		title = row[6]
		imginfo = ImgInfo(id_notice, video, doc_start, doc_end, title)
		infos[id_notice] = imginfo

image_plan_folder = sys.argv[2]
images_names = [name for name in listdir(image_plan_folder) if isfile(join(image_plan_folder, name))]

for file_name in images_names:
	id_image = file_name.split("_")[0]
	print(file_name)
	time = file_name.split("_")[2].split(".")[0].replace("'", "")
	doc_start = infos[id_image].doc_start
	frame_start = int(time)+int(doc_start)
	if frame_start % 100 >= 60:
		frame_start += 40
	seconds = (frame_start // 100)*60 + (frame_start % 100) 
	frame_number = seconds * 25
	for i in range(25):
		infos[id_image].add_frame(frame_number+i)
	print(id_image, time, doc_start, frame_start, seconds, frame_number)

videos_path = sys.argv[3]
for documentary in infos.values():
	text_file_name = videos_path + "/" + documentary.video + "_faces.txt"
	lines = []

	# parse input file
	with open(text_file_name, "r") as f:
		for line in f:
			words = line.split()
			frame = words[0]
			x_top = int(words[1])
			y_top = int(words[2])
			x_down = int(words[3])
			y_down = int(words[4])
			if int(frame) in documentary.frames:
				lines.append(line)
	
	print(len(lines))	
	with open(videos_path + "/" + documentary.video + "_frames_" + image_plan_folder.split("/")[-1] + ".txt", "a+") as f:
		for line in lines:
			f.write(line)
	
