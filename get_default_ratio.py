import sys

with open(sys.argv[1], 'r') as f:
	area = 0
	count = 0
	height = 0
	max_area = -1
	min_area = 512*384
	max_height = -1
	min_height = 384
	for line in f:
		line = line.split()
		x_top = int(line[1])
		if(x_top < 0): x_top = 0
		y_top = int(line[2])
		if(y_top < 0): y_top = 0
		x_down = int(line[3])
		if(x_down > 512): x_down = 512
		y_down = int(line[4])
		if(y_down > 384): y_down = 384

		new_area = abs(x_top - x_down)*abs(y_top-y_down)
		area += new_area
		new_height = 384-y_down
		height += new_height
		count += 1

		new_area /= 512*384 # im interested in the ratio
		
		if(new_area > max_area):
			max_area = new_area
		if(new_area < min_area):
			min_area = new_area
		if(new_height > max_height):
			max_height = new_height
		if(new_height < min_height):
			min_height = new_height
	mean_area = area/count
	mean_height = height/count
	print(sys.argv[1])
	print("Min Height:", min_height, "Max height:", max_height)
	print("Min Area:", min_area, "Max Area:", max_area)
	print("MeanArea:", mean_area/(512*384), "Height:", mean_height)
